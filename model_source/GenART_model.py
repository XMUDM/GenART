""" PyTorch GenART model."""
import pdb
# torch
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN

# misc
import numpy as np
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from .GenART_config import GenARTConfig
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

logger = logging.get_logger(__name__)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HELPER FUNCTIONS AND CONSTANTS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class GenARTModelOutput(ModelOutput):
    nucleotide_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    nucleotide_attention_mask: Optional[torch.FloatTensor] = None
    token_hidden_states: torch.FloatTensor = None
    token_attention_mask: Optional[torch.FloatTensor] = None
    
@dataclass
class LanaguageModelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_isolated(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch_size,
    num_key_value_heads, seq_len, head_dim) to (batch_size, num_key_value_heads * n_rep, seq_len,
    """
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# BASIC BLOCK
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class GenARTLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
    
    def forward(self, hidden_states):
        return self.norm(hidden_states)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->GenART
class GenARTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class GenARTFlashAttention2(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. 
    """

    def __init__(self, config: GenARTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
            
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.projection_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.projection_k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = GenARTRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward method to compute multi-head attention output.

        Args:
            hidden_states (torch.Tensor): Input hidden states, shape `(batch_size, seq_len, hidden_size)`.
            attention_mask (torch.Tensor, optional): Attention mask, shape `(batch_size, seq_len)`,
                indicates which positions are padding elements, these positions will have attention scores set to negative infinity.
            position_ids (torch.LongTensor, optional): Position IDs, shape `(batch_size, seq_len)`,
                indicates the position of each token in the input sequence.

        Returns:
            torch.Tensor: Output tensor, shape `(batch_size, seq_len, hidden_size)`.
        """

        batch_size, seq_len, _ = hidden_states.size()

        # Linear projection for hidden states to get query, key, and value
        query_states = self.projection_q(hidden_states)
        key_states = self.projection_k(hidden_states)
        value_states = self.projection_v(hidden_states)

        # Reshape tensors to separate heads
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # Because the input can be padded, the absolute sequence length depends on the max position id.
        # Apply rotary position embedding (RoPE)
        rotary_seq_len = seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids[:,:rotary_seq_len])

        # If the number of key and value heads is less than the number of query heads, repeat key and value to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.projection_q.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            seq_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size).contiguous()

        attn_output = self.projection_o(attn_output)

        return attn_output

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        seq_len_q,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states,indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask)
            key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(key_states, attention_mask)
            value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v = unpad_input(value_states, attention_mask)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=False,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)
        else:

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=False,
            )

        return attn_output


        
class GenARTFlashCrossAttention2(nn.Module):
    """
    Cross attention mechanism based on Flash Attention, used in GenART model.
    """
    def __init__(self, config: GenARTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads # Dimension per attention head
        self.num_key_value_heads = config.num_key_value_heads # Number of key and value attention heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # Number of key and value groups
        self.max_position_embeddings = config.max_position_embeddings # Maximum position embeddings
        self.rope_theta = config.rope_theta # RoPE (rotary position embedding) base value
        self.attention_dropout = config.attention_dropout # Attention dropout probability

        # Check if hidden_size can be divided by num_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # Define linear projection layers for query, key, value, and output
        self.projection_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False) 
        self.projection_k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Define rotary position embedding (RoPE)
        self.rotary_emb = GenARTRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    def forward(
        self,
        hidden_states_q: torch.Tensor,
        hidden_states_kv: torch.Tensor,
        attention_mask_q: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        
        """
        Forward method to compute cross attention output.

        Args:
            hidden_states_q (torch.Tensor): Query hidden states, shape `(batch_size, seq_len_q, hidden_size)`.
            hidden_states_kv (torch.Tensor): Key and value hidden states, shape `(batch_size, seq_len_kv, hidden_size)`.
            attention_mask_query (torch.Tensor, optional): Query attention mask, shape `(batch_size, seq_len_q)`.
            attention_mask_kv (torch.Tensor, optional): Key and value attention mask, shape `(batch_size, seq_len_kv)`.
            position_ids (torch.LongTensor, optional): Position IDs, shape `(batch_size, seq_len)`,
                indicates the position of each token in the input sequence.

        Returns:
            torch.Tensor: Output tensor, shape `(batch_size, seq_len_q, hidden_size)`.
        """
        batch_size, seq_len_q, _ = hidden_states_q.size()   
        batch_size, seq_len_kv, _ = hidden_states_kv.size()  
        
        query_states = self.projection_q(hidden_states_q)
        key_states = self.projection_k(hidden_states_kv)
        value_states = self.projection_v(hidden_states_kv)  

        # Reshape tensors to separate heads
        query_states = query_states.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        # Apply rotary position embedding (RoPE)
        rotary_seqeuence_length_q = seq_len_q
        cos_q, sin_q = self.rotary_emb(query_states, seq_len=rotary_seqeuence_length_q)
        
        rotary_seq_len_kv = seq_len_kv
        cos_kv, sin_kv = self.rotary_emb(value_states, seq_len=rotary_seq_len_kv)

        query_states = apply_rotary_pos_emb_isolated(query_states, cos_q, sin_q, position_ids[:,:rotary_seqeuence_length_q])
        key_states = apply_rotary_pos_emb_isolated(key_states, cos_kv, sin_kv, position_ids[:,:rotary_seq_len_kv])


       # If the number of key and value heads is less than the number of query heads, repeat key and value to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # Check if input is float32, if so convert to float16 to ensure correct computation
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.projection_q.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape tensors to fit Flash Attention expected input
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask_kv,
            attention_mask_q,
            seq_len_kv,
            seq_len_q,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(batch_size,seq_len_q, self.hidden_size).contiguous()
        
        attn_output = self.projection_o(attn_output)

        return attn_output
    
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask_kv,
        attention_mask_q,
        seq_len_kv,
        seq_len_q,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Call Flash Attention forward method to compute cross attention output.

        Args:
            query_states: Query tensor.
            key_states: Key tensor.
            value_states: Value tensor.
            attention_mask_kv: Key and value attention mask.
            attention_mask_q: Query attention mask.
            seq_len_kv: Key and value sequence length.
            seq_len_q: Query sequence length.
            dropout: Dropout probability.
            softmax_scale: Softmax scaling factor.

        Returns:
            attn_output: Attention output tensor.
        """
        
        # Contains at least one padding token in the sequence
        if attention_mask_kv is None and attention_mask_q is None:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=False
            )
        else:
            if attention_mask_kv is None:
                attention_mask_kv = torch.ones(query_states.shape[0], seq_len_kv).to(query_states.device)
            if attention_mask_q is None:
                attention_mask_q = torch.ones(query_states.shape[0], seq_len_q).to(query_states.device)
            batch_size = query_states.shape[0]
            query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask_q)
            key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(key_states, attention_mask_kv)
            value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v = unpad_input(value_states, attention_mask_kv)
        
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=False
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)
        
        return attn_output


class GenARTFFNBlock(nn.Module):
    def __init__(self, config: GenARTConfig):
        super().__init__()
        self.ffn = GenARTMLP(config)
        
    def forward(self, hidden_states,attention_mask = None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        current_hidden_states = self.ffn(hidden_states)
        if attention_mask is not None:
            current_hidden_states = pad_input(current_hidden_states, indices, batch_size, seq_len)
        return current_hidden_states

class GenARTMLP(nn.Module):
    def __init__(self, config: GenARTConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size


        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class LMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL LAYER
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class GenARTTokenizationLayer(nn.Module):
    """
    Tokenization Layer implemented with attention mechanism and Gumbel Softmax.
    Modifications:
    1. Each aggregated token representation is divided by its corresponding length.
    2. Tokens with special_tokens_mask=1 do not participate in summation, i.e., merge_decisions=0.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size  
        self.layer_idx = layer_idx  

        # MLP module for generating merge scores
        self.merge_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )
        self.temperature = 1.0
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Padding parameters for each convolution kernel
        self.padding_params = [
            ((kernel_size - 1) // 2, kernel_size // 2)
            for kernel_size in range(2, 7)
        ]
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=kernel_size,
                padding=0
            ) for kernel_size in range(2, 7)
        ])
        # Initialize weights for each CNN
        for conv in self.convs:
            torch.nn.init.trunc_normal_(conv.weight, mean=0.0, std=config.initializer_range)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def manual_pad(self, x: torch.Tensor, padding: tuple) -> torch.Tensor:
        """Manually implement asymmetric padding
        Args:
            x: input tensor [batch, channels, seq_len]
            padding: (left_pad, right_pad)
        Returns:
            padded tensor [batch, channels, padded_seq_len]
        """
        return F.pad(x, pad=(padding[0], padding[1]), mode='constant', value=0)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor, **kwargs):
        """
        Forward pass:
          1. Layer normalization on input, then use MLP to get merge scores for each token.
          2. Calculate dynamic threshold and binarize merge scores to get merge_decisions.
          3. Use merge_decisions and special_tokens_mask to compute group IDs for each token.
          4. Use scatter_add to sum hidden_states within each group and divide by group length.
        
        Returns:
          new hidden_states, attention_mask, and special_tokens_mask
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        special_mask = special_tokens_mask.unsqueeze(-1) 

        conv_outputs = []
        for i, (conv, padding) in enumerate(zip(self.convs, self.padding_params)):
            hidden_states_cnn = hidden_states.transpose(1, 2)
            # Manually apply asymmetric padding
            padded_input = self.manual_pad(hidden_states_cnn, padding)
            conv_out = conv(padded_input)
            
            conv_outputs.append(conv_out.transpose(1, 2))
        
        cnn_hidden_states = torch.stack(conv_outputs).mean(dim=0)

        # Residual connection to enhance information flow
        cnn_hidden_states = hidden_states + cnn_hidden_states 
        cnn_hidden_states = torch.where(special_mask.bool(), cnn_hidden_states, hidden_states)
        
        hidden_states_norm = self.input_layernorm(cnn_hidden_states)

        # Convert scores to two-class logits (merge vs not merge)
        merge_scores = self.merge_mlp(hidden_states_norm)  # (B, seq_len, 1)
        merge_scores = merge_scores.squeeze(-1)  # (B, seq_len)
        merge_scores = merge_scores[:, :-1] + merge_scores[:, 1:]  # (B, seq_len-1)
        logits = torch.stack([-merge_scores, merge_scores], dim=-1)

        # Use Gumbel Softmax to generate binary decisions
        merge_decisions = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        merge_decisions = merge_decisions[..., 1]  # (B, seq_len-1)
        
        # Special tokens do not participate in merging, set corresponding merge_decisions to 0
        special_tokens_mask_shifted = special_tokens_mask[:, 1:]  # Align with merge_decisions
        merge_decisions = merge_decisions * (1 - special_tokens_mask_shifted)
        

        group_start = torch.cat(
            [torch.ones(batch_size, 1, device=hidden_states.device, dtype=torch.long), 
            (1 - merge_decisions[:, :-1]).long(),
            torch.zeros(batch_size, 1, device=hidden_states.device, dtype=torch.long)],
            dim=1
        )
        group_ids = torch.cumsum(group_start, dim=1) - 1  

        max_groups = int(group_ids.max().item()) + 1

        new_hidden_states = torch.zeros(batch_size, max_groups, hidden_dim, device=hidden_states.device)

        mask_group_1 = (group_start == 1).unsqueeze(-1).expand(-1, -1, hidden_dim)

        hidden_states_group_1 = cnn_hidden_states * mask_group_1

        merge_decisions_padded = torch.cat([torch.zeros(batch_size, 1, device=hidden_states.device), merge_decisions], dim=1)
        merge_decisions_padded = merge_decisions_padded.unsqueeze(-1).expand(-1, -1, hidden_dim)

        # Special tokens do not participate in weighting
        merge_decisions_padded = merge_decisions_padded * (1 - special_tokens_mask.unsqueeze(-1))
        hidden_states_weighted = cnn_hidden_states * merge_decisions_padded

        new_hidden_states = new_hidden_states.scatter_add(1, group_ids.unsqueeze(-1).expand(-1, -1, hidden_dim), hidden_states_group_1)
        new_hidden_states = new_hidden_states.scatter_add(1, group_ids.unsqueeze(-1).expand(-1, -1, hidden_dim), hidden_states_weighted)

        group_lengths = torch.zeros(batch_size, max_groups, device=hidden_states.device)
        group_lengths = group_lengths.scatter_add(1, group_ids, torch.ones_like(group_ids).float())
        group_lengths = torch.clamp(group_lengths, min=1.0) 

        # Normalize hidden_states for each group
        new_hidden_states = new_hidden_states / group_lengths.unsqueeze(-1)

        new_attention_mask = torch.zeros(batch_size, max_groups, device=attention_mask.device)
        new_attention_mask = new_attention_mask.scatter_add(1, group_ids, attention_mask.float())
        new_attention_mask = (new_attention_mask > 0).long()

        new_special_tokens_mask = torch.zeros(batch_size, max_groups, device=special_tokens_mask.device)
        new_special_tokens_mask = new_special_tokens_mask.scatter_add(1, group_ids, special_tokens_mask.float())
        new_special_tokens_mask = (new_special_tokens_mask > 0).long()

        return new_hidden_states, new_attention_mask, new_special_tokens_mask
    

class GenARTEncoderLayer(nn.Module):
    def __init__(self, config: GenARTConfig, layer_idx: int):
        """
        Initialization method for creating components of GenART encoder layer.
        
        Args:
            config (GenARTConfig): Model configuration object, such as hidden_size, layer_norm_eps, etc.
            layer_idx (int): Index of the current layer, used to identify its position in the model.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GenARTFlashAttention2(config, layer_idx) 
        self.ffn = GenARTFFNBlock(config) 
        self.input_layernorm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Post-attention layer normalization module


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward method to compute output of GenART encoder layer.
        
        Args:
            hidden_states (torch.FloatTensor): Input tensor to the layer, shape `(batch_size, seq_len, hidden_size)`.
            attention_mask (torch.FloatTensor, optional): Attention mask, shape `(batch_size, seq_len)`,
                indicates which positions are padding elements, these positions will have attention scores set to negative infinity.
            position_ids (torch.LongTensor, optional): Position IDs, shape `(batch_size, seq_len)`,
                indicates the position of each token in the input sequence.
        
        Returns:
            torch.FloatTensor: Output tensor, shape `(batch_size, seq_len, hidden_size)`.
        """

        residual = hidden_states 
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states,attention_mask)
        hidden_states = residual + hidden_states

        return hidden_states
    
class GenARTDecoderLayer(nn.Module):
    def __init__(self, config: GenARTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = GenARTFlashCrossAttention2(config, layer_idx)
        self.ffn = GenARTFFNBlock(config)
        self.input_layernorm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states_q: torch.Tensor,
        hidden_states_kv: torch.Tensor,
        attention_mask_q: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
            
        residual = hidden_states_q
        hidden_states_q = self.input_layernorm(hidden_states_q)
        hidden_states_kv = self.input_layernorm(hidden_states_kv)
        hidden_states_q = self.cross_attn(
            hidden_states_q=hidden_states_q,
            hidden_states_kv=hidden_states_kv,
            attention_mask_query=attention_mask_q,
            attention_mask_kv=attention_mask_kv,
            position_ids=position_ids,
        )
        hidden_states_q = residual + hidden_states_q
        
        residual = hidden_states_q
        hidden_states_q = self.post_attention_layernorm(hidden_states_q)
        hidden_states_q = self.ffn(hidden_states_q,attention_mask_q)
        hidden_states_q = residual + hidden_states_q
        
        return hidden_states_q
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class GenARTPreTrainedModel(PreTrainedModel):
    config_class = GenARTConfig
    base_model_prefix = "model"
    _no_split_modules = ["GenARTLearntTokenizationLayer", "GenARTDecoderLayer", "GenARTEncoderLayer"]
    _supports_flash_attn_2 = True
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight,mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight,mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GenARTModel(GenARTPreTrainedModel):
    """
    Transformer encoder consisting of *config.num_hidden_layers* layers. Each layer is a ["GenARTLearntTokenizationLayer", "GenARTDecoderLayer", "GenARTEncoderLayer"]

    Args:
        config: GenARTConfig
    """

    def __init__(self, config: GenARTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size


        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        nucleotide_layers = []  
        token_layers = [] 

        self.conversion_layer_idx = config.conversion_layer_idx

        for layer_idx in range(0, self.conversion_layer_idx):
            nucleotide_layers.append(GenARTEncoderLayer(config, layer_idx))
        self.nucleotide_layers = nn.ModuleList(nucleotide_layers)
        
        self.conversion_layer = GenARTTokenizationLayer(config, self.conversion_layer_idx)
        
        for layer_idx in range(self.conversion_layer_idx+1, config.num_hidden_layers):
            token_layers.append(GenARTEncoderLayer(config, layer_idx))
        self.token_layers = nn.ModuleList(token_layers)


        self.norm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False 
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        return self.embed_tokens

    def set_input_embeddings(self, value):

        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, GenARTModelOutput]:


        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len) 
        else:
            position_ids = position_ids.view(-1, seq_len).long() 


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = attention_mask 
     
        hidden_states = inputs_embeds 

        nucleotide_attention_mask = attention_mask 
        # nucleotide layers
        for nucleotide_layer in self.nucleotide_layers:
            hidden_states = nucleotide_layer(
                    hidden_states,
                    attention_mask=nucleotide_attention_mask,
                    position_ids=position_ids,
                )
            
        nucleotide_hidden_states = hidden_states 
        hidden_states, token_attention_mask, token_special_tokens_mask = self.conversion_layer(
                hidden_states,
                attention_mask=nucleotide_attention_mask,
                special_tokens_mask = special_tokens_mask,
                position_ids=position_ids, 
        )

        # token layers
        for token_layer in self.token_layers:
            
            hidden_states = token_layer(
                    hidden_states,
                    attention_mask=token_attention_mask,
                    position_ids=position_ids,
                )
        token_hidden_states = self.norm(hidden_states)

        return GenARTModelOutput(
            token_hidden_states=token_hidden_states,
            nucleotide_hidden_states=nucleotide_hidden_states,
            token_attention_mask=token_attention_mask,
            nucleotide_attention_mask=nucleotide_attention_mask
        )

class GenARTForMaskedLM(GenARTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = GenARTModel(config)
        self.vocab_size = config.vocab_size
        self.decoder = GenARTDecoderLayer(config,config.num_hidden_layers)
        self.lm_head = LMHead(config)
        self.norm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, LanaguageModelingOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        if input_ids is not None:
            batch_size,seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size,seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # if not provided, generate position_ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        token_hidden_states = outputs.token_hidden_states
        nucleotide_hidden_states = outputs.nucleotide_hidden_states
        token_attention_mask = outputs.token_attention_mask
        nucleotide_attention_mask = attention_mask


        last_hidden_states = self.decoder(
            hidden_states_q=nucleotide_hidden_states,
            hidden_states_kv=token_hidden_states,
            attention_mask_q=nucleotide_attention_mask,
            attention_mask_kv=token_attention_mask,
            position_ids=position_ids,
        )
        

        last_hidden_states = self.norm(last_hidden_states)

        logits = self.lm_head(last_hidden_states)
        logits = logits.float()


        loss = None
        lm_loss = None        
        
        if labels is not None:
            # No shift for masked language modeling
            mlm_logits = logits[..., :, :].contiguous()
            mlm_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            mlm_logits = mlm_logits.view(-1, self.config.vocab_size)
            mlm_labels = mlm_labels.view(-1)
            # Enable model parallelism
            mlm_labels = mlm_labels.to(mlm_logits.device)
            lm_loss = loss_fct(mlm_logits, mlm_labels)
        
        loss = lm_loss
        
        return LanaguageModelingOutput(
            loss=loss, 
            lm_loss=lm_loss,
            logits=logits
        )
        
class GenARTForSequenceClassification(GenARTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GenARTModel(config)
        self.task_head= ClassificationHead(config)
        # Initialize weights and apply final processing
        self.norm = GenARTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if input_ids is not None:
            batch_size,seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size,seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()
            
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        token_hidden_states = outputs.token_hidden_states

        last_hidden_states = token_hidden_states
        last_hidden_states = self.norm(last_hidden_states)
        pooled_logits = self.task_head(last_hidden_states[:,0])
        loss = None
        cls_loss = None

        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    cls_loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    cls_loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                cls_loss = loss_fct(pooled_logits, labels)
                
        loss = cls_loss
        
            
        return SequenceClassifierOutput(
            loss=loss, 
            cls_loss=cls_loss, 
            logits=pooled_logits
        )
