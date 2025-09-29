#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GenART model pretraining script - supports multi-GPU and gradient accumulation
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR
import json
import time

from model_source.GenART_model import GenARTForMaskedLM
from model_source.GenART_config import GenARTConfig

# Configure logging system
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Type alias definitions
BatchType = Dict[str, torch.Tensor]
CheckpointsType = Dict[str, float]

def get_args():
    """Parse and return command-line arguments"""
    parser = argparse.ArgumentParser(description="Model pretraining parameter settings")
    
    # Data parameters
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--eval_data", type=str, help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Model output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                       help="Number of gradient accumulation steps to increase effective batch size")
    parser.add_argument("--epochs", type=int, default=0, help="Total training epochs")
    parser.add_argument("--max_steps", type=int, default=300000, help="Total training steps, if 0 use epochs parameter")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=300000, help="Model save interval steps")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging interval steps")
    parser.add_argument("--init_from_checkpoint", type=str, help="Checkpoint path to initialize model")
    parser.add_argument("--global_start_step", type=int, default=0, help="Global training start step")
    parser.add_argument("--total_train_steps", type=int, default=0, help="Total training steps, 0 means use max_steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    return parser.parse_args()

class DNASequenceDataset(Dataset):
    """DNA sequence preprocessing dataset, all threads directly read raw data file"""
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int = 1024, 
                 batch_size: int = 1000):
        """
        Initialize dataset
        Args:
            tokenizer: Tokenizer instance
            file_path: Data file path
            block_size: Maximum sequence length
            batch_size: Number of samples per batch (used only in data processing stage)
        """
        try:
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if len(line.strip()) > 0]
        except FileNotFoundError:
            logger.error(f"Data file {file_path} not found")
            raise

        logger.info(f"Processing {len(lines)} lines")
        self.examples = []

        for i in tqdm(range(0, len(lines), batch_size), desc="Batch processing"):
            batch_lines = lines[i:min(i+batch_size, len(lines))]
            batch_lines = [" ".join(line) for line in batch_lines]
            encoded = tokenizer.batch_encode_plus(
                batch_lines,
                add_special_tokens=True,
                max_length=block_size,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )
            for j in range(len(batch_lines)):
                self.examples.append({
                    "input_ids": encoded["input_ids"][j],
                    "attention_mask": encoded["attention_mask"][j],
                    "special_tokens_mask": encoded["special_tokens_mask"][j],
                })

    def __len__(self):
        """Return total size of dataset"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        return self.examples[idx]


def mask_sequence(batch: BatchType, tokenizer: PreTrainedTokenizer, mlm_prob: float=0.15 ):
    """
    Implement BERT-style dynamic masking
    Args:
        batch: Original input batch
        tokenizer: Tokenizer for mask token
        mlm_prob: Masking probability
    Returns:
        masked_inputs: Masked input
        labels: Corresponding labels
    """
    labels = batch["input_ids"].clone()
    prob_matrix = torch.full(labels.shape, mlm_prob)
    
    special_tokens_mask = batch["special_tokens_mask"]
    prob_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
    
    masked_indices = torch.bernoulli(prob_matrix).bool()
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    batch["input_ids"][indices_replaced] = tokenizer.mask_token_id
    
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() 
        & masked_indices 
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    batch["input_ids"][indices_random] = random_words[indices_random]
    
    labels[~masked_indices] = -100
    
    return batch, labels

def evaluate(
    model,
    tokenizer: PreTrainedTokenizer, 
    eval_loader: DataLoader, 
    device: torch.device,
    args: argparse.Namespace ):
    """Model evaluation process"""

    if isinstance(model, DDP):
        model_to_eval = model.module
    else:
        model_to_eval = model
        
    model_to_eval.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            inputs, labels = mask_sequence(batch, tokenizer, mlm_prob=0.15)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = model_to_eval(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    special_tokens_mask=batch["special_tokens_mask"].to(device),
                    labels=labels.to(device),
                )

            total_loss += outputs.lm_loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl 

def save_checkpoint(
    model,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    global_step: int ):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}-steps")

    if isinstance(model, DDP):
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir

def train_one_step(
    model,
    batch: BatchType,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    optimizer,
    global_step: int,
    update_optimizer: bool = False):
    """Perform a single training step and return loss value"""

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    special_tokens_mask = batch["special_tokens_mask"].to(device)

    inputs, labels = mask_sequence(batch, tokenizer, mlm_prob=0.15)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            labels=labels.to(device),
        )
    
    loss = outputs.loss
    
    loss = loss / args.gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    if update_optimizer:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss.item() * args.gradient_accumulation_steps

def get_lr_lambda(max_steps, warmup_steps=2000):
    """Learning rate scheduler: linear warmup for first 2000 steps, then constant"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0 
    return lr_lambda

def setup_distributed(args):
    """Set up distributed training environment"""
    if args.local_rank == -1:
        return False
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    return True

def train(args: argparse.Namespace, device: torch.device, is_distributed: bool, is_main_process: bool):
    """Main training loop"""
    tokenizer = AutoTokenizer.from_pretrained("./model_source/1mertokenizer")
    config = GenARTConfig.from_pretrained("./model_source/config/config_350M.json")
    
    global_step = args.global_start_step

    continue_training = args.init_from_checkpoint is not None and args.init_from_checkpoint != ""

    if continue_training:
        logger.info(f"Loading model from checkpoint: {args.init_from_checkpoint}")
        model = GenARTForMaskedLM.from_pretrained(args.init_from_checkpoint).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 
                                       lr=args.lr, 
                                       betas=(0.9, 0.98), 
                                       eps=1e-6, 
                                       weight_decay=0.01)

        optimizer_state_path = os.path.join(args.init_from_checkpoint, "optimizer.pt")
        scheduler_state_path = os.path.join(args.init_from_checkpoint, "scheduler.pt")

        if os.path.exists(optimizer_state_path):
            if is_main_process:
                logger.info(f"Loading optimizer state: {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
    else:
        model = GenARTForMaskedLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 
                                       lr=args.lr, 
                                       betas=(0.9, 0.98), 
                                       eps=1e-6, 
                                       weight_decay=0.01)
    
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    scheduler = LambdaLR(
        optimizer, 
        get_lr_lambda(max_steps=args.max_steps, warmup_steps=2000),
        last_epoch=global_step-1 if global_step > 0 else -1
    )
    
    if continue_training and os.path.exists(scheduler_state_path):
        if is_main_process:
            logger.info(f"Loading scheduler state: {scheduler_state_path}")
        scheduler.load_state_dict(torch.load(scheduler_state_path))
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    train_dataset = DNASequenceDataset(tokenizer, args.train_data, args.max_len)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
        
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True  
    )
    
    if args.eval_data and is_main_process:  
        eval_dataset = DNASequenceDataset(tokenizer, args.eval_data, args.max_len)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    else:
        eval_loader = None
    
    checkpoints_dict = {}
    
    # If max_steps is set, calculate training epochs based on max_steps, otherwise use epochs parameter
    if args.max_steps > 0:
        args.epochs = (args.max_steps-args.global_start_step) // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        args.max_steps = args.epochs * (len(train_loader) // args.gradient_accumulation_steps)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if is_distributed:
        world_size = dist.get_world_size()
        effective_batch_size *= world_size
    
    if is_main_process:
        logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"Batch size per device: {args.batch_size}")
        logger.info(f"Effective total batch size: {effective_batch_size}")
        logger.info(f"Total training steps: {args.max_steps}")
        logger.info(f"Total training epochs: {args.epochs}")

    optimizer.zero_grad()
    
    for epoch in range(args.epochs):

        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        

        if is_main_process:
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=True,
                position=0,
                dynamic_ncols=True,
            )
        else:
            progress_bar = train_loader
        
        accumulated_loss = 0.0
        
        for step, batch in enumerate(progress_bar):

            is_update_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader))
            

            loss = train_one_step(
                model, batch, tokenizer, device, scaler, args, optimizer, 
                global_step, update_optimizer=is_update_step
            )
            
            accumulated_loss += loss
            
            if is_update_step:

                global_step += 1
                
                avg_loss = accumulated_loss / args.gradient_accumulation_steps
                accumulated_loss = 0.0
                
                scheduler.step()
                
                if is_main_process and global_step % args.logging_steps == 0:

                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            "steps":f"{global_step}",
                            "loss": f"{avg_loss:.4f}"
                        })
                    
                    logger.info(f"Step {global_step}: train/loss={avg_loss:.4f}")
                    
                    if eval_loader:
                        eval_loss, ppl = evaluate(model, tokenizer, eval_loader, device, args)
                        logger.info(f"Step {global_step}: valid/loss={eval_loss:.4f}, valid/ppl={ppl:.2f}")

                if is_main_process and global_step % args.save_steps == 0:
                    checkpoint_path = save_checkpoint(
                        model, tokenizer, args.output_dir, global_step
                    )

                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
                    
                    logger.info(f"Save checkpoint and optimizer/scheduler states at {checkpoint_path}")
                    
                    if eval_loader:
                        eval_loss, ppl = evaluate(model, tokenizer, eval_loader, device, args)
                        logger.info(f"Checkpoint {checkpoint_path}: valid/loss={eval_loss:.4f}, valid/ppl={ppl:.2f}")
                        checkpoints_dict[checkpoint_path] = ppl
                
                if args.max_steps > 0 and global_step >= args.max_steps:
                    if isinstance(progress_bar, tqdm):
                        progress_bar.close()
                    break
                
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if is_distributed:
        dist.barrier()
            
    return checkpoints_dict


def main():
    args = get_args()
    
    # Set up distributed environment
    is_distributed = setup_distributed(args)
    
    # Determine if main process
    is_main_process = not is_distributed or dist.get_rank() == 0
    
    # Prepare output directory
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Device initialization
    if is_distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        logger.info(f"Using device: {device}")
        if is_distributed:
            logger.info(f"Distributed training with {dist.get_world_size()} processes")
        
    # Execute training
    checkpoints_dict = train(args, device, is_distributed, is_main_process)
    
    # Print validation PPL for all checkpoints
    if is_main_process and len(checkpoints_dict) > 0:
        logger.info("All checkpoints validation PPL:")
        for ckpt, ppl in checkpoints_dict.items():
            logger.info(f"{ckpt}: valid/ppl={ppl:.2f}")
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
#     --train_data /path/to/train_data.csv \
#     --eval_data /path/to/test_data.csv \
#     --output_dir /path/to/output_dir \
#     --batch_size 8 \
#     --max_steps 10000 \
#     --save_steps 10000

# CUDA_VISIBLE_DEVICES=0 python pretrain.py \
#     --train_data /path/to/train_data.csv \
#     --eval_data /path/to/test_data.csv \
#     --output_dir /path/to/output_dir \
#     --batch_size 8 \
#     --max_steps 10000 \
#     --save_steps 10000