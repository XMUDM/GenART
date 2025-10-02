#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-GPU finetuning script - supports distributed training and gradient accumulation
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR
import json
import random
from sklearn.metrics import matthews_corrcoef, f1_score

from model_source.GenART_model import GenARTForSequenceClassification
from model_source.GenART_config import GenARTConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BatchType = Dict[str, torch.Tensor]
CheckpointsType = Dict[str, float]

def get_args():
    parser = argparse.ArgumentParser(description="Model finetuning parameter settings")
    
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--eval_data", type=str, help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Model output directory, including final model and checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--max_steps", type=int, default=0, help="Total training steps, if 0 use epochs parameter")
    parser.add_argument("--lr", type=float, default=3e-5, help="Initial learning rate")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length, default is 1024")
    parser.add_argument("--save_steps", type=int, default=0, help="Model save steps; if 0, do not save checkpoints")
    parser.add_argument("--logging_steps", type=int, default=200, help="Logging steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

class ClassificationDataset(Dataset):
    """DNA classification dataset (directly uses numeric labels)"""
    def __init__(self, 
                tokenizer: PreTrainedTokenizer,
                data_path: str,
                max_length: int): 
        """
        Initialize dataset
        Args:
            tokenizer: Tokenizer instance
            data_path: Data file path (CSV)
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.examples = []

        logger.info(f"Processing data from {data_path}")
        df = pd.read_csv(data_path, sep='\t')
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
            sequence = " ".join(row['sequence'].strip())
            label = int(row['label'])  
            encoded = tokenizer.encode_plus(
                text=sequence,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )
            
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "special_tokens_mask": encoded["special_tokens_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

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
    preds = []
    labels_list = []
    
    with torch.no_grad():
        for batch in eval_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = model_to_eval(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    special_tokens_mask=batch["special_tokens_mask"].to(device),
                    labels=batch['labels'].to(device),
                )
            total_loss += outputs.cls_loss.item()
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_list.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(eval_loader)
    acc = np.mean(np.array(preds) == np.array(labels_list))
    mcc = matthews_corrcoef(labels_list, preds)
    f1 = f1_score(labels_list, preds, average='macro')
    return avg_loss, acc, mcc, f1

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
    update_weights: bool = True ):
    """
    Perform a single training step and return loss value
    Args:
        update_weights: Whether to update model weights (only update after gradient accumulation)
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    special_tokens_mask = batch["special_tokens_mask"].to(device)
    labels = batch["labels"].to(device)
    
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            labels=labels,
        )
    
    loss = outputs.loss
    scaled_loss = loss / args.gradient_accumulation_steps
    scaler.scale(scaled_loss).backward()
    
    if update_weights:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss.item()

def get_lr_lambda(max_steps, warmup_steps):
    """Learning rate scheduler function"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, 1.0 - float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps)))
    return lr_lambda

def setup_distributed(args):
    """Set up distributed training environment"""
    if args.local_rank == -1:
        return False
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    return True

def train(args: argparse.Namespace, device: torch.device, is_distributed: bool, is_main_process: bool):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    config = GenARTConfig.from_pretrained(args.pretrained_model)

    train_dataset = ClassificationDataset(tokenizer, args.train_data, args.max_len)
    
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
        eval_dataset = ClassificationDataset(tokenizer, args.eval_data, args.max_len)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    else:
        eval_loader = None

    config.num_labels = len(set([example["labels"].item() for example in train_dataset.examples]))
    
    total_train_batch_size = args.batch_size * args.gradient_accumulation_steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs
        args.max_steps = t_total
        
    if is_main_process:
        logger.info(f"Total training steps: {t_total}, total epochs: {args.epochs}")
        logger.info(f"Real batch size: {args.batch_size}, gradient accumulation steps: {args.gradient_accumulation_steps}, effective batch size: {total_train_batch_size}")

    model = GenARTForSequenceClassification.from_pretrained(args.pretrained_model, config=config).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
    optimizer = torch.optim.AdamW(model.parameters() if isinstance(model, DDP) else model.parameters(), 
                                lr=args.lr, 
                                betas=(0.9, 0.999), 
                                eps=1e-8, 
                                weight_decay=0.01)

    warmup_steps = len(train_loader) // args.gradient_accumulation_steps
    scheduler = LambdaLR(optimizer, get_lr_lambda(max_steps=args.max_steps, warmup_steps=warmup_steps))
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    global_step = 0
    checkpoints_dict = {}
    
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
        
        for step, batch in enumerate(progress_bar):
            
            update_weights = (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1
            
            loss = train_one_step(model, batch, tokenizer, device, scaler, args, optimizer, update_weights)
            
            if update_weights:
                global_step += 1
                scheduler.step()
            
                if is_main_process and global_step % args.logging_steps == 0:

                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            "steps":f"{global_step}",
                            "loss": f"{loss:.4f}"
                        })
                    
                    logger.info(f"Step {global_step} | Train Loss: {loss:.4f}")
                    
                    if eval_loader:
                        eval_loss, eval_acc, eval_mcc, eval_f1 = evaluate(model, tokenizer, eval_loader, device, args)
                        logger.info(f"Step {global_step} | Val Loss: {eval_loss:.4f} | ACC: {eval_acc:.4f} | MCC: {eval_mcc:.4f} | F1: {eval_f1:.4f}")
                        
                if is_main_process and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_path = save_checkpoint(
                        model, tokenizer, args.output_dir, global_step
                    )
                    logger.info(f"Save checkpoint at {checkpoint_path}")
                    
                    if eval_loader:
                        eval_loss, eval_acc, eval_mcc, eval_f1 = evaluate(model, tokenizer, eval_loader, device, args)
                        checkpoints_dict[checkpoint_path] = eval_mcc
                        
                        metrics_file = os.path.join(checkpoint_path, "metrics.json")
                        with open(metrics_file, "w") as f:
                            json.dump({
                                "step": global_step,
                                "mcc": float(eval_mcc),
                                "acc": float(eval_acc),
                                "f1": float(eval_f1),
                                "loss": float(eval_loss)
                            }, f, indent=4)
                        
                        logger.info(f"Checkpoint metrics: MCC: {eval_mcc:.4f}, ACC: {eval_acc:.4f}, F1: {eval_f1:.4f}")
            
                if args.max_steps > 0 and global_step >= args.max_steps:
                    if isinstance(progress_bar, tqdm):
                        progress_bar.close()
                    break

    if is_distributed:
        dist.barrier()
        
    return checkpoints_dict

def main():
    args = get_args()
    
    seed = 2025
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

    is_distributed = setup_distributed(args)
    
    is_main_process = not is_distributed or dist.get_rank() == 0
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if is_distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process:
        logger.info(f"Using device: {device}")
        if is_distributed:
            logger.info(f"Distributed training with {dist.get_world_size()} processes")

    checkpoints_dict = train(args, device, is_distributed, is_main_process)
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
