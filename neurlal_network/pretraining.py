import os
import json
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from pretraining.neuralnet import Pretraining
from pretraining.m_loader import MLMDataset
from pretraining.utils import parse_args

# Set random seed for reproducibility
SEED = 1717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def create_output_dir(output_path, test_number):
    test_dir = os.path.join(output_path, f"test_{test_number}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created directory: {test_dir}")
    return test_dir

def validation(MASK_TOKEN):
    model.eval()
    total_correct, total_masked = 0, 0
    validation_loss = 0
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(validation_dataloader), desc="Validation"):
            model_input = batch_inputs.to(device, dtype=torch.int64)
            labels = batch_labels.to(device, dtype=torch.int64).view(-1)
            logits = model(model_input)
            loss = criterion(logits, labels)
            
            masked_tokens = labels != -100
            batch_masked = masked_tokens.sum().item()
            validation_loss += loss.item() if batch_masked > 0 else 0
            
            predictions = logits.argmax(dim=1)
            batch_correct = ((predictions == labels) & masked_tokens).sum().item()
            total_correct += batch_correct
            total_masked += batch_masked
    
    validation_accuracy = round((total_correct * 100) / total_masked, 2) if total_masked > 0 else 0
    validation_loss /= (batch_idx + 1)
    
    return validation_loss, validation_accuracy

def train(MASK_TOKEN):
    model.train()
    total_correct, total_masked = 0, 0
    train_loss = 0
    
    for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(train_dataloader), desc="Training"):
        optimizer.zero_grad()
        
        model_input = batch_inputs.to(device, dtype=torch.int64)
        labels = batch_labels.to(device, dtype=torch.int64).view(-1)
        logits = model(model_input)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        my_lr_scheduler.step()
        
        masked_tokens = labels != -100
        batch_masked = masked_tokens.sum().item()
        train_loss += loss.item() if batch_masked > 0 else 0
        
        predictions = logits.argmax(dim=1)
        batch_correct = ((predictions == labels) & masked_tokens).sum().item()
        total_correct += batch_correct
        total_masked += batch_masked
    
    train_accuracy = round((total_correct * 100) / total_masked, 2) if total_masked > 0 else 0
    train_loss /= (batch_idx + 1)
    
    return train_loss, train_accuracy

def load():
    global model, train_dataloader, validation_dataloader, criterion, optimizer, my_lr_scheduler
    
    output_path = create_output_dir(args.output_dir, 1)
    all_args = {
        "embedding": args.embedding_dim,
        "num_encoder": args.nlayers,
        "num_attention": args.nheads,
        "initial_lr": str(args.lr),
        "sequence": args.sequence_budget,
        "api": args.sequence_length,
        "masking_scheme": args.masking_scheme
    }
    
    with open(os.path.join(output_path, "parameters.json"), "w") as file:
        json.dump(all_args, file)
    
    train_loader = MLMDataset(args, seed=SEED, split="train")
    validation_loader = MLMDataset(args, seed=SEED, split="validation")
    
    train_dataloader = DataLoader(train_loader, num_workers=0, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_loader, num_workers=5, batch_size=args.batch_size * 2, shuffle=False)
    
    model = Pretraining(
        embedding_dim=args.embedding_dim,
        nhead=args.nheads,
        dim_feedforward=args.d_hid,
        num_layers=args.nlayers,
        dropout_prob=args.dropout,
        vocab_size=train_loader.vocab_size,
        device=device,
        token_padding=train_loader.vocabulary["<PAD>"],
        seq_len=args.sequence_length
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01)
    
    warmup_steps = len(train_loader) * (args.epochs - args.epoch_to_load)
    my_lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(step / (0.1 * warmup_steps), 1.0))
    
    if args.load_weight:
        checkpoint_files = [f for f in os.listdir(output_path) if f.endswith(".pt")]
        if checkpoint_files:
            checkpoint = torch.load(os.path.join(output_path, checkpoint_files[0]))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            my_lr_scheduler.load_state_dict(checkpoint['scheduler'])
            epoch = checkpoint['epoch'] + 1
        else:
            epoch = 1
    else:
        epoch = 1
    
    best_training_loss = float('inf')
    result_per_epoch = {}
    
    while epoch <= args.epochs:
        train_loss, train_accuracy = train(train_loader.vocabulary["<MASK>"])
        validation_loss, validation_accuracy = validation(train_loader.vocabulary["<MASK>"])
        
        result_per_epoch[epoch] = {
            "training_loss": train_loss,
            "training_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }
        
        if train_loss < best_training_loss:
            best_training_loss = train_loss
            best_result = result_per_epoch[epoch]
            best_result["epoch"] = epoch
            
            with open(os.path.join(output_path, "best_result.json"), "w") as file:
                json.dump(best_result, file)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': my_lr_scheduler.state_dict()
            }, os.path.join(output_path, "saved_wt.pt"))
        epoch += 1

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    load()
