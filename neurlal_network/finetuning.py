import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json
import numpy as np
import random
from torch.utils.data import DataLoader
from fine_tuning.fine_tune_loader import FinetuningDataset
from fine_tuning.neuralnet import FineTuningNN
from fine_tuning.utils import parse_args

# Set a fixed seed for reproducibility
SEED = 1717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def create_output_dir(output_path, test_number):
    """
    Creates a directory for storing test outputs.
    """
    test_dir = os.path.join(output_path, f"Test_{test_number}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created directory: {test_dir}")
    return test_dir


def extract_top_attention(sequence_vector, inputs, batch_idx, k):
    """
    Extracts top k attention sequences, filtering out padded sequences.
    """
    sample_input = inputs[batch_idx]  # Fetch the correct sample input
    top_indices = torch.argsort(sequence_vector, descending=True)[:k].cpu().numpy()
    top_indices = [idx for idx in top_indices if idx < len(sample_input)]
    return [sample_input[idx] for idx in top_indices]


def validation():
    """
    Performs validation and computes validation loss and accuracy.
    """
    model.eval()
    validation_loss, total_correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels, _) in tqdm(enumerate(validation_dataloader), desc="Validation"):
            model_input = batch_inputs.to(device, dtype=torch.int64)
            labels = batch_labels.to(device, dtype=torch.float64)
            logits, _ = model(model_input)
            
            loss = criterion(logits, labels).item()
            validation_loss += loss
            
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
    
    validation_accuracy = round((total_correct * 100) / total_samples, 2)
    return validation_loss / (batch_idx + 1), validation_accuracy


def testing():
    """
    Performs testing and computes loss, accuracy, precision, recall, and confusion matrix.
    """
    model.eval()
    testing_loss, total_correct, total_samples = 0, 0, 0
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels, _) in tqdm(enumerate(testing_dataloader), desc="Testing"):
            model_input = batch_inputs.to(device, dtype=torch.int64)
            labels = batch_labels.to(device, dtype=torch.float64)
            logits, sequence_attention = model(model_input)
            
            loss = criterion(logits, labels).item()
            testing_loss += loss
            
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
            
            TP = ((predictions == 1) & (labels == 1)).sum().item()
            FP = ((predictions == 1) & (labels == 0)).sum().item()
            TN = ((predictions == 0) & (labels == 0)).sum().item()
            FN = ((predictions == 0) & (labels == 1)).sum().item()
            
            true_positives += TP
            false_positives += FP
            false_negatives += FN
            true_negatives += TN

    testing_accuracy = round((total_correct * 100) / total_samples, 2)
    precision = round((true_positives * 100) / (true_positives + false_positives), 2) if (true_positives + false_positives) > 0 else 0.0
    recall = round((true_positives * 100) / (true_positives + false_negatives), 2) if (true_positives + false_negatives) > 0 else 0.0
    confusion_matrix = {"TP": true_positives, "TN": true_negatives, "FP": false_positives, "FN": false_negatives, "precision": precision, "recall": recall}
    
    return testing_loss / (batch_idx + 1), testing_accuracy, precision, recall, confusion_matrix


def train():
    """
    Trains the model for one epoch.
    """
    model.train()
    train_loss, total_correct, total_samples = 0, 0, 0
    
    for batch_idx, (batch_inputs, batch_labels, _) in tqdm(enumerate(train_dataloader), desc="Training"):
        model_input = batch_inputs.to(device, dtype=torch.int64)
        labels = batch_labels.to(device, dtype=torch.float64)
        
        optimizer.zero_grad()
        logits, _ = model(model_input)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == labels).sum().item()
        total_samples += len(labels)
    
    train_accuracy = round((total_correct * 100) / total_samples, 2)
    return train_loss / (batch_idx + 1), train_accuracy


def load(tech_used):
    """
    Loads dataset, initializes model, and starts training.
    """
    global model, train_dataloader, validation_dataloader, testing_dataloader, criterion, optimizer
    
    # Create output directory
    output_path = os.path.join(args.output_dir, tech_used)
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize datasets and data loaders
    train_loader = FinetuningDataset(args, "training")
    validation_loader = FinetuningDataset(args, "validation")
    testing_loader = FinetuningDataset(args, "testing")
    
    train_dataloader = DataLoader(train_loader, num_workers=2, shuffle=True, batch_size=10)
    validation_dataloader = DataLoader(validation_loader, num_workers=2, batch_size=16)
    testing_dataloader = DataLoader(testing_loader, num_workers=2, batch_size=16)
    
    # Initialize model
    model = FineTuningNN(embedding_dim=args.embedding_dim, nhead=args.nheads, 
                        dim_feedforward=args.d_hid, num_layers=args.nlayers, 
                        dropout_prob_1=args.dropout_1, dropout_prob_2=args.dropout_2, vocab_size=train_loader.vocab_size, 
                        device=device, token_padding=train_loader.vocabulary["<PAD>"], 
                        unique_apis=train_loader.vocab_size, num_classes=args.num_classes,
                        num_sequences=args.sequence_budget, cnn_kernel_size=args.cnn_kernel_size, 
                        cnn_kernel_stride=args.cnn_kernel_stride, 
                        maxpool_kernel_size =args.maxpool_kernel_size, 
                        maxpool_kernel_stride=args.maxpool_kernel_stride,
                        cnn_output_channels = args.cnn_output_channels)
    
    #loading saved weights
    if args.load_weight==1:
        print("loading saved weights")
        checkpoint_file_path = args.checkpoint_file_path
        checkpoint = torch.load(checkpoint_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_validation_loss = float("inf")

    print("Now Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train()
        validation_loss, validation_accuracy = validation()
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy}%")
        print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy}%")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            print("Weights saved at: ", os.path.join(output_path, "best_model.pth"))
            validation_results = {"validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "epoch": epoch}
            with open(os.path.join(output_path, "validation_results.json"), "w") as my_file:
                json.dump(validation_results, my_file, indent=4)
    
    model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pth")))
    testing_loss, testing_accuracy, precision, recall, confusion_matrix = testing()
    print(f"Testing Loss: {testing_loss:.4f}, Testing Accuracy: {testing_accuracy}%, Precision: {precision}%, Recall: {recall}%")
    
    testing_results = {"testing_loss": testing_loss, "testing_accuracy": testing_accuracy, "precision": precision, "recall": recall, "confusion_matrix": confusion_matrix}
    with open(os.path.join(output_path, "testing_results.json"), "w") as my_file:
        json.dump(testing_results, my_file, indent=4)
    



if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    load(args.tech_used)
