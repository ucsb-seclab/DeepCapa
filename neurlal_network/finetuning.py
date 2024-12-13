import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from fine_tuining_get_data import fetch_features
import argparse
import os
import json
from finetuining_dataloader import FineTuiningLoader
from neuralnet import FineTuining
import time
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import random

SEED = 1717

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def create_output_dir(output_path, test_number):
    test = "Test_{}".format(test_number)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path,test)):
        os.mkdir(os.path.join(output_path,test))
        print("created directory {}".format(os.path.join(output_path,test)))
    print("output_path : {}".format(os.path.join(output_path,test)))
    return os.path.join(output_path,test)
    
def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser(description="deepcapa")
    ir = parser.add_argument_group("Dataset parameters")
    ir.add_argument("--input-file", default='', help="input_file_path", required=True)
    ir.add_argument("--labels-file", default='', help="labels_file_path", required=True)
    ir.add_argument("--output-path", default ="", help="output path to store model results", required=True)
    ir.add_argument("--unique-api-path", default="", help="path to unique APIs", required=True)
    ir.add_argument("--training-split",type=float, default=0.60, help="to specify test number, this is only used for hyper-parameter fine-tuining")
    ir.add_argument("--label-order-path", type=str, default="/data/saastha/models_and_results/hyper_parameter/Transformer_out/base_training/label_order.json")
    
    #Trianing hyper-parameters
    hr = parser.add_argument_group("Training Hyper-parameters")
    hr.add_argument("--api-padding",type=int, default=20,help="API padding value")
    hr.add_argument("--seq-budget",type=int, default=20,help="API padding value")
    hr.add_argument("--threshold",type=float, default=0.5, help="thresold used(only for stage2)")
    hr.add_argument("--lr", type=float, default=0.0001)
    hr.add_argument("--variable-lr", type=bool, default=True, help="False if you want to have fixed learning rate")
    hr.add_argument('--batch-size', help="Training batch size", default=32, type=int)
    hr.add_argument('--epochs', help="Epochs, i.e., number of passes over the training set", type=int, default=20)
    hr.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    hr.add_argument('--gpu', type=str, default="0", help="if device is cuda, determines which gpu to use")
    hr.add_argument('--tech-used', type=str, default='', help="tech to perform training", required=True)
    hr.add_argument('--load-weight', type=int, default=1, help="whether to load pre-trained weights")
    hr.add_argument('--checkpoint-file-path', type=str, default="/data/saastha/models_and_results/hyper_parameter/transformer_approach/fine_tuining/without_pretraining/T1036/epoch_199_nheads_4_d_hid_4_embedding_dim_768_nlayers_4_dropout_10.1_dropout_20.0/_saved_wt.pt", help="path to checkpoint file")
    #hr.add_argument('--saved-weight-path', type=int, default=199, help="which epoch to load")
    # Model Parameters
    mr = parser.add_argument_group("Parameters Related to Neural Network Architecture")
    mr.add_argument('--embedding-dim', help="Embedding dimension added after the input layer", type=int, default=768)
    mr.add_argument('--penultimate-features-num', type=int, default=768,
                    help="This determines the number of neurons (i.e., features) at the last layer of the main body"
                         "of the neural network (before linear classifiers)")
    mr.add_argument('--num-classes', type=int, default=2, help="number of classes in penultimate layer")
    mr.add_argument('--nheads', help="number of atention heads", type=int,
                    default=4)
    mr.add_argument('--nlayers', help="number of transformer layers", type=int,
                    default=4)
    mr.add_argument('--dropout-1', help="Dropout Probability for transformer", type=float, default=0.1) 
    mr.add_argument('--dropout-2', help="Dropout Probability before classifier", type=float, default=0.0) 
    mr.add_argument('--d_hid', type=int, help="The dimension of the feedforward network model for transformer encoder", default=2048)
    mr.add_argument('--cnn-kernel-size', type=int, help="The dimention of kernel size", default=3)
    mr.add_argument('--cnn-kernel-stride', type=int, help="The dimention of kernel stride", default=3)
    mr.add_argument('--maxpool-kernel-size', type=int, help="Maxpool kernel size", default=3)
    mr.add_argument('--maxpool-kernel-stride', type=int, help="Maxpool kernel stride", default=3)
    mr.add_argument('--cnn-output-channels', type=int, default=8)
    args = parser.parse_args()
    return args



def extract_top_attention(sequence_vector, inputs, batch_idx, k):
    # Getting the sequences for correct sample
    sample_input = inputs[relative_index[batch_idx]]
    # Getting the top sequence indices
    top_sequence_indices = torch.argsort(sequence_vector, descending=True)[:k]
    top_sequence_indices_list = top_sequence_indices.cpu().data.numpy().tolist()
    # This is requried to remove padded sequences
    top_sequence_idx = [index for index in top_sequence_indices_list if index < len(sample_input)]
    top_sequences = [sample_input[index] for index in top_sequence_idx]
    return top_sequences

def validation():
    # Set the model to evaluation mode
    model.eval()
    validation_loss = float(0)
    # Disable gradient computation
    total_correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    output_dict = {}
    outputs = []
    epoch_api_attention = []
    epoch_sequence_attention = []
    #confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    with torch.no_grad():
        #Validation Batch Loop
        for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(validation_dataloader), desc="validation"):  
            #print("within validation")
            # Compute the model's output
            model_input = batch_inputs[0].to(device, dtype=torch.int64)
            #labels = batch_labels[0].to(device, dtype=torch.float64)
            labels = batch_labels.to(device, dtype=torch.float64)
            # Model output
            logits, _ = model(model_input)
            
            loss = criterion(logits, labels)
            batch_loss = loss.item()
            validation_loss += batch_loss

            #probabilities
            probabilities = torch.sigmoid(logits)
            
            # Calculate binary classification predictions by taking the threshold (typically 0.5)
            predictions = (probabilities >= 0.5).float()
            outputs.append([predictions, batch_labels])
            # Calculate the number of correct predictions in this batch
            batch_correct = (predictions == labels).all().sum().item()
            
            # Update total correct predictions
            total_correct += batch_correct
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
    # Compute the average validation loss and accuracy
    
    validation_accuracy = round((float(total_correct * 100) / (batch_idx + 1)), 2)
    validation_loss = (validation_loss / (batch_idx + 1))

    if true_positives + false_positives == 0:
        precision = 0.
    else:
        precision = round(float(true_positives*100) / (true_positives + false_positives), 2)
    if true_positives + false_negatives == 0:
        recall = 0.
    else:
        recall = round(float(true_positives*100) / (true_positives + false_negatives), 2)
    confusion_matrix = {"TP":true_positives, "TN": true_negatives, "FP": false_positives, "FN": false_negatives, "precision": round(precision, 2), "recall": round(recall, 2)}
    
    return validation_loss, validation_accuracy, precision, recall, confusion_matrix

def testing(inputs):
    # Set the model to evaluation mode
    model.eval()
    testing_loss = float(0)
    # Disable gradient computation
    total_correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    detailed_result = {}
    true_negatives = 0
    confusion_matrix = {}
    with torch.no_grad():
        #Validation Batch Loop
        for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(testing_dataloader), desc="testing"):  
            #print("within validation")
            # Compute the model's output
            model_input = batch_inputs[0].to(device, dtype=torch.int64)
            #labels = batch_labels[0].to(device, dtype=torch.float64)
            labels = batch_labels.to(device, dtype=torch.float64)
            # Model output
            logits, sequence_attention = model(model_input)
            loss = criterion(logits, labels)
            batch_loss = loss.item()
            testing_loss += batch_loss

            #probabilities
            probabilities = torch.sigmoid(logits)
            
            # Calculate binary classification predictions by taking the threshold (typically 0.5)
            predictions = (probabilities >= 0.5).float()
            
            # Calculate the number of correct predictions in this batch
            batch_correct = (predictions == labels).all().sum().item()
            
            # Update total correct predictions
            total_correct += batch_correct
            TP = ((predictions == 1) & (labels == 1)).sum().item()
            FP = ((predictions == 1) & (labels == 0)).sum().item()
            TN = ((predictions == 0) & (labels == 0)).sum().item()
            FN = ((predictions == 0) & (labels == 1)).sum().item()
            true_positives += TP
            false_positives += FP
            false_negatives += FN
            true_negatives += TN
            top_sequences, important_apis = extract_top_attention(sequence_attention, inputs, batch_idx, 20)
   
    testing_accuracy = round((float(total_correct * 100) / (batch_idx + 1)), 2)
    testing_loss = (testing_loss / (batch_idx + 1))
    if true_positives + false_positives == 0:
        precision = 0.
    else:
        precision = round(float(true_positives*100) / (true_positives + false_positives), 2)
    if true_positives + false_negatives == 0:
        recall = 0.
    else:
        recall = round(float(true_positives*100) / (true_positives + false_negatives), 2)
    confusion_matrix = {"TP": true_positives, "TN": true_negatives, "FP": false_positives, 
                        "FN": false_negatives, "precision": precision, "recall": recall,
                        "Total": true_positives + true_negatives + false_positives + false_negatives}
    return testing_loss, testing_accuracy, precision, recall, confusion_matrix


def train():
    model.train()
    train_loss = float(0)
    #loss_values = []
    batches = 0
    total_correct = 0
    
    for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(train_dataloader), desc="training"):  
        
        #print(batch_idx)
        
        model_input = batch_inputs[0].to(device, dtype=torch.int64)
        
        labels = batch_labels.to(device, dtype=torch.float64)
        
        logits, _ = model(model_input)
        
        loss = criterion(logits, labels)
        
        batch_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += batch_loss
        
        #probabilities
        probabilities = torch.sigmoid(logits)
        
        # Calculate binary classification predictions by taking the threshold (typically 0.5)
        predictions = (probabilities >= 0.5).float()
        
        # Calculate the number of correct predictions in this batch
        batch_correct = (predictions == labels).all().sum().item()

        # Update total correct predictions
        total_correct += batch_correct
        
    
    train_accuracy = round((float(total_correct * 100) / (batch_idx + 1)), 2)
    train_loss = (train_loss / (batch_idx + 1))
    
    return train_loss, train_accuracy

def load(tech_used):
    global model, train_dataloader, validation_dataloader, testing_dataloader, criterion, optimizer, threshold
    epochs = args.epochs
    
    all_args = {"embedding": args.embedding_dim, "num_encoder": args.nlayers, 
                "num_attention": args.nheads, "initial_lr": str(args.lr), "sequence": 350, "api": 20}
    
    #if the output path does not exist
    output_path = os.path.join(args.output_path, tech_used)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        print("created output path : {}".format(output_path))
    
    with open(os.path.join(output_path, "parameters.json"), "w") as my_file:
        json.dump(all_args, my_file)
    print("parameters written to : {}".format(os.path.join(output_path, "parameters.json")))
   
    X_training, X_validation, X_testing, Y_training, Y_validation, Y_testing, total_tokens, unique_apis, special_tokens  = fetch_features(args.debug_mode, tech_used, args.input_path, args.labels_path)
    
    print("total training samples : {}".format(len(X_training)))
    print("total validation labels: {}".format(len(X_validation)))
    num_sequences = len(X_training[0])

    #initializing the model
    model = FineTuining(embedding_dim=args.embedding_dim, nhead=args.nheads, 
                        dim_feedforward=args.d_hid, num_layers=args.nlayers, 
                        dropout_prob_1=args.dropout_1, dropout_prob_2=args.dropout_2, vocab_size=total_tokens, 
                        device=device, token_padding=special_tokens["<PAD>"], 
                        unique_apis=len(unique_apis), num_classes=args.num_classes,
                        num_sequences=num_sequences, cnn_kernel_size=args.cnn_kernel_size, 
                        cnn_kernel_stride=args.cnn_kernel_stride, 
                        maxpool_kernel_size =args.maxpool_kernel_size, 
                        maxpool_kernel_stride=args.maxpool_kernel_stride,
                        cnn_output_channels = args.cnn_output_channels)
    #loading saved weights
    
    model.to(device)
    if args.load_weight==1:
        print("loading saved weights")
        checkpoint_file_path = args.checkpoint_file_path
        checkpoint = torch.load(checkpoint_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.to(device)  
    #extracting the label_order to extract the linear layer associated with tech_used
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)

    # Define the loss function (CrossEntropyLoss) for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()
    epoch = 1             
    
    train_dataset = FineTuiningLoader(X_training, Y_training)
    validation_dataset = FineTuiningLoader(X_validation, Y_validation)
    testing_dataset = FineTuiningLoader(X_testing, Y_testing)
        
    train_dataloader = DataLoader(train_dataset, num_workers=5, shuffle=True, batch_size=1)
    validation_dataloader = DataLoader(validation_dataset, num_workers=5, batch_size=1)
    testing_dataloader = DataLoader(testing_dataset, num_workers=5, batch_size=1)

    result_per_epoch = {}
    best_result = {}
    best_f1 = 0
    print("Now Training...")
    attention = {}
    #epoch = 7
    # setting val loss to infinity
    best_val_loss = float("inf")
    while epoch <= args.epochs:
        
        train_loss, train_accuracy  = train()
        print("Epoch: {}, train_loss: {}, train accuracy: {}".format(epoch,\
                                     round(train_loss, 4), train_accuracy))
        validation_loss, validation_accuracy, precision, recall, confusion_matrix = validation()
        f1_score = 0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = float(2*precision*recall)/(precision + recall)
        
        print("Validation:  loss: {}, Accuracy {}, Confusion: {}  F1: {}".format(round(validation_loss, 4), validation_accuracy, confusion_matrix,f1_score ))
        
        if best_val_loss < validation_loss:
            best_val_loss = validation_loss
            testing_loss, testing_accuracy, testing_precision, testing_recall, testing_confusion_matrix = testing(X_testing)
        else:
            continue
        
        
        print("Testing: loss: {}, Accuracy: {}, Confusion: {}".format(round(testing_loss), validation_accuracy, testing_confusion_matrix))
        result = {"training_loss":train_loss, "training_accuracy": train_accuracy, "validation_loss":validation_loss, "validation_accuracy": validation_accuracy, "validation_confusion_matrix": confusion_matrix, "testing_confusion_matrix": testing_confusion_matrix}
        result_per_epoch[epoch] = result
        
        model_arch = "epoch_"+ str(epoch) + "_nheads_" + str(args.nheads) + "_d_hid_" + str(args.nheads) + "_embedding_dim_" + str(args.embedding_dim) + "_nlayers_" + str(args.nlayers) \
                        + "_dropout_1" + str(args.dropout_1)  + "_dropout_2" + str(args.dropout_2) + "/"
        
        output_path_current = os.path.join(output_path, model_arch)
        if not os.path.exists(output_path_current):
            os.mkdir(output_path_current)
        with open("{}".format(os.path.join(output_path_current, "result.json")), "w") as my_file:
            json.dump(result, my_file)

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "{}".format(os.path.join(output_path_current, "_saved_wt.pt")))
        print("weigths saved at : {}".format(os.path.join(output_path_current, "_saved_wt.pt")))
        
        if best_f1 < f1_score:
            best_f1 = f1_score
            print("best epoch changed to : {}".format(epoch))

            best_result["epoch"] = epoch
            best_result["val_accuracy"] = validation_accuracy
            best_result["val_loss"] = validation_loss
            best_result["train_accuracy"] = train_accuracy
            best_result["train_loss"] = train_loss
            best_result["validation_confusion_matrix"] = confusion_matrix
            best_result["f1_score"] = best_f1
            best_result["testing_confusion_matrix"] = testing_confusion_matrix
            
            
            with open("{}".format(os.path.join(output_path,"best_result.json")), "w") as my_file:
                json.dump(best_result, my_file)
            print("best result written to: {} ".format(os.path.join(output_path,"best_result.json")))
            

        epoch += 1
    with open("{}".format(os.path.join(output_path,"all_result.json")), "w") as my_file:
            json.dump(result_per_epoch, my_file)
    print("writing attention ;;;;;")
    print("attention written")
    print("attention written")
    print("DONE")    
    



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print("device is : {}".format(device))
    
    generic_set = ["T1036", "T1568", "T1047", "T1552", "T1027",
                "T1014", "T1053", "T1059", "T1203", "T1485",
                "T1564", "T1562"]
    api_dependent = [ 
                    "T1486", "T1082", "T1055", "T1056","T1083",
                    "T1070", "T1095", "T1071", "T1112", "T1547",
                    "T1497", "T1134", "T1518","T1049", "T1543", "T1033"]
    techs_to_use = api_dependent+ generic_set
    for tech_used in techs_to_use:
        print(tech_used)
        load(tech_used)
       

    import IPython
    IPython.embed()
    assert False