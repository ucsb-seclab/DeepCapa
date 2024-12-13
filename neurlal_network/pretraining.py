import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from pretrain_get_data import fetch_features
import argparse
import os
import json
from dataloader import loader
from dataloader import Dataset
#from neuralnet import TransformerCNNClassificationModel
from neuralnet import MaskedLearningModel
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch.multiprocessing as mp
# from sklearn.metrics import multilabel_confusion_matrix
from m_loader import MLMDataset
from tqdm import tqdm
import random
import torch.distributed as dist
from datetime import timedelta
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
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
    ir.add_argument("--input-file", default='',  help="input_file_path", required=True)
    ir.add_argument("--labels-file", default='', help="input_file_path", required=True)
    ir.add_argument("--training-set-ratio", default=0.7, help="ratio of trainingset/validation set")
    ir.add_argument("--output-path", default ='', help="output path to store model results", required=True)
    ir.add_argument("--unique-api-path", default='', help="path to unique APIs", required=True)
    ir.add_argument("--test-no", default="1", help="to specify test number, this is only used for hyper-parameter fine-tuining")
    #Trianing hyper-parameters
    hr = parser.add_argument_group("Training Hyper-parameters")
    ir.add_argument('--local-rank', type=int, default=0)
    hr.add_argument("--api-padding",type=int, default=20,help="API padding value")
    hr.add_argument("--sequence-budget",type=int, default=350,help="API padding value")
    hr.add_argument("--training-stage",type=int, default=1, help="1 for base training, 2 for fine_tuining")
    hr.add_argument("--threshold",type=float, default=0.5, help="thresold used(only for stage2)")
    hr.add_argument("--lr", type=float, default=0.00001)
    hr.add_argument("--variable-lr", type=bool, default=True, help="False if you want to have fixed learning rate")
    hr.add_argument('--batch-size', help="Training batch size", default=32, type=int)
    hr.add_argument('--epochs', help="Epochs, i.e., number of passes over the training set", type=int, default=200)
    hr.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    hr.add_argument('--gpu', type=str, default="0", help="if device is cuda, determines which gpu to use")
    hr.add_argument('--multi-gpu', type=bool, default=False, help="Use multiple GPUs")
    hr.add_argument('--debug', type=int, default=0, help="run in debug mode")
    hr.add_argument('--use-gpu', type=int, default=0, help="Running on GPU or CPU mode")
    hr.add_argument('--tech-used', type=str, default="T1036", help="tech to perform training for(only for stage 2)")
    hr.add_argument('--mask-prob', type=float, default=0.15)
    hr.add_argument('--data-parallel', type=int, default=0, help="To run mult-gpu setting using nn.DataParallel")
    hr.add_argument('--masking-scheme', type=int, default=0, help="(0: sequence masking, 1: random masking, 2: both)")
    # Model Parameters
    
    mr = parser.add_argument_group("Parameters Related to Neural Network Architecture")
    mr.add_argument('--embedding-dim', help="Embedding dimension added after the input layer", type=int, default=512)
    mr.add_argument('--penultimate-features-num', type=int, default=64,
                    help="This determines the number of neurons (i.e., features) at the last layer of the main body"
                         "of the neural network (before linear classifiers)")
    mr.add_argument('--num-cnn-filters', help="CNN filters", type=int,
                    default=100)
    mr.add_argument('--cnn-kernel-size', help="CNN kernel size", type=int,
                    default=3)
    mr.add_argument('--cnn-kernel-stride', help="CNN kernel stride", type=int,
                    default=3)
    mr.add_argument('--nheads', help="number of atention heads", type=int,
                    default=1)
    mr.add_argument('--nlayers', help="number of transformer layers", type=int,
                    default=2)
    mr.add_argument('--dropout', help="Dropout Probability", type=float, default=0.1) 
    mr.add_argument('--d_hid', type=int, help="The dimension of the feedforward network model for transformer encoder", default=2048)
    mr.add_argument('--load-weight', type=int, help="whether to load saved weight", default=0)
    mr.add_argument('--epoch-to-load', type=int, help="which epoch weight should be loaded", default=0)
    args = parser.parse_args()
    return args



def validation(MASK_TOKEN):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the total validation loss and number of correct predictions
    total_correct, total_masked = 0, 0
    total_masked = 0
    validation_loss = 0
    total_replaced_with_mask = 0  # Initialize the count of tokens replaced with the masked token
    total_replaced_with_random = 0
    # Disable gradient computation
    with torch.no_grad():
        #Validation Batch Loop
        for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(validation_dataloader), desc="validation"):  
            #print("within validation")
            # Compute the model's output
            model_input = batch_inputs[0].to(device, dtype=torch.int64)
            labels = batch_labels[0].to(device, dtype=torch.int64).view(-1) 
            logits = model(model_input)
            loss = criterion(logits, labels)
            batch_loss = loss.item()

            masked_tokens = labels != -100
            batch_masked = masked_tokens.sum().item()
            # if the number of api in sequence is too low then we dont mask anythin, but if this persisit for every sequence we have to append 0
            if batch_masked == 0:
                validation_loss += 0
            else:
                validation_loss += batch_loss

            
            # Calculate predictions by taking the argmax of logits
            predictions = logits.argmax(dim=1)

            # Calculate the number of correct predictions in this batch, only for unmasked tokens
            batch_correct = ((predictions == labels) & masked_tokens).sum().item()

            # Update total correct and total samples, only for unmasked tokens
            total_correct += batch_correct

            total_masked += batch_masked

            total_replaced_with_mask += (masked_tokens & (model_input.view(-1) == MASK_TOKEN)).sum().item()
            total_replaced_with_random += (masked_tokens & (model_input.view(-1) != MASK_TOKEN)).sum().item()

    # Compute the average validation loss and accuracy
    validation_accuracy = round((float(total_correct * 100) / total_masked), 2)
    validation_loss = (validation_loss / (batch_idx + 1))
    #t_loss.append(float("%.5f" % train_loss))
    #print(f"Total Masked Tokens: {total_masked}")
    #print(f"Total Input Tokens Replaced with [MASK]: {total_replaced_with_mask}")
    #print(f"Total Input Tokens Replaced with Random Value: {total_replaced_with_random}")
    return validation_loss, validation_accuracy


def train(MASK_TOKEN):
    model.train()
    #batch_counter = 0
    total_correct, total_masked = 0, 0
    total_masked = 0
    train_loss = float(0)
    total_replaced_with_mask = 0  # Initialize the count of tokens replaced with the masked token
    total_replaced_with_random = 0 
    #loss_values = []
    batches = 0
    for batch_idx, (batch_inputs, batch_labels) in tqdm(enumerate(train_dataloader), desc="training"):  
        optimizer.zero_grad()
        batches += 1
        #print(batch_idx)
        model_input = batch_inputs[0].to(device, dtype=torch.int64)
        labels = batch_labels[0].to(device, dtype=torch.int64).view(-1) 
        logits = model(model_input)
        loss = criterion(logits, labels)
        
        batch_loss = loss.item()
        loss.backward()
        optimizer.step()

        masked_tokens = labels != -100
        batch_masked = masked_tokens.sum().item()
        # if the number of api in sequence is too low then we dont mask anythin, but if this persisit for every sequence we have to append 0
        if batch_masked == 0:
            train_loss += 0
        else:
            train_loss += batch_loss
        
        # Calculate predictions by taking the argmax of logits
        predictions = logits.argmax(dim=1)

        # Calculate the number of correct predictions in this batch, only for unmasked tokens
        batch_correct = ((predictions == labels) & masked_tokens).sum().item()

        # Update total correct and total samples, only for unmasked tokens
        total_correct += batch_correct
        
        total_masked += batch_masked
        
        total_replaced_with_mask += (masked_tokens & (model_input.view(-1) == MASK_TOKEN)).sum().item()
        total_replaced_with_random += (masked_tokens & (model_input.view(-1) != MASK_TOKEN)).sum().item()
        
        my_lr_scheduler.step()
    
    train_accuracy = round((float(total_correct * 100) / total_masked), 2)
    train_loss = (train_loss / (batch_idx + 1))
    
    return train_loss, train_accuracy



def load():
    global model, train_dataloader, validation_dataloader, criterion, optimizer, my_lr_scheduler
    epochs = args.epochs

    #output_path = create_output_dir(args.output_path, args.test_no)
   
    output_path = os.path.join(args.output_path)
    all_args = {"embedding": args.embedding_dim, "num_encoder": args.nlayers, "num_attention": args.nheads, "initial_lr": str(args.lr), \
                "sequence": args.sequence_budget, "api": args.api_padding, "masking_scheme": args.masking_scheme}
    
    print("parameters: {}".format(all_args))
    print("masking scheme used: {}".format(args.masking_scheme))
    with open(os.path.join(output_path, "parameters.json",), "w") as my_file:
        json.dump(all_args, my_file)
    print("written parameters to {}".format(os.path.join(output_path, "parameters.json")))
    
    #fetching input, output, unique_input_token, unique_output_labels, hash_list
    X_train, X_validation, total_tokens, unique_apis, special_tokens = fetch_features(args)
    print("special tokens: {}".format(special_tokens))
    #initializing the model
    model = MaskedLearningModel(embedding_dim=args.embedding_dim, nhead=args.nheads, 
                                            dim_feedforward=args.d_hid, num_layers=args.nlayers, 
                                            dropout_prob=args.dropout, vocab_size=total_tokens, 
                                            device=device, token_padding=special_tokens["<PAD>"], unique_apis=len(unique_apis)).to(device)
    model.to(device)
    
    if args.multi_gpu:
        if args.data_parallel == 1:
            model = nn.DataParallel(model, device_ids=[0,1])
            #torch.distributed.barrier()
        else:
            world_size = os.environ['WORLD_SIZE']
            rank = int(os.environ['LOCAL_RANK'])
            model.to(rank)
            #torch.distributed.barrier()
            # model = DDP(model,device_ids = [device], find_unused_parameters=True)
            # model = Model().to(rank)
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    model.to(device)
    
    if args.multi_gpu == 1 and args.data_parallel == 0:
        print("distributed sampler for DDP")
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        #Masking
        train_masker = MLMDataset(X, len(unique_apis), args.mask_prob, special_tokens, SEED)
        #Sampling
        train_sampler = DistributedSampler(train_masker, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        train_dataloader = DataLoader(train_masker, num_workers=20, sampler=train_sampler, pin_memory=False, batch_size=1)
        #val_dataloader = DataLoader(val_combined_dataset,  num_workers=16, sampler=val_sampler, pin_memory=False, batch_size=args.batch_size)
    else:
        train_masker = MLMDataset(X_train, len(unique_apis), args.mask_prob, special_tokens, SEED, args.masking_scheme)
        train_dataloader = DataLoader(train_masker, num_workers=20, batch_size=1, shuffle=False)  
        #train_dataloader = DataLoader(train_masker, batch_size=1, shuffle=False) 
        validation_masker = MLMDataset(X_validation, len(unique_apis), args.mask_prob, special_tokens, SEED, args.masking_scheme)
        validation_dataloader   = DataLoader(validation_masker, num_workers=12, batch_size=1, shuffle=False) 
        #validation_dataloader = DataLoader(validation_masker, batch_size=, shuffle=False)
        
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas = (0.9, 0.99), eps=1e-6, weight_decay=0.01)
    #print("warmup steps: {}".format(warmup_steps))
    warmup_steps = len(train_masker) * (args.epochs - args.epoch_to_load)
    #initializing optimzer
    lr_lambda = lambda step: min(step / (0.1 * warmup_steps), 1.0)
    # Create the scheduler using LambdaLR
    my_lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    if args.load_weight==1:
        print("loading saved weights")
        architecuture_name = "epoch_{}_nheads_{}_d_hid_{}_embedding_dim_{}_nlayers_{}_dropout_{}".format(args.epoch_to_load, args.nheads, args.nheads,args.embedding_dim,args.nlayers,args.dropout )
        architecure_dir = os.path.join(output_path, architecuture_name)
        checkpoint_file = [f for f in os.listdir(architecure_dir) if ".pt" in f]
        checkpoint_file_path = os.path.join(architecure_dir, checkpoint_file[0])
        checkpoint = torch.load(checkpoint_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        my_lr_scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch'] + 1
        print("continuing from epoch :{}".format(epoch))
        with open(os.path.join(architecure_dir,"best_result.json"), "r" ) as my_file:
            best_result = json.load(my_file)

        best_training_loss = best_result["train_loss"]
        print("current train_loss:{}".format(best_training_loss))
    else:
        epoch = 1
        best_training_loss = float('inf')
    
    result_per_epoch = {}
    print("total epochs: {}, starting from : {}".format(args.epochs, epoch))
    
    best_result = {}

    # for epoch in range(args.epochs):
    while epoch <= args.epochs:
        
        train_loss, train_accuracy = train(special_tokens["<MASK>"])
       
        validation_loss, validation_accuracy = validation(special_tokens["<MASK>"])
        print("Epoch: {}, train_loss: {}, train accuracy: {}, validation_loss: {}, validation accuracy: {}".format(epoch, round(train_loss, 4), train_accuracy, round(validation_loss, 4), validation_accuracy))
        #print("Epoch: {}, validation_loss: {}, validation accuracy: {}".format(epoch, round(validation_loss, 4), validation_accuracy))

    
        result_per_epoch[epoch] = {"training_loss":train_loss, "training_accuracy": train_accuracy, "validation_loss":validation_loss, "validation_accuracy": validation_accuracy,}
        print("validation: {}".format(result_per_epoch[epoch]))
        if  best_training_loss > train_loss:
            print("best epoch changed to : {}".format(epoch))
            best_training_loss = train_loss
            best_result["epoch"] = epoch
            best_result["val_accuracy"] = validation_accuracy
            best_result["val_loss"] = validation_loss
            best_result["train_accuracy"] = train_accuracy
            best_result["train_loss"] = train_loss
        
            output_path = args.output_path
            
            if epoch < 50:
                continue
            
            model_arch = "epoch_"+ str(epoch) + "_nheads_" + str(args.nheads) + "_d_hid_" + str(args.nheads) + "_embedding_dim_" + str(args.embedding_dim) + "_nlayers_" + str(args.nlayers) + "_dropout_" + str(args.dropout)  + "/"
            output_path = os.path.join(output_path, model_arch)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            with open("{}".format(os.path.join(output_path,"best_result.json")), "w") as my_file:
                json.dump(best_result, my_file)
            print("best result written to: {} ".format(os.path.join(output_path,"best_result.json")))
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler': my_lr_scheduler.state_dict()}, "{}".format(os.path.join(output_path, "_saved_wt.pt")))
            print("weights saved to: {} ".format(os.path.join(output_path,"saved_wt.pt")))

            with open("{}".format(os.path.join(output_path,"all_result.json")), "w") as my_file:
                json.dump(result_per_epoch, my_file)
        epoch += 1
    import IPython
    IPython.embed()
    assert False


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    #required for reading
    mp.set_sharing_strategy('file_system')
    
    import torch.backends.cudnn as cudnn
    if args.device == "cuda":
        if args.multi_gpu==1:
            device = torch.device("cuda")
            if args.data_parallel==0:
                # dist.init_process_group(backend='nccl')
                LOCAL_RANK = int(os.environ['LOCAL_RANK'])
                WORLD_SIZE = int(os.environ['WORLD_SIZE'])
                WORLD_RANK = int(os.environ['RANK'])
                device = torch.device("cuda:{}".format(LOCAL_RANK))
                dist.init_process_group(backend = 'nccl', rank=WORLD_RANK, world_size=WORLD_SIZE, timeout = timedelta(seconds=9000))
            else:
                gpu = args.gpu
                device=torch.device("cuda:{}".format(gpu))
        else:
            gpu = args.gpu
            print("Using single GPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            device = torch.device("cuda:{}".format(gpu))
        #cudnn.benchmark = True
    else:
        print("running on CPU")
        device="cpu"
    
    #os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
    print("device is : {}".format(device))
    load()

