import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import os
import json
from tqdm import tqdm

class FinetuningDataset(Dataset):
    def __init__(self, args, split="train"):
        #self.seed = seed
        #self.masking_scheme = args.masking_scheme
        self.num_sequences = args.sequence_budget
        self.sequence_length = args.sequence_length
        self.unique_api_path = args.unique_api_path
        self.tech_used = args.tech_used
        self.vocab_size = 0
        self.vocabulary = {}
        self.data = []
        self.tech_used = args.tech_used
        self.special_tokens = ["<PAD>", "<MASK>", "<START>", "</START>", "<UNK>"]
        self.vocabulary = {}
        self.datalist = []
        self.lables = []
        self.sample_hashes = []

        with open(self.unique_api_path, "r") as my_file:
            self.unique_apis = my_file.read().split("\n")
        self.input_path = args.input_dir
        self.datalist_dir = os.path.join("./dataset")
        if not os.path.exists( self.datalist_dir):
            os.makedirs( self.datalist_dir)

        if not os.path.exists(os.path.join(self.datalist_dir, "vocabulary.json")):
            raise FileNotFoundError("Vocabulary file not found at {}".format(os.path.join(self.datalist_dir, "vocabulary.json")))

        with open(os.path.join(self.datalist_dir, "vocabulary.json"), "r") as my_file:
            self.vocabulary = json.load(my_file)
            self.vocab_size = len(self.vocabulary)
        
        if not os.path.exists(os.path.join(self.datalist_dir, "fine_tuning", "{}/{}.json".format(self.tech_used, split))):
            raise FileNotFoundError("{} dataset not found at {}".format(split, os.path.join(self.datalist_dir, "{}/{}.json".format(self.tech_used, split))))

        with open(os.path.join(self.datalist_dir, "fine_tuning", "{}/{}.json".format(self.tech_used, split)), "r") as my_file:
            data = json.load(my_file)
            self.data = data

        for sample_hash in self.data:
            self.sample_hashes.append(sample_hash)
            if self.data[sample_hash]["output"]:
                label = 1
            else:
                label = 0
            api_sequences = self.data[sample_hash]["input"]
            tokenized_sample_sequences = []
            for sequence in api_sequences:
                tokenized_sequence = []
                for api in sequence:
                    if api in self.vocabulary:
                        tokenized_sequence.append(self.vocabulary[api])
                    else:
                        tokenized_sequence.append(self.vocabulary["<UNK>"])
                if len(tokenized_sequence) < self.sequence_length:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence +  [self.vocabulary["</START>"]] + [self.vocabulary["<PAD>"]] * (self.sequence_length - len(tokenized_sequence))
                else:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence[:self.sequence_length] + [self.vocabulary["</START>"]]
                tokenized_sample_sequences.append(tokenized_sequence)
            if len(tokenized_sample_sequences) < self.num_sequences:
                # This is to make sure that the number of sequences in each sample is equal to self.num_sequences
                tokenized_sample_sequences += [[self.vocabulary["<START>"]] + [self.vocabulary["</START>"]] + [self.vocabulary["<PAD>"]] * self.sequence_length] * (self.num_sequences - len(tokenized_sample_sequences))
            self.datalist.append([tokenized_sample_sequences, label, sample_hash])
        
    
    def __len__(self):
        return len(self.data)    


    def __getitem__(self, idx):
        sequences = self.datalist[idx][0]
        labels = self.datalist[idx][1]
        sample_hash = self.datalist[idx][2]

        # Tensorize the sequences and labels
        sequences = torch.tensor(sequences, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return sequences, labels, sample_hash
        