import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import os
import json
from tqdm import tqdm
# Mask1: sequence masking
# Mask2: sequence and random masking
# Mask3: sequence and random with seed (randomly chosen)


class MLMDataset(Dataset):
    def __init__(self, args, seed, split="train"):
        self.mask_prob = args.mask_prob
        self.seed = seed
        self.masking_scheme = args.masking_scheme
        self.num_sequences = args.sequence_budget
        self.sequence_length = args.sequence_length
        self.unique_api_path = args.unique_api_path
        self.vocab_size = 0
        self.vocabulary = {}
        self.data = []
        self.special_tokens = ["<PAD>", "<MASK>", "<START>", "</START>", "<UNK>"]

        with open(self.unique_api_path, "r") as my_file:
            self.unique_apis = my_file.read().split("\n")
        self.input_path = args.input_dir
        self.datalist_dir = os.path.join("./dataset")
        if not os.path.exists( self.datalist_dir):
            os.makedirs( self.datalist_dir)
        
        if not os.path.exists(os.path.join(self.datalist_dir, "{}_datalist.json".format(split))):
            all_files = os.listdir(self.input_path)
            split_ratio = args.training_set_ratio 
            split_idx = int(len(all_files) * split_ratio)
            train_file_names = random.sample(all_files, split_idx)
            validation_file_names = list(set(all_files) - set(train_file_names))
            self.training_sequences = []
            self.validation_sequences = []
            unique_train_tokens = set()

            for idx, file_name in tqdm(enumerate(train_file_names), desc="Reading Training Files"):
                with open(os.path.join(self.input_path, file_name), "r") as my_file:
                    data = json.load(my_file)
                    
                    # Assuming that 'api_sequences' key contains list of sequences
                    for itr in data['api_sequences']:
                        sequence = []
                        for token in data['api_sequences'][itr]:
                            if token in self.unique_apis:
                                unique_train_tokens.add(token)
                                sequence.append(token)
                            else:
                                print("Token not in unique apis: ", token)
                        if len(sequence) > 0:
                            self.training_sequences += [sequence]       

            for idx, file_name in tqdm(enumerate(validation_file_names), desc="Reading Validation Files"):
                with open(os.path.join(self.input_path, file_name), "r") as my_file:
                    data = json.load(my_file)
                    for itr in data['api_sequences']:
                        sequence = []
                        for token in data['api_sequences'][itr]:
                            if token in self.unique_apis:
                                sequence.append(token)
                        if len(sequence) > 0:
                            self.validation_sequences += [sequence]
                            
            
            ## Tokenization
            self.vocabulary = {token: idx for idx, token in enumerate(unique_train_tokens)}
            for token in self.special_tokens:
                self.vocabulary[token] = len(self.vocabulary)
            
            print("Vocabulary Size: ", len(self.vocabulary))    
            
            self.train_datalist = []
            self.validation_datalist = []

            
            for idx, sequence in tqdm(enumerate(self.training_sequences), desc="Tokenizing Training Sequences"):
                tokenized_sequence = [self.vocabulary[token] for token in sequence]
                # Adding start and end tokens
                if len(tokenized_sequence) < self.sequence_length:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence +  [self.vocabulary["</START>"]] + [self.vocabulary["<PAD>"]] * (self.sequence_length - len(tokenized_sequence))
                else:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence[:self.sequence_length] + [self.vocabulary["</START>"]]
                self.train_datalist.append(tokenized_sequence)
            

            ## For validation it is possible that some tokens are not present in training set so we will use <UNK> token for them
            for idx, sequence in tqdm(enumerate(self.validation_sequences), desc="Tokenizing Validation Sequences"):
                tokenized_sequence = []
                for token in sequence:
                    if token in self.vocabulary:
                        tokenized_sequence.append(self.vocabulary[token])
                    else:
                        tokenized_sequence.append(self.vocabulary["<UNK>"])
                if len(tokenized_sequence) < self.sequence_length:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence +  [self.vocabulary["</START>"]] + [self.vocabulary["<PAD>"]] * (self.sequence_length - len(tokenized_sequence))
                else:
                    tokenized_sequence = [self.vocabulary["<START>"]] + tokenized_sequence[:self.sequence_length] + [self.vocabulary["</START>"]]
                self.validation_datalist.append(tokenized_sequence)
            
            # We will now save training datalist, validation datalist and vocabulary
            with open(os.path.join(self.datalist_dir, "train_datalist.json"), "w") as my_file:
                json.dump(self.train_datalist, my_file)
            
            with open(os.path.join(self.datalist_dir, "validation_datalist.json"), "w") as my_file:
                json.dump(self.validation_datalist, my_file)
            
            with open(os.path.join(self.datalist_dir, "vocabulary.json"), "w") as my_file:
                json.dump(self.vocabulary, my_file)

        with open(os.path.join(self.datalist_dir, "{}_datalist.json".format(split)), "r") as my_file:
            self.data = json.load(my_file)
        
        with open(os.path.join(self.datalist_dir, "vocabulary.json"), "r") as my_file:
            self.vocabulary = json.load(my_file)
            self.vocab_size = len(self.vocabulary)

    
    def __len__(self):
        return len(self.data)

    # maskes token randomly
    def mask_tokens_2(self, inputs):
        # Convert inputs to a tensor if not already
        sequence = torch.tensor(inputs)
        labels = torch.full_like(sequence, -100)  # Initialize labels with -100 (ignore index)

        # Create a mask for special tokens
        special_token_mask = (
            (sequence == self.vocabulary["<PAD>"]) |
            (sequence == self.vocabulary["<START>"]) |
            (sequence == self.vocabulary["</START>"])
        )

        # Create a mask for non-special tokens
        non_special_mask = ~special_token_mask

        # Get the indices of non-special tokens
        non_special_indices = torch.nonzero(non_special_mask, as_tuple=False).squeeze(-1)

        # Determine how many tokens to mask
        total_tokens = len(non_special_indices)
        num_tokens_to_mask = int(total_tokens * self.mask_prob)

        if num_tokens_to_mask > 0:
            # Randomly select positions to mask from the non-special tokens
            mask_indices = random.sample(non_special_indices.tolist(), num_tokens_to_mask)

            # Apply masking and handle labels
            for idx in mask_indices:
                if random.random() < 0.5:
                    # Replace with the <MASK> token
                    sequence[idx] = self.vocabulary["<MASK>"]
                else:
                    # Replace with a random token
                    sequence[idx] = random.randint(0, self.vocab_size - 1)
                
                # Set the label to the original token
                labels[idx] = inputs[idx]

        return sequence, labels


    # Masks token randomly but in sequence
    def mask_tokens_1(self, inputs):
        if self.seed is not None:
            random.seed(self.seed)
        sequence = torch.tensor(inputs)
        labels = torch.full_like(sequence, -100)

        special_token_mask = (
            (sequence == self.vocabulary["<PAD>"]) |
            (sequence == self.vocabulary["<START>"]) |
            (sequence == self.vocabulary["</START>"])
        )

        non_special_mask = ~special_token_mask
        non_special_indices = torch.nonzero(non_special_mask, as_tuple=False).squeeze(-1)
        total_tokens = len(non_special_indices)
        num_tokens_to_mask = int(total_tokens * self.mask_prob)

        if num_tokens_to_mask > 0:
            non_special_indices = non_special_indices.tolist()
            start_idx = random.randint(0, len(non_special_indices) - num_tokens_to_mask)
            selected_indices = non_special_indices[start_idx:start_idx + num_tokens_to_mask]
            for idx in selected_indices:
                if random.random() < 0.5:
                    sequence[idx] = self.vocabulary["<MASK>"]
                    labels[idx] = inputs[idx]
                else:
                    # We want to exclude special tokens from the random replacement
                    replacement_token = random.randint(0, self.vocab_size - 5)
                    sequence[idx] = replacement_token
                    labels[idx] = inputs[idx]
        else:
            pass

        return sequence, labels
    

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        
        if self.masking_scheme == 1:
            masked_input_ids, labels = self.mask_tokens_1(input_ids)
        elif self.masking_scheme==2:
            masked_input_ids, labels = self.mask_tokens_2(input_ids)
        elif self.masking_scheme==3:
            if random.random() < 0.5:
                masked_input_ids, labels = self.mask_tokens_1(input_ids)
            else:
                masked_input_ids, labels = self.mask_tokens_2(input_ids)
        else:
            raise ValueError("Invalid masking scheme")
        
        return masked_input_ids, labels