import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random


# Mask1: just sequence with seed 0.00001lr
# Mask4: just sequence with without seed: 0.0001lr

# Mask2: sequence and random without seed 0.0001lr
# Mask3: sequence and random with seed head=4, layer=4 0.00001lr
# Mask6: sequence and random without seed nlayer=8 head=12, layer=4 0.00001lr

# Mask5: random without seed 0.0001lr

class MLMDataset(Dataset):
    def __init__(self, data, total_apis, mask_prob, special_tokens, seed, masking_scheme):
        self.data = data  # List of 2D lists, each inner list represents a sequence of tokens
        self.vocab_size = total_apis
        self.mask_prob = mask_prob
        self.special_tokens = special_tokens
        self.seed = seed 
        self.masking_scheme = masking_scheme
    def __len__(self):
        return len(self.data)

    # maskes token randomly
    def mask_tokens_2(self, inputs):
        # if self.seed is not None:
        #     random.seed(self.seed)
        masked_inputs = torch.tensor(inputs)
        labels = torch.full_like(masked_inputs, -100)

        # Iterating through each sequence
        for idx, sequence in enumerate(masked_inputs):
            # Create a mask for special tokens
            special_token_mask = (
                (sequence == self.special_tokens["<PAD>"]) |
                (sequence == self.special_tokens["<SEQ>"]) |
                (sequence == self.special_tokens["</SEQ>"])
            )

            # Create a mask for non-special tokens
            non_special_mask = ~special_token_mask

            # Count the total number of non-special tokens
            total_tokens = torch.sum(non_special_mask)
            num_token_to_mask = int(total_tokens * self.mask_prob)
            visited = []
            if num_token_to_mask > 0:
                tokens_masked = 0
                non_special_indices = torch.arange(len(sequence))[non_special_mask]
                while tokens_masked < num_token_to_mask:
                    
                    chosen_idx = random.randint(0, len(non_special_indices)-1)
                    selected_index = non_special_indices[chosen_idx]

                    if selected_index in visited:
                        continue
                    else:
                        # Adding to visited list so that we do not mask same token twice
                        visited.append(selected_index)
                        non_special_indices
                        tokens_masked += 1
                        if random.random() < 0.5:
                            masked_inputs[idx][selected_index] = self.special_tokens["<MASK>"]
                            labels[idx][selected_index] = inputs[idx][selected_index]
                        else:
                            replacement_token = random.randint(0, self.vocab_size - 1)
                            masked_inputs[idx][selected_index] = replacement_token
                            labels[idx][selected_index] = inputs[idx][selected_index]
            else:
                # If there does not exist enough token to mask, we leave it as it is
                continue
        return masked_inputs, labels



    # Masks token randomly but in sequence
    def mask_tokens_1(self, inputs):
        if self.seed is not None:
            random.seed(self.seed)
        masked_inputs = torch.tensor(inputs)
        labels = torch.full_like(masked_inputs, -100)

        for idx, sequence in enumerate(masked_inputs):
            # Create a mask for special tokens
            special_token_mask = (
                (sequence == self.special_tokens["<PAD>"]) |
                (sequence == self.special_tokens["<SEQ>"]) |
                (sequence == self.special_tokens["</SEQ>"])
            )

            # Create a mask for non-special tokens
            non_special_mask = ~special_token_mask

            # Count the total number of non-special tokens
            total_tokens = torch.sum(non_special_mask)
            num_token_to_mask = int(total_tokens * self.mask_prob)

            if num_token_to_mask > 0:
                # Get the indices of the tokens to mask in the continuous sequence
                non_special_indices = torch.arange(len(sequence))[non_special_mask]
                # Choose a random starting point for continuous masking  
                start_idx = random.randint(0, len(non_special_indices) - num_token_to_mask)
                selected_indices = non_special_indices[start_idx:start_idx + num_token_to_mask]
                for i in selected_indices: 
                    if random.random() < 0.5:
                        masked_inputs[idx][i] = self.special_tokens["<MASK>"]
                        labels[idx][i] = inputs[idx][i]
                    else:
                        # Replace with a random token that is not a special token
                        #print("replacement")
                        replacement_token = random.randint(0, self.vocab_size - 1)
                        masked_inputs[idx][i] = replacement_token
                        labels[idx][i] = inputs[idx][i]
            else:
                continue
        return masked_inputs, labels

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        if self.masking_scheme == 0:
            masked_input_ids, labels = self.mask_tokens_1(input_ids)
        elif self.masking_scheme==1:
            masked_input_ids, labels = self.mask_tokens_2(input_ids)
        else:
            if random.random() < 0.5:
                masked_input_ids, labels = self.mask_tokens_1(input_ids)
            else:
                masked_input_ids, labels = self.mask_tokens_2(input_ids)
        return masked_input_ids, labels