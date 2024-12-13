
import json
import os
from tqdm import tqdm

'''
    Fetches unique mitre technique, new_techs and tech_histogram
'''

def fetch_unique_apis(inputs):
    unique = []
    for sequence in inputs:
        for api in sequence:
            if api not in unique:
                unique.append(api) 

def get_data(input_dataset, labels_dataset):
    dataset_files = {}
    
    all_input_files = os.listdir(input_dataset)
    #all_labels_files = os.listdir(labels_dataset)
    #for sample_file
    inputs = []
    labels = []
    hashes = []
    for sequence_file in tqdm(all_input_files, desc="processing input"):
        sample_hash = sequence_file.split(".json")[0]
        with open(os.path.join(input_dataset, sequence_file), "r") as my_file:
            sequence_data = json.load(my_file)
            hashes.append(sample_hash)
            inputs.append([sequence_data[idx] for idx in sequence_data])
        with open(os.path.join(labels_dataset, sequence_file), "r") as my_file:
            labels.append(json.load(my_file))
            

    return labels, inputs, hashes



def fetch_features(args):
    num_sequences = 350
    api_size = 20
    unique_api_path = args.unique_api_path
    #unique_apis = []
    input_path = args.input_path
    labels_path = args.labels_path
    metadata_path = args.metadata_path
    labels, inputs, hashes = get_data(input_path, labels_path)
    with open(metadata_path, "r") as my_file:
        metadata = json.load(my_file)
    
    X = list()
    #fetching unqiue APIs from unique_api_path
    #all_input_files = os.listdir(dataset_path)
    api_mapping = {}
    unique_apis = []
    print("starting API mapping")
    with open(unique_api_path) as my_file:
        unique_apis = my_file.read().split("\n")
        api_idx = 0
        for api in unique_apis:
            if api not in api_mapping:
                api_mapping[api] = api_idx
                api_idx += 1
            else:
                continue
    print("Finished API mapping")
    special_tokens = {"<PAD>": len(api_mapping), 
                    "<MASK>": len(api_mapping) + 1,
                    "<SEQ>": len(api_mapping)+ 2, 
                    "</SEQ>": len(api_mapping) + 3,
                    "<CLASS>": len(api_mapping)+ 4}
   

    total_tokens = len(api_mapping) + len(special_tokens)

    SEQUENCE_PADDING = [special_tokens["<PAD>"]] * api_size
    
    for data in tqdm(inputs, desc='mapping input files'):
    
        sample_input = list()
        for sequence in data:
            mapped_sequence = [api_mapping[api] for api in sequence]
            if len(mapped_sequence) < api_size:
                sample_input.append([special_tokens["<SEQ>"]] + mapped_sequence + SEQUENCE_PADDING[0: api_size - len(mapped_sequence)] + [special_tokens["</SEQ>"]])
            else:
                sample_input.append([special_tokens["<SEQ>"]] + mapped_sequence[0:api_size] + [special_tokens["</SEQ>"]])

        X.append(sample_input)
    
    if debug == 1:
        print("debug mode")
        X_training =[X[i] for i in metadata["train_index"][0:100]]
        X_validation = [X[i] for i in metadata["validation_index"][0:10]]

    else:
        X_training =[X[i] for i in metadata["train_index"]]
                     
        X_validation = [X[i] for i in metadata["validation_index"]]

    
    return X_training, X_validation, total_tokens, unique_apis, special_tokens