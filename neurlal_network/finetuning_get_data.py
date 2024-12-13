
import json
import os
from tqdm import tqdm
from forecast_get_data import forecast_initialize_dataset
from forecast_get_data import forecast_create_inputs
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
        #extracting labels
        # with open(os.path.join(labels_dataset, sequence_file), "r") as my_file:
        #     labels.append(json.load(my_file))
            
    labels  = ""
    return labels, inputs, hashes



def fetch_features(debug, technique, input_path, labels_path):
    num_sequences = 350
    api_size = 20
    unique_api_path = "unique_apis.txt"
    labels, inputs, hashes = get_data(input_path, labels_path)    
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
    
    
    with open("training_index_dict.json", "r") as my_file:
        training_index_dict = json.load(my_file)
    
    
    count = 0
    
    for data in tqdm(inputs, desc='mapping input files'):
    
        sample_input = list()
        for sequence in data:
            mapped_sequence = [api_mapping[api] for api in sequence]
            if len(mapped_sequence) < api_size:
                #sample_input.append([special_tokens["<SEQ>"]] + mapped_sequence + SEQUENCE_PADDING[0: api_size - len(mapped_sequence)] + [special_tokens["</SEQ>"]])
                sample_input.append(mapped_sequence + SEQUENCE_PADDING[0: api_size - len(mapped_sequence)] + SEQUENCE_PADDING[0:2])
            else:
                sample_input.append(mapped_sequence[0:api_size] +SEQUENCE_PADDING[0:2])
                #sample_input.append([special_tokens["<SEQ>"]] + mapped_sequence[0:api_size] + [special_tokens["</SEQ>"]])
        if len(sample_input) < num_sequences:
            count += 1
            # Making sure dim of input is 350x22
            sample_input +=  [SEQUENCE_PADDING + SEQUENCE_PADDING[0:2]] * (num_sequences - len(sample_input))
        X.append(sample_input)
    
    
    with open("metadata_path", "r") as my_file:
        meta_data_2 = json.load(my_file)

    
    X_training =[X[i] for i in training_index_dict[technique]['positive']] 
    X_training += [X[i] for i in training_index_dict[technique]['negative']]
    
    train_positive = len(training_index_dict[technique]['positive'])
    train_negative = len(training_index_dict[technique]['negative'])
    Y_training = [1] * train_positive  + [0] * train_negative       
    
    print("Training: positive: {}, negative: {}".format(train_positive, train_negative))
    len("total labels: {}".format(train_positive + train_negative))
    
    X_validation = []
    
    validation_index_positive = meta_data_2["validation"][technique]["present"]["present"] + meta_data_2["validation"][technique]["present"]["not_present"] + meta_data_2["validation"][technique]["not_present"]["present"] 
    validation_index_negative =  meta_data_2["validation"][technique]["not_present"]["not_present"] 
    validation_positive = len(validation_index_positive)
    validation_negative = len(validation_index_negative)
    X_validation = [X[i] for i in validation_index_positive]
    X_validation += [X[i] for i in validation_index_negative]
    Y_validation = ([1] * len(validation_index_positive)) + ( [0] * len(validation_index_negative))
    
    print("Validation: positive: {}, negative: {}".format(validation_positive, validation_negative))

    testing_index_positive = meta_data_2["testing"][technique]["present"]["present"] + meta_data_2["testing"][technique]["present"]["not_present"] + meta_data_2["testing"][technique]["not_present"]["present"] 
    testing_index_negative = meta_data_2["testing"][technique]["not_present"]["not_present"] 
    
    X_testing = []
    X_testing = [X[i] for i in testing_index_positive]
    X_testing += [X[i] for i in testing_index_negative]
    
    Y_testing  = [1] * len(testing_index_positive)  + [0] * len(testing_index_negative)
    testing_positive = len(testing_index_positive)
    testing_negative = len(testing_index_negative)
    print("Testing: positive: {}, negative: {}".format(testing_positive, testing_negative))

    
    return X_training, X_validation, X_testing, Y_training, Y_validation, Y_testing, total_tokens, unique_apis, special_tokens
