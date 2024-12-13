import os
import sys
from graph_reducer import GraphReduction
from sequence_extractor import ApisExtraction
from producer import Stage2Producer, Stage3Producer, Stage2TimeShiftProducer, Stage3TimeShiftProducer
import multiprocessing as mp
import threading
import psutil
import time
import pymongo
import gridfs
import json
import bson.json_util as json_util
from search_db import (
    ReduceDB, 
    ApiExtractDB, 
    APIExtractDBTimeShift, 
    ReduceDBTimeShift, 
    ReduceDBx64
)
# from merging_sample_environments import merge_environments
from tqdm import tqdm
import argparse

'''
High Level Approach:
1. Clean the control flow (CF) information:
   - Check for repeating functions and handle them (completed)
2. Create a dictionary mapping function addresses to block addresses and calls (completed)
3. Initialize stack array and API sequence array
4. Select a random function address
5. Build a graph for the given function
6. Extract multiple Depth-First Search (DFS) paths for the chosen function
7. Randomly select a DFS path
8. Traverse through each block in the selected path:
   - If an API call exists, append it to a list
   - If a function call exists:
     - Store the current path state in a custom data type (e.g., stack)
     - Go back to Step 5 with the argument of the next function call
9. If the length of the API list is less than 200, abort execution and return to Step 4.
'''

def api_extraction(sample_hash, sample_data, sequences, api):
    """Extract APIs from the given sample data."""
    obj = ApisExtraction(sample_hash, sample_data, sequences, api)
    ret = obj.api_extractor()

def graph_reduction(sample_hash, sample_data):
    """Reduce the graph for the given sample data."""
    obj = GraphReduction(sample_hash, sample_data)
    ret = obj.graph_reducer()

def graph_reducer_sample_hashes(sample_hashes_path):
    """Load and return a dictionary mapping sample hashes to file IDs."""
    with open(sample_hashes_path, "r") as my_file:
        hash_to_file_id = json.load(my_file)
    return hash_to_file_id

def iterator():
    """Main iteration function to process samples and perform API extraction."""
    
    # Load the hash to file ID mapping
    hash_to_file_id = graph_reducer_sample_hashes(args.sample_hashes_path)
    print("Total samples: {}".format(len(hash_to_file_id)))
    
    # Initialize database connection for ReduceDBTimeShift
    db = ReduceDBTimeShift()
    
    # Create a list of hashes from the loaded data
    hash_list = [i for i in hash_to_file_id]
    
    print("MODE: {}".format(args.mode))
    
    # Iterate over each hash in the hash list
    for itr in tqdm(range(len(hash_list))):
        # Check if the CPU usage is below the threshold before processing
        if psutil.cpu_percent() < 80.00:
            sample_hash = hash_list[itr]
            
            try:
                # Load the sample data from the database
                sample_data = json.loads(db.fs.get(json_util.loads(json.dumps(hash_to_file_id[sample_hash]))).read())
                # Skip if the data is empty
                if not sample_data["data"]:
                    continue
            except Exception as e:
                print(e)
                continue
            
        else:
            # If CPU usage is high, wait for a while before retrying
            print("Sleeping due to high CPU usage")
            time.sleep(5)

    # Wait for a while before exiting
    time.sleep(100)
    sys.exit(0)

def parse_args():
    """Parse input arguments for mode, number of APIs, and sequences."""
    parser = argparse.ArgumentParser(description="GraphReduction & API Extraction")
    ir = parser.add_argument_group("parameters")
    ir.add_argument(
        '--mode', default='reduce', type=str, help="Mode: reduce or extract"
    )
    ir.add_argument(
        '--apis', default=30, type=str, help="Number of APIs to extract for each sequence"
    )
    ir.add_argument(
        '--sequences', default=300, type=str, help="Number of sequences to extract for each sample"
    )
    ir.add_argument(
        '--sample_hashes_path', default=300, type=str, help="Path to the keys of the saved db"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    iterator()
