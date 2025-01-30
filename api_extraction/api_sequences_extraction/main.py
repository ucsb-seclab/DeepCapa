import os
import sys
from graph_reducer import GraphReduction
import multiprocessing as mp
import psutil
import time
import json
from search_db import ReduceDB
import argparse


def graph_reduction(sample_hash, sample_data, api_sequences_budget, api_calls_budget, output_dir=None):
    """Reduce the graph for the given sample data."""
    if output_dir is None:
        obj = GraphReduction(sample_hash, sample_data, api_sequences_budget=api_sequences_budget, api_calls_budget=api_calls_budget)
    else:
        obj = GraphReduction(sample_hash, sample_data, api_sequences_budget=api_sequences_budget, api_calls_budget=api_calls_budget, output_dir=output_dir)
    ret = obj.graph_reducer()



def iterator():
    """Main iteration function to process samples and perform API extraction."""
    
    if args.use_mongo:
        db = ReduceDB()
        all_files = db.fs.list()
        itr = 0
        while itr < len(all_files):
            if psutil.cpu_percent() < 80.00:
                sample_data = db.fs.find_one({"filename": all_files[itr]})
                if sample_data is None:
                    print("Error: No data found for file: {}".format(all_files[itr]))
                    itr += 1
                    continue
                
                sample_data = json.loads(sample_data.read())
                sample_hash = sample_data["hash"]
                proc = mp.Process(target=graph_reduction, args=(sample_hash, sample_data, args.api_sequences, args.api_calls))
                proc.start()
                itr += 1

            else:
                print("Sleeping due to high CPU usage")
                time.sleep(10)            
    else:
        input_dir = args.input_path
        output_dir = args.output_path
        all_files = os.listdir(input_dir)
        itr = 0
        while itr < len(all_files):
            if psutil.cpu_percent() < 80.00:
                print("Processing file: {}".format(all_files[itr])) 
                with open(os.path.join(input_dir, all_files[itr]), 'r') as f:
                    sample_data = json.load(f)
                    if sample_data is None:
                        print("Error: No data found for file: {}".format(all_files[itr]))
                        itr += 1
                        continue

                    sample_hash = sample_data["hash"]
                    ret = graph_reduction(sample_hash, sample_data, args.num_api_sequences, args.api_sequence_length, output_dir)
                    #proc = mp.Process(target=graph_reduction, args=(sample_hash, sample_data, args.num_api_sequences, args.api_sequence_length, output_dir))
                    #proc.start()
                    itr += 1

            else:
                print("Sleeping due to high CPU usage")
                time.sleep(5)

    print("\n\nCompleted API call extraction")
    sys.exit(0)

def parse_args():
    """Parse input arguments for mode, number of APIs, and sequences."""
    parser = argparse.ArgumentParser(description="GraphReduction & API Extraction")
    ir = parser.add_argument_group("parameters")
    ir.add_argument(
        '--api_sequence_length', default=30, type=str, help="Number of APIs to extract for each sequence"
    )
    ir.add_argument(
        '--num_api_sequences', default=300, type=str, help="Number of sequences to extract for each sample"
    )
    ir.add_argument(
        '--sample_hashes_path', default=300, type=str, help="Path to the keys of the saved db"
    )
    ir.add_argument('--use_mongo', action='store_true', help="Use MongoDB", default=False)

    ir.add_argument('--input_path', help="Path to the input file")

    ir.add_argument('--output_path', help="Path to the output file")

    args = parser.parse_args()

    # If not using Mongo DB, input path, and output path are required
    if not args.use_mongo:
        if not args.input_path or not args.output_path:
            parser.error("Input and output paths are required when not using MongoDB")

    return args

if __name__ == '__main__':
    args = parse_args()
    iterator()
