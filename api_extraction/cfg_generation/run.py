import json
import threading
import psutil
import time
import os
import shutil
import pika
import datetime
import argparse
from tqdm import tqdm
import sys

# Fetch environment variables for Pika credentials
pika_user = os.getenv("PIKA_USER")
pika_password = os.getenv("PIKA_PASSWORD")
pika_host = os.getenv("PIKA_HOST")
pika_port = os.getenv("PIKA_PORT")
routing_key = os.getenv("ROUTING_KEY") 
pika_queue = os.getenv("PIKA_QUEUE")

# Ensure all necessary environment variables are set
required_env_vars = [pika_user, pika_password, pika_host, pika_port, routing_key, pika_queue]
if not all(required_env_vars):
    print("Please set the environment variables for pika credentials")
    sys.exit(1)

class Producer:
    """Producer class to handle Pika message queue connections and publishing."""

    def __init__(self):
        credentials = pika.PlainCredentials(pika_user, pika_password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(pika_host, pika_port, '/', credentials)
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=pika_queue)

    def send_to_queue(self, message):
        """Send message to Pika queue."""
        self.channel.basic_publish(exchange='', routing_key=routing_key, body=message)

    def close_connection(self):
        """Close the connection to Pika."""
        self.connection.close()

def run_command(ida_location, interpreter_command, hash_path):
    """Run the IDA command for a given hash path."""
    command = f"{ida_location} -A {interpreter_command} {hash_path}"
    os.system(command)

def send_end_message():
    """Send end message to indicate completion."""
    message = {"method": "end", "data": {}}
    producer = Producer()
    producer.send_to_queue(json.dumps(message))
    producer.close_connection()

def clean_dump(dataset_path):
    """Clean up temporary files and directories."""
    # Remove specific files from dataset path
    to_remove = [f for f in os.listdir(dataset_path) if ".id" in f or ".nam" in f or ".til" in f]
    for f_name in to_remove:
        os.remove(os.path.join(dataset_path, f_name))
        print(f"Removed: {f_name}")

    # Clean specific temporary directories
    temp_folders = [
        f"/tmp/{i}" for i in os.listdir("/tmp/")
        if os.path.isdir(f"/tmp/{i}") and "tmp" in i and "tmp." not in i and "-" not in i and ".tmp" not in i
    ]
    for folder in temp_folders:
        try:
            shutil.rmtree(folder)
            print(f"Removed: {folder}")
        except OSError:
            continue

def fetch_snapshot_paths(dataset_path):
    """Return list of snapshot paths in the dataset directory."""
    return [os.path.join(dataset_path, sp) for sp in os.listdir(dataset_path)]

def meta_data_extractor(dataset_path, ida_location):
    """Extract metadata from memory snapshots using IDA."""
    dataset = fetch_dataset(dataset_path)
    lookup_dict = {
        sample_file_name.split(".")[0]: os.path.join(dataset_path, sample_file_name) for sample_file_name in dataset
    }
    all_hashes = list(lookup_dict.keys())
    print(f"Total number of hashes: {len(all_hashes)}")

    clean_dump(dataset_path)
    interpreter_command = ' -T"Lastline Process Snapshot v.3"'
    start_time = datetime.datetime.now()

    index = 0
    while index < len(all_hashes):
        if psutil.cpu_percent() < 80:
            if index % 100 == 0:
                print(f"Processing hash {index}")
            
            hash_path = lookup_dict.get(all_hashes[index])
            if not hash_path or not os.path.exists(hash_path):
                print(f"File not found: {hash_path}")
                index += 1
                continue

            # Start thread to run command
            threading.Thread(target=run_command, args=(ida_location, interpreter_command, hash_path)).start()
            index += 1

            # Cleanup and wait after processing every 2000 hashes
            if index % 2000 == 0:
                print("Sleeping for 2 minutes to clean up")
                time.sleep(120)
                clean_dump(dataset_path)
        else:
            print("CPU utilization high, sleeping for 10 seconds")
            time.sleep(10)

    # Final cleanup
    end_time = datetime.datetime.now()
    time.sleep(600)
    send_end_message()
    clean_dump(dataset_path)
    print(f"Total time taken: {end_time - start_time}")

def fetch_dataset(path):
    """Fetch all files in the given dataset path."""
    return os.listdir(path)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CFG extraction")
    parser.add_argument("--data", type=str, required=True, help="Path to the memory snapshot directory")
    parser.add_argument("--ida_root_dir", type=str, required=True, help="Root directory of IDA")
    parser.add_argument("--architecture", type=str, choices=["x86", "x64"], default="x86", help="Binary architecture")
    return parser.parse_args()

def main():
    """Main function to execute the CFG extraction process."""
    args = parse_arguments()
    dataset_path = args.data
    ida_location = args.ida_root_dir

    # Adjust IDA executable based on architecture
    if args.architecture == "x64":
        ida_location = os.path.join(ida_location, "idat64")
    elif args.architecture == "x86":
        ida_location = os.path.join(ida_location, "idat")
    else:
        raise ValueError("Unsupported architecture. Choose between x86 and x64.")

    meta_data_extractor(dataset_path, ida_location)

if __name__ == "__main__":
    main()
