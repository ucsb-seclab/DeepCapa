# Neural Network

## Overview

DeepCapa uses neural network to map API call sequences to malicious capabilities. We train the DeepCapa in two stages: pretraining and fine-tuning.

## Pretraining

In the pretraining phase, we train our model using Masked Language Modeling objective. Here we randomly mask API calls within the API call sequences and the model is tasked to predict the actual value of the masked API calls. 

### Usage

To execute the pretraining script, use the following command:

```bash
python pretraining.py --input-dir="../api_extraction/sample_output/" \
                      --output-dir="./output/" \
                      --unique-api-path="./../api_extraction/api_sequences_extraction/unique_apis.txt"
```

- `--input-dir`: Directory containing input data for pretraining.
- `--output-dir`: Directory where output will be saved.
- `--unique-api-path`: Path to the file with unique API sequences.

## Fine-tuning
The fine-tuning phase builds upon the pretraining results. Here we train our model to detect malicious capabilities as binary classification task. 

### Usage

To run the fine-tuning script, use the following command:

```bash
python finetuning.py --input-dir="./dataset/fine_tuning/T1055/" \
                     --output-dir="./output/" \
                     --unique-api-path="./../api_extraction/api_sequences_extraction/unique_apis.txt" \
                     --checkpoint-file-path="./output/test_1/_saved_wt.pt" \
                     --tech-used="T1055" \
                     --epochs=2
```

- `--input-dir`: Directory containing the dataset for fine-tuning.
- `--output-dir`: Directory where output will be saved.
- `--unique-api-path`: Path to the file with unique API sequences.
- `--checkpoint-file-path`: Path to the pretraining checkpoint file.
- `--tech-used`: Specifies the technique used, e.g., "T1055".
- `--epochs`: Number of epochs for training.

