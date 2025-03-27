# DEEPCAPA

DEEPCAPA is an advanced framework for automatic malware post-detection, designed to identify malicious capabilities in Windows malware by mapping them to MITRE ATT&CK Techniques. Developed by [Saastha Vasan](https://saasthavasan.github.io/), DEEPCAPA provides the following key functionalities:

- **Loading of Process Memory Snapshots**: Efficiently loads memory snapshots into a disassembler (IDA) and extracts control flow graphs (CFGs).
- **API Call Sequence Extraction**: Processes extracted CFGs from different snapshots, generates a unified CFG, and simulates program execution via random walks to extract API call sequences.
- **Neural Network Pipeline**: Utilizes a neural network pipeline to process the extracted API call sequences and map them to potentially malicious MITRE ATT&CK techniques.

## Quick Start

### 1. Download DEEPCAPA

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/ucsb-seclab/DeepCapa && cd DeepCapa
```

### 2. Create Conda Environment and Install Dependencies

Set up a dedicated environment for DEEPCAPA with the required dependencies:

```bash
conda create -n deepcapa python=3.10 ipython
pip install -r requirements.txt
```

### 3. Install PyTorch for Neural Network Model Execution

To run the neural network model, you need to install PyTorch. Follow the [official instructions](https://pytorch.org/) or use the provided command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### API Extraction

The `api_extraction` directory includes scripts for:
- Loading memory snapshots into IDA.
- Extracting API call sequences from CFGs of multiple process memory snapshots.

### Neural Network

The `neural_network` directory houses scripts necessary for:
- Pretraining DEEPCAPA models.
- Fine-tuning the models to improve detection capabilities.

## Citation

If you use DEEPCAPA in your research or projects, please cite it as follows:

```bibtex
@INPROCEEDINGS{10917772,
  author={Vasan, Saastha and Aghakhani, Hojjat and Ortolani, Stefano and Vasilenko, Roman and Grishchenko, Ilya and Kruegel, Christopher and Vigna, Giovanni},
  booktitle={2024 Annual Computer Security Applications Conference (ACSAC)}, 
  title={DEEPCAPA: Identifying Malicious Capabilities in Windows Malware}, 
  year={2024},
  volume={},
  number={},
  pages={826-842},
  keywords={Reverse engineering;Switches;Manuals;Machine learning;Artificial neural networks;Probabilistic logic;Malware;Computer security;malware analysis;deep learning},
  doi={10.1109/ACSAC63791.2024.00072}}

```
