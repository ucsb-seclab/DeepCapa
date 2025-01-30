import argparse

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="deepcapa")
    ir = parser.add_argument_group("Dataset parameters")
    ir.add_argument("--input-dir", default='',  help="input_dir containing  API sequences in json format", required=True)
    ir.add_argument("--training-set-ratio", default=0.7, help="ratio of trainingset/validation set")
    ir.add_argument("--output-dir", default ='', help="output path to store model results", required=True)
    ir.add_argument("--unique-api-path", default='', help="path to unique APIs", required=True)
    #Trianing hyper-parameters
    hr = parser.add_argument_group("Training Hyper-parameters")
    hr.add_argument("--sequence-length",type=int, default=25,help="Length of sequnce")
    hr.add_argument("--sequence-budget",type=int, default=350,help="Total sequences per sample")
    hr.add_argument("--lr", type=float, default=0.00001)
    hr.add_argument("--variable-lr", type=bool, default=True, help="False if you want to have fixed learning rate")
    hr.add_argument('--batch-size', help="Training batch size", default=32, type=int)
    hr.add_argument('--epochs', help="Epochs, i.e., number of passes over the training set", type=int, default=1000)
    hr.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    hr.add_argument('--gpu', type=str, default="0", help="if device is cuda, determines which gpu to use")
    hr.add_argument('--mask-prob', type=float, default=0.20)
    hr.add_argument('--masking-scheme', type=int, default=1, choices=[1, 2, 3], help="1:Sequence Masking, 2: Random Masking, 3: Randomly choose between sequence and random masking")
    # Model Parameters
    mr = parser.add_argument_group("Parameters Related to Neural Network Architecture")
    mr.add_argument('--embedding-dim', help="Embedding dimension added after the input layer", type=int, default=768)
    
    mr.add_argument('--nheads', help="number of atention heads", type=int,
                    default=4)
    mr.add_argument('--nlayers', help="number of transformer layers", type=int,
                    default=4)
    mr.add_argument('--dropout', help="Dropout Probability", type=float, default=0.1) 
    mr.add_argument('--d_hid', type=int, help="The dimension of the feedforward network model for transformer encoder", default=2048)
    mr.add_argument('--load-weight', type=int, help="whether to load saved weight", default=0)
    mr.add_argument('--epoch-to-load', type=int, help="which epoch weight should be loaded", default=0)
    args = parser.parse_args()
    return args