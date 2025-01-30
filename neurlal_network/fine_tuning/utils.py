import argparse

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser(description="deepcapa")
    ir = parser.add_argument_group("Dataset parameters")
    ir.add_argument("--input-dir", default='',  help="input_dir containing  API sequences in json format", required=True)
    ir.add_argument("--output-dir", default ='', help="output path to store model results", required=True)
    ir.add_argument("--unique-api-path", default='', help="path to unique APIs", required=True)
    
    #Trianing hyper-parameters
    hr = parser.add_argument_group("Training Hyper-parameters")
    hr.add_argument("--sequence-length",type=int, default=25,help="Length of sequnce")
    hr.add_argument("--sequence-budget",type=int, default=350,help="Total sequences per sample")
    hr.add_argument("--threshold",type=float, default=0.5, help="thresold used(only for stage2)")
    hr.add_argument("--lr", type=float, default=0.0001)
    hr.add_argument("--variable-lr", type=bool, default=True, help="False if you want to have fixed learning rate")
    hr.add_argument('--batch-size', help="Training batch size", default=32, type=int)
    hr.add_argument('--epochs', help="Epochs, i.e., number of passes over the training set", type=int, default=20)
    hr.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    hr.add_argument('--gpu', type=str, default="0", help="if device is cuda, determines which gpu to use")
    hr.add_argument('--tech-used', type=str, default='', help="tech to perform training", required=True)
    hr.add_argument('--load-weight', type=int, default=1, help="whether to load pre-trained weights, 1:yes, 0:no")
    hr.add_argument('--checkpoint-file-path', type=str, default="", help="path to checkpoint file (pretrained weights)")
    #hr.add_argument('--saved-weight-path', type=int, default=199, help="which epoch to load")
    # Model Parameters
    mr = parser.add_argument_group("Parameters Related to Neural Network Architecture")
    mr.add_argument('--embedding-dim', help="Embedding dimension added after the input layer", type=int, default=768)
    mr.add_argument('--d_hid', type=int, help="The dimension of the feedforward network model for transformer encoder", default=2048)
    mr.add_argument('--num-classes', type=int, default=2, help="number of classes in penultimate layer")
    mr.add_argument('--nheads', help="number of atention heads", type=int,
                    default=4)
    mr.add_argument('--nlayers', help="number of transformer layers", type=int,
                    default=4)
    mr.add_argument('--dropout-1', help="Dropout Probability for transformer", type=float, default=0.1) 
    mr.add_argument('--dropout-2', help="Dropout Probability before classifier", type=float, default=0.0) 
    mr.add_argument('--cnn-kernel-size', type=int, help="The dimention of kernel size", default=3)
    mr.add_argument('--cnn-kernel-stride', type=int, help="The dimention of kernel stride", default=3)
    mr.add_argument('--maxpool-kernel-size', type=int, help="Maxpool kernel size", default=3)
    mr.add_argument('--maxpool-kernel-stride', type=int, help="Maxpool kernel stride", default=3)
    mr.add_argument('--cnn-output-channels', type=int, default=8)
    args = parser.parse_args()
    return args