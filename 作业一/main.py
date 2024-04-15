import urllib.request
import gzip
import argparse
import os
from full_connect_network import FullConnectNet
from optimization import Optim
from solver import Solver
from utils import *

def download_minist():
    urls = {
        "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
    }

    download = not os.path.exists('data')

    if download:
        os.makedirs('data')

    train_images = download_and_extract_data(urls["train_images"], os.path.join("data", "train-images-idx3-ubyte.gz"), download)
    train_labels = download_and_extract_data(urls["train_labels"], os.path.join("data", "train-labels-idx1-ubyte.gz"), download)
    test_images = download_and_extract_data(urls["test_images"], os.path.join("data", "t10k-images-idx3-ubyte.gz"), download)
    test_labels = download_and_extract_data(urls["test_labels"], os.path.join("data", "t10k-labels-idx1-ubyte.gz"), download)

    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    return data

def download_and_extract_data(url, file_name, download=True):
    """
    Download the date from the given url and load it as a numpy array with reshape.

    Input:
    - url: the url of the dataset
    - file_name: the storage path of the dataset
    - download: whether to download the dataset, defaule is True
    """
    if download:
        urllib.request.urlretrieve(url, file_name)

    with gzip.open(file_name, 'rb') as f:
        if 'images' in file_name:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            data = data.reshape((-1, 28, 28))
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a three layer neural network in the Fashion-MNIST dataset\n")
    # parameters of the three layer neural network
    parser.add_argument("--hidden_dims", type=int, nargs='+', required=True, help="The hidden layer sizes.")
    parser.add_argument("--activation", type=str, nargs='+', required=True, help="The type(s) of activation functions.")
    parser.add_argument("--reg", type=float, default=0.0, help="The regularization strength, default is 0.")
    parser.add_argument("--weight_scale", type=float, default=1e-2, help="The weight initialization scale, default is 0.01.")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="The loss function, default is cross-entrophy.")

    # parameters of the solver
    parser.add_argument("--epochs", type=int, default=10, help="The number of training epochs, default is 10.")
    parser.add_argument("--update_rule", type=str, default="sgd", help="The optimization method, default is sgd.")
    parser.add_argument("--optim_config", type=dict, default={}, help="The optimization configuration.")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="The learning rate decay, default is 1.0.")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size, default is 100.")

    args = parser.parse_args()

    data = download_minist()
    three_layer_model = FullConnectNet(
        hidden_dims=args.hidden_dims, 
        types=args.activation, 
        reg=args.reg, 
        weight_scale=args.weight_scale,
        loss=args.loss
    )
    three_layer_net = Solver(
        model=three_layer_model, 
        data=data,
        epochs=args.epochs,
        update_rule=Optim(args.update_rule), 
        optim_config=args.optim_config, 
        lr_decay=args.lr_decay, 
        batch_size=args.batch_size,
    )

    three_layer_net.train() 