import argparse
from full_connect_network import FullConnectNet
from optimization import get_optim_func
from solve import Solver
from utils import *


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
    parser.add_argument("--optim_config", type=Dict, default={}, help="The optimization configuration.")
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
        update_rule=get_optim_func(args.update_rule), 
        optim_config=args.optim_config, 
        lr_decay=args.lr_decay, 
        batch_size=args.batch_size,
    )

    three_layer_net.train() 