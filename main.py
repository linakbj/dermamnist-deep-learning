import argparse
import numpy as np
import torch
import time
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes

def main(args):
    """
    The main function of the script.
    """
     
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    print(f"Training data shape: {xtrain.shape}")
    print(f"Test data shape: {xtest.shape}")

    # 2. Data preparation
    if not args.test:
        # Create validation set (80% train, 20% validation)
        perm = np.random.permutation(len(xtrain))
        xtrain, ytrain = xtrain[perm], ytrain[perm]
        validation_size = int(0.2 * len(xtrain))
        xval, yval = xtrain[:validation_size], ytrain[:validation_size]
        xtrain, ytrain = xtrain[validation_size:], ytrain[validation_size:]
        print(f"Training set size: {len(xtrain)}")
        print(f"Validation set size: {len(xval)}")

    # Normalize data
    means = np.mean(xtrain, axis=(0, 1, 2), keepdims=True)
    stds = np.std(xtrain, axis=(0, 1, 2), keepdims=True)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)
    if not args.test:
        xval = normalize_fn(xval, means, stds)

    # 3. Initialize model
    n_classes = get_n_classes(ytrain)
    
    if args.nn_type == "mlp":
        # Reshape data for MLP (flatten images)
        xtrain_flat = xtrain.reshape(xtrain.shape[0], -1)
        xtest_flat = xtest.reshape(xtest.shape[0], -1)
        if not args.test:
            xval_flat = xval.reshape(xval.shape[0], -1)
        
        input_size = xtrain_flat.shape[1]
        model = MLP(input_size=input_size, n_classes=n_classes).to(device)
        print("\nMLP Model Summary:")
        summary(model, input_size=(args.nn_batch_size, input_size))
        
        xtrain, xtest = xtrain_flat, xtest_flat
        if not args.test:
            xval = xval_flat
            
    elif args.nn_type == "cnn":
    
        if xtrain.shape[1] != 3: 
            xtrain = np.transpose(xtrain, (0, 3, 1, 2))
            xtest = np.transpose(xtest, (0, 3, 1, 2))
            if not args.test:
                xval = np.transpose(xval, (0, 3, 1, 2))
        
        model = model = CNN(input_channels=3, n_classes=n_classes, dropout_rate=args.dropout_rate).to(device)
        print("\nCNN Model Summary:")
        summary(model, input_size=(args.nn_batch_size, 3, 28, 28))
    else:
        raise ValueError(f"Unknown network type: {args.nn_type}")

    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    
    start_time = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    start_time = time.time()
    if args.test:
        preds = method_obj.predict(xtest)
        eval_time = time.time() - start_time
        print(f"Test set evaluation completed in {eval_time:.2f} seconds")
        
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"\nTest set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    else:
        preds = method_obj.predict(xval)
        eval_time = time.time() - start_time
        print(f"Validation set evaluation completed in {eval_time:.2f} seconds")
        
        acc = accuracy_fn(preds, yval)
        macrof1 = macrof1_fn(preds, yval)
        print(f"\nValidation set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"Training set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--test', action="store_true",
                      help="train on whole training data and evaluate on the test data")
    
    parser.add_argument('--nn_type', default="mlp", choices=['mlp', 'cnn'],
                      help="which network architecture to use")
    parser.add_argument('--nn_batch_size', type=int, default=32,
                      help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                      choices=['cpu', 'cuda', 'mps'],
                      help="Device to use for training")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="dropout rate for regularization")
    
    parser.add_argument('--lr', type=float, default=1e-4,
                      help="learning rate")
    parser.add_argument('--max_iters', type=int, default=30,
                      help="number of epochs")
    
    args = parser.parse_args()
    
    main(args)
