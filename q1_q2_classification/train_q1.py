import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import random

class ARGS():
    def __init__(self, 
            epochs=10, 
            inp_size=64, 
            use_cuda=True, 
            val_every=70, 
            lr=1e-3, 
            batch_size=32, 
            step_size=5, 
            gamma=0.1, 
            test_batch_size=32, 
            log_every=1,
            save_freq=5,
            save_at_end=True):
        self.epochs = epochs
        self.inp_size = inp_size
        self.use_cuda = use_cuda
        self.val_every = val_every
        self.lr = lr
        self.batch_size = batch_size
        self.step_size = step_size
        self.gamma = gamma
        self.test_batch_size = test_batch_size
        self.log_every = log_every
        self.save_freq = save_freq
        self.save_at_end = save_at_end

        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of
    # 224x224 for the rest of the questions.
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    ##################################################################
    args = ARGS(
        epochs=5, # maybe train for 10 epochs
        inp_size=64,
        use_cuda=True,
        val_every=20,
        lr=1e-3,
        batch_size=256,
        step_size=20, # maybe increase
        gamma=0.5,
        test_batch_size=256,
        log_every=5,
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    print(args)

    # initializes the model
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3)
    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
