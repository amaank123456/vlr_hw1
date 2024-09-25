import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
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


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the parameters in the fully connected layer and others
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        self.resnet.layer1[1].conv1.weight.requires_grad = True
        self.resnet.layer4[0].bn2.bias.requires_grad = True
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        out = self.resnet(x)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 0.8 in 50 epochs
    ##################################################################
    args = ARGS(
        epochs=50, # maybe train for 10 epochs
        inp_size=224,
        use_cuda=True,
        val_every=50,
        lr=1e-3,
        batch_size=128,
        step_size=20, # maybe increase
        gamma=0.5,
        test_batch_size=256,
        log_every=5,
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
