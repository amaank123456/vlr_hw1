import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import torchvision
import torch.nn as nn
from voc_dataset import VOCDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

model = torch.load("./checkpoint-model-epoch50.pth")

test_loader = utils.get_data_loader('voc', train=False, batch_size=1000, split='test', inp_size=224)
data, target, wgt = next(iter(test_loader))
data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
output = model(data)
X_embedded = TSNE().fit_transform(output.detach().cpu().numpy())

class_colors_rgb = [    
    np.array([230, 25, 75]),   # Red    
    np.array([60, 180, 75]),   # Green    
    np.array([255, 225, 25]),  # Yellow    
    np.array([0, 130, 200]),   # Blue    
    np.array([245, 130, 48]),  # Orange    
    np.array([145, 30, 180]),  # Purple    
    np.array([70, 240, 240]),  # Cyan    
    np.array([240, 50, 230]),  # Magenta    
    np.array([210, 245, 60]),  # Lime    
    np.array([250, 190, 212]), # Pink    
    np.array([0, 128, 128]),   # Teal    
    np.array([220, 190, 255]), # Lavender    
    np.array([170, 110, 40]),  # Brown    
    np.array([255, 250, 200]), # Beige    
    np.array([128, 0, 0]),     # Maroon    
    np.array([170, 255, 195]), # Mint    
    np.array([128, 128, 0]),   # Olive    
    np.array([255, 215, 180]), # Coral    
    np.array([0, 0, 128]),     # Navy    
    np.array([128, 128, 128])  # Grey 
]

plt.figure(figsize=(10,6))
for i in range(target.shape[0]):
    idx = [x.item() for x in list(torch.where(target[i] == 1)[0])]
    if len(idx) > 1:
        class_sum = np.zeros(3)
        for j in idx:
            class_sum += class_colors_rgb[j]
        class_sum /= len(idx)
    else:
        class_sum = class_colors_rgb[idx[0]]
    
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c=tuple(class_sum/255))
plt.savefig('tsne_plot.png')