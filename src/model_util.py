import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_shape, hidden_layers, classes, adv = False):
        super().__init__()
        
        self.adv = adv
        
        self.layers = []
        self.layers.append(nn.Linear(input_shape, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden_layers[-1], classes))
        
        if self.adv:
            self.layers.append(nn.LeakyReLU())
        
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x):
        
        latent = self.mlp(x)
        
#         if self.adv:
#             latent = torch.sigmoid(latent) + 1
        
        return latent
        
        
def MC_criterion(output, target, weight, soft = 0.1):
    #multi-class classification
    
    target = ((target == 0) * soft) / (target.shape[-1] - 1 ) + (target == 1) * (1 - soft)
    
    weighted_output = - (weight * F.log_softmax(output, -1) * abs(target)).sum(dim = -1)
    loss = torch.mean(weighted_output)
    
    return loss, weighted_output

def ML_criterion(output, target, weight):
    
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss = torch.mean(torch.sum(criterion(output, target) * weight, -1))
    
    return loss

def normalize_weight(weight):
    weight = torch.sigmoid(weight)
    
    weight = 1 + weight/(torch.mean(torch.sigmoid(weight),0))
    
    return weight