import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils

import numpy as np


# class WeightModule(nn.Module):
#     def __init__(self, args, encoder, num_verb):

#         super(VerbClassification, self).__init__()
#         print("Build a VerbClassification Model")
#         self.num_verb = num_verb
        
#         self.base_network = encoder

# #         self.base_network = models.resnet50(pretrained = True)
# #         print('Load weights from Resnet18/50 done')

# #         if not args.finetune:
# #             for param in self.base_network.parameters():
# #                 param.requires_grad = False

#         output_size = self.num_verb
#         self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

#     def forward(self, image):
#         x = self.base_network.conv1(image)
#         x = self.base_network.bn1(x)
#         x = self.base_network.relu(x)
#         x = self.base_network.maxpool(x)

#         x = self.base_network.layer1(x)
#         x = self.base_network.layer2(x)
#         x = self.base_network.layer3(x)
#         x = self.base_network.layer4(x)

#         # avg pool or max pool
#         x = self.base_network.avgpool(x)
#         image_features = x.view(x.size(0), -1)

#         preds = self.finalLayer(image_features)

#         return preds
    
class Encoder(nn.Module):
    def __init__(self, model_name = 'resnet50'):

        super(Encoder, self).__init__()
        
        if model_name == 'resnet50':
            self.base_network = models.resnet50(pretrained = True)
            print('Load weights from Resnet18/50 done')


    def forward(self, image, y = None):
        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)

        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)

        # avg pool or max pool
        x = self.base_network.avgpool(x)
        image_features = x.view(x.size(0), -1)
        
        return image_features
    

class VerbClassification(nn.Module):
    def __init__(self, args, model_type, bn = True):

        super(VerbClassification, self).__init__()
        
        self.model_type = model_type
        if args.task_type == 'action' and args.dataset == 'imsitu':
            self.num_verb = args.num_verb
        elif args.task_type == 'gender':
            self.num_verb = 2
        elif args.dataset == 'celeba':
            self.num_verb = 2

        if self.model_type == 'classifier':
            print("Build a VerbClassification Model")
            self.weighing = False
            self.hidden_layers = args.hidden_layers
            
        elif self.model_type == 'weighing':
            print("Build a Weighing Model")
            self.weighing = True
            self.hidden_layers = args.hidden_layers_weight
            
        elif self.model_type == 'ARL':
            print("Build a Weighing Model")
            self.weighing = True
            self.hidden_layers = args.hidden_layers_weight
            
        else:
            print('wrong assignment to model')

        output_size = self.num_verb
        self.layers = []
        
        if not len(self.hidden_layers) == 0:
            out_shape = self.hidden_layers[0]

            if self.weighing:
                self.layers.append(nn.Linear(args.encoder.base_network.fc.in_features + output_size, out_shape))
            elif not self.weighing:
                self.layers.append(nn.Linear(args.encoder.base_network.fc.in_features, out_shape))

            if bn:
                self.layers.append(nn.BatchNorm1d(out_shape))
            self.layers.append(nn.ReLU())
#             self.layers.append(nn.Dropout())
        
            for i in range(len(self.hidden_layers)-1):
                in_shape = self.hidden_layers[i]
                out_shape = self.hidden_layers[i+1]

                self.layers.append(nn.Linear(in_shape, out_shape))
                self.layers.append(nn.BatchNorm1d(out_shape))
                self.layers.append(nn.ReLU())
#                 self.layers.append(nn.Dropout())
                
        else:
            if self.weighing:
                out_shape = args.encoder.base_network.fc.in_features + output_size
            else:
                out_shape = args.encoder.base_network.fc.in_features
                
        if self.model_type == 'ARL':
            self.layers.append(nn.Linear(out_shape, 1))
        else:
            self.layers.append(nn.Linear(out_shape, output_size))
            
        self.finalLayer = nn.Sequential(*self.layers)

    def forward(self, image_features, y = None):
        if self.weighing:
            image_features = torch.cat((image_features, y), -1)

        preds = self.finalLayer(image_features)
        
        if self.weighing:
            if self.model_type == 'weighing':
                preds = torch.softmax(preds, -1)
            elif self.model_type == 'ARL':
                preds = torch.sigmoid(preds)
                
        return preds

class Classifier(nn.Module):
    def __init__(self, encoder, last_layer):

        super(Classifier, self).__init__()
        self.encoder = encoder
        self.last_layer = last_layer

    def forward(self, image):
        features = self.encoder(image)
        preds = self.last_layer(features)

        return preds, features
    
# class VerbClassification(nn.Module):
#     def __init__(self, args, num_verb, model_type):

#         super(VerbClassification, self).__init__()
#         self.num_verb = num_verb
        
#         self.base_network = args.encoder

# #         self.base_network = models.resnet50(pretrained = True)
# #         print('Load weights from Resnet18/50 done')

# #         if not args.finetune:
# #             for param in self.base_network.parameters():
# #                 param.requires_grad = False

#         if model_type == 'classifier':
#             print("Build a VerbClassification Model")
#             self.weighing = False
            
#         elif model_type == 'weighing':
#             print("Build a Weighing Model")
#             self.weighing = True
            
# #             for param in self.base_network.parameters():
# #                 param.requires_grad = False
                
#         else:
#             print('wrong assignment to model')

#         output_size = self.num_verb
        
#         if self.weighing:
#             self.finalLayer = nn.Linear(self.base_network.fc.in_features + 1, output_size)
#         elif not self.weighing:
#             self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

#     def forward(self, image, y = None):
#         x = self.base_network.conv1(image)
#         x = self.base_network.bn1(x)
#         x = self.base_network.relu(x)
#         x = self.base_network.maxpool(x)

#         x = self.base_network.layer1(x)
#         x = self.base_network.layer2(x)
#         x = self.base_network.layer3(x)
#         x = self.base_network.layer4(x)

#         # avg pool or max pool
#         x = self.base_network.avgpool(x)
#         image_features = x.view(x.size(0), -1)
        
#         if self.weighing:
#             image_features = torch.cat((image_features, y), -1)

#         preds = self.finalLayer(image_features)
        
#         if self.weighing:
#             preds = 1 + torch.softmax(preds, -1)

#         return preds


class GenderClassifier(nn.Module):
    def __init__(self, args, num_verb):
        super(GenderClassifier, self).__init__()
        print('Build a GenderClassifier Model')

        hid_size = args.hid_size

        mlp = []
        mlp.append(nn.BatchNorm1d(num_verb))
        mlp.append(nn.Linear(num_verb, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input_rep):

        return self.mlp(input_rep)
