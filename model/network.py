
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
import torch.nn.functional as F
import numpy as np
import GradientReversalLayer as GRL

from model.layers import Flatten, L2Norm, GeM

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,

    "mobilenet_v3_small": 576,#TESTED
    "efficientnet_b0": 1280,#TESTED
    "efficientnet_b3": 1536,#TESTED
    "efficientnet_v2_s": 1280,#TESTED
    "mobilenet_v3_small": 576,#TESTED
    "mobilenet_v3_large": 960,#TESTED
    "regnet_y_1_6gf":888,#TESTED
    "convnext_small" : 768,#TESTED
    "convnext_base": 1024,
    "regnet_y_16gf":888,#TESTED
    "swin_t": 768,#TESTED
    "swin_v2_t": 768,#TESTED
}


'''
def feature_L2_norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature.contiguous(), norm)


def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = feature_L2_norm(F.relu(correlation_tensor))
    return correlation_tensor

class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16, kernel_sizes=[7, 5], channels=[225, 128, 64], padding=0):
        super().__init__()
        assert len(kernel_sizes) == len(channels) - 1, \
            f"In HomographyRegression the number of kernel_sizes must be less than channels, but you said {kernel_sizes} and {channels}"
        nn_modules = []
        for in_channels, out_channels, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes):
            nn_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            nn_modules.append(nn.BatchNorm2d(out_channels))
            nn_modules.append(nn.ReLU())
        self.conv = nn.Sequential(*nn_modules)
        # Find out output size of last conv, aka the input of the fully connected
        shape = self.conv(torch.ones([2, 225, 15, 15])).shape
        output_dim_last_conv = shape[1] * shape[2] * shape[3]
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
        # Initialize the weights/bias with identity transformation
        init_points = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1]).type(torch.float)
        init_points = torch.cat((init_points, init_points))
        self.linear.bias.data = init_points
        self.linear.weight.data = torch.zeros_like((self.linear.weight.data))
    
    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x.reshape(B, 8, 2)
'''

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, grl_discriminator=None, homography_regression=None):
        """Return a model for GeoLocalization.        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
     
        self.backbone, features_dim = get_backbone(backbone)
        
        self.grl_discriminator=grl_discriminator
        # self.grl_discriminator=GRL.get_discriminator(features_dim, 1) #NEED TO PASS FEATURES DIM AS PARAMETER?

        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x, grl=False, operation=None, args=None):
        if grl:
            x = self.backbone(x)
            x= self.grl_discriminator(x)
        else:
            x = self.backbone(x)
            x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    #Efficienet backbones

    elif backbone_name.startswith("efficientnet_b0"):
        
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")

    elif backbone_name.startswith("efficientnet_b3"):
        
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")

    elif backbone_name.startswith("efficientnet_v2_s"):
        
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")

    #Mobilenet backbones

    elif backbone_name.startswith("mobilenet_v3_small"):

        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of MobileNet, freeze the previous ones")

    elif backbone_name.startswith("mobilenet_v3_large"):

        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of MobileNet, freeze the previous ones")

    #Regnet backbones

    elif backbone_name.startswith("regnet_y_1_6gf"):
        
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
        for layer in layers[:-1]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of RegNet, freeze the previous ones")
    
    elif backbone_name.startswith("regnet_y_16gf"):
        
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
        for layer in layers[:-1]: # freeze all the layers except the last one
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of RegNet, freeze the previous ones")

    #ConvNext backbones

    elif backbone_name.startswith("convnext_small"):
        
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")
    
    elif backbone_name == "convnext_base":
       for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
       logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
       layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer


    #Swin backbones

    elif backbone_name.startswith("swin_t"):

        layers = list(backbone.children())[:-3] # Remove avg pooling and FC layer
        for x in layers[0][:-1]:
            for p in x.parameters():
                p.requires_grad = False # freeze all the layers except the last three blocks
        logging.debug("Train last three layers of Swin, freeze the previous ones")

    elif backbone_name.startswith("swin_v2_t"):

        layers = list(backbone.children())[:-3] # Remove avg pooling and FC layer
        for x in layers[0][:-1]:
            for p in x.parameters():
                p.requires_grad = False # freeze all the layers except the last three blocks
        logging.debug("Train last three layers of Swin, freeze the previous ones")




    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim



