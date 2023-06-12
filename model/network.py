
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

    "mobilenet_v3_small": 576,
    "efficientnet_b0": 1280,
    "efficientnet_b3": 1536,
    "efficientnet_v2_s": 1280,
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
    "regnet_y_1_6gf":888,
    "convnext_small" : 768,
    "convnext_base": 1024,
    "regnet_y_16gf":888,
    "swin_t": 768,
    "swin_v2_t": 768,
}


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
        for layer in layers[:-1]: # freeze all the layers except the last one
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layer of RegNet, freeze the previous ones")
    
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
        logging.debug("Train last two layers of Convnext, freeze the previous ones")
    

    #Swin backbones

    elif backbone_name.startswith("swin_t"):

        layers = list(backbone.children())[:-3] # Remove avg pooling and FC layer
        for x in layers[0][:-1]:
            for p in x.parameters():
                p.requires_grad = False 
        logging.debug("Train last three layers of Swin, freeze the previous ones")

    elif backbone_name.startswith("swin_v2_t"):

        layers = list(backbone.children())[:-3] # Remove avg pooling and FC layer
        for x in layers[0][:-1]:
            for p in x.parameters():
                p.requires_grad = False 
        logging.debug("Train Swin, freeze the previous ones")


    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim



