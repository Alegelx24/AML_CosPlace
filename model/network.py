
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
import torch.nn.functional as F
import numpy as np


from model.layers import Flatten, L2Norm, GeM

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "ConvNext_base": 1024,
    "ConvNext_tiny": 768,
    "efficientnet_v2_l": 1280,
    "mobilenet_v2" : 3,
    "squeezenet1_1": 3,
    "efficientnet_b0": 1280,
    "resnext50_32x4d": 2048,
    "maxvit_t" : 64,
    "vit_b_32":768,
    "resnext50_32x4d" : 2048,
    "swin_t": 768,
    "SWIN_V2_B": 1024,
    "shufflenet_v2_x2_0": 976,
    "regnet_y_1_6gf": 888
}



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

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, grl_discriminator=None, homography_regression=None, attention=None):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone)

        self.grl_discriminator=grl_discriminator #NEED TO PASS FEATURES DIM AS PARAMETER?

        self.homography_regression=homography_regression
        
        self.attention = attention

        self.weight_softmax = nn.Linear(512 , 1000).weight



        
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x, grl=False, operation=None, args=None):

        if operation == "similarity":
            tensor_img_1, tensor_img_2 = args
            return self.similarity(tensor_img_1, tensor_img_2)
        
        elif operation == "regression":
            similarity_matrix = args
            return self.regression(similarity_matrix)
        
        elif operation == "similarity_and_regression":
            tensor_img_1, tensor_img_2 = args
            similarity_matrix_1to2, similarity_matrix_2to1 = self.similarity(tensor_img_1, tensor_img_2)
            return self.regression(similarity_matrix_1to2), self.regression(similarity_matrix_2to1)            

        elif operation==None:
            if grl:
                x = self.backbone(x)
                x= self.grl_discriminator(x)

            else:
                if self.attention:
                    fc_out, feature_conv, feature_convNBN = self.backbone(input)
                    bz, nc, h, w = feature_conv.size()
                    feature_conv_view = feature_conv.view(bz, nc, h * w)
                    probs, idxs = fc_out.sort(1, True)
                    class_idx = idxs[:, 0]
                    scores = self.weight_softmax[class_idx].to(input.device)
                    cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
                    attention_map = F.softmax(cam.squeeze(1), dim=1)
                    attention_map = attention_map.view(attention_map.size(0), 1, h, w)
                    attention_features = feature_convNBN * attention_map.expand_as(feature_conv)
                    
                    x = attention_features
                    
                else:
                    x = self.backbone(x)
                    x = self.aggregation(x)
            
            return x
        
   
    def similarity(self, tensor_img_1, tensor_img_2):
        features_1 = self.features_extractor(tensor_img_1.cuda())
        features_2 = self.features_extractor(tensor_img_2.cuda())
        similarity_matrix_1to2 = compute_similarity(features_1, features_2)
        similarity_matrix_2to1 = compute_similarity(features_2, features_1)
        return similarity_matrix_1to2, similarity_matrix_2to1
    
    def regression(self, similarity_matrix):
        return self.homography_regression(similarity_matrix)


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
   
    elif backbone_name == "ConvNext_base":
       for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
       logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
       layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "ConvNext_tiny":
       for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
       logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
       layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer


    elif backbone_name == "efficientnet_v2_l":
        for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "mobilenet_v2":
        for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    
    elif backbone_name == "squeezenet1_1":
        for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "efficientnet_b0":
        for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "resnext50_32x4d":
        for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "maxvit_t":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vit_b_32":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "resnext50_32x4d":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "swin_t":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  

    elif backbone_name == "SWIN_V2_B":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  
    
    elif backbone_name == "shufflenet_v2_x2_0":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  

    elif backbone_name == "regnet_y_1_6gf":
        for name, child in backbone.named_children():
               
                for params in child.parameters():
                    params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  

    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim



