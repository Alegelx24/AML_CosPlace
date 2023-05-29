

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mlambda = [
                lambda x: x**0,
                lambda x: x**1,
                lambda x: 2*x**2-1,
                lambda x: 4*x**3-3*x,
                lambda x: 8*x**4-8*x**2+1,
                lambda x: 16*x**5-20*x**3+5*x
            ]
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        #Formulations from https://github.com/clcarwin/sphereface_pytorch
        cos_m_theta = self.mlambda[self.m](cosine)
        theta = Variable(cosine.acos())
        k = (self.m*theta/3.14159265).floor()
        n_one = k*0.0 - 1
        
        phi_theta = (n_one**k) * cos_m_theta - 2*k
 
        my_cosine_vector = one_hot * phi_theta + (1.0-one_hot) * cosine
        
        output = self.s * (my_cosine_vector)

        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
