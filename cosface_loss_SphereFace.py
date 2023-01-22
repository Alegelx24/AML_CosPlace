

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

"""
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()

        self.s = 64.0 if not s else s
        self.m = 1.35 if not m else m

        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)





def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, s=None, m=1.5):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = True
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

   

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        #Cosine similarity between x and weights.
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine)
        #label.view => vettore colonna. 
        #Mette m solo dove c'è yi. Il resto rimane senza m. 
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        #Formulations from https://github.com/clcarwin/sphereface_pytorch
        cos_m_theta = self.mlambda[self.m](cosine)
        theta = Variable(cosine.acos())
        k = (self.m*theta/3.14159265).floor()
        n_one = k*0.0 - 1
        
        phi_theta = (n_one**k) * cos_m_theta - 2*k
        #x_norm = inputs.pow(2).sum(1).pow(0.5)
        #x_norm = torch.norm(inputs, 2, dim=1)
        #x_norm = x_norm.view(-1, 1)
        my_cosine_vector = one_hot * phi_theta + (1.0-one_hot) * cosine
        
        output = self.s * (my_cosine_vector)

        #output sul quale verrà applicata la cross entropy loss.
        return output


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss
"""
