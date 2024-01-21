import torch
import torch.nn as nn
import numpy as np
import pdb
import torch.optim as optim
import tqdm
import time
import matplotlib.pyplot as plt
import scipy.stats as ss
from copy import deepcopy
import csv
import sys,os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dtype = torch.float32

class mlp(torch.nn.Module):

    def __init__(self, input_size,h_dim, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = torch.nn.Linear(self.input_size, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(h_dim, h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(h_dim, h_dim)
        self.bn3 = nn.BatchNorm1d(h_dim)
        self.activation3 = torch.nn.ReLU()
        #self.linear4 = torch.nn.Linear(1024, 512)
        #self.activation4 = torch.nn.ReLU()


        self.output_layer = torch.nn.Linear(h_dim, self.output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        #torch.nn.init.xavier_uniform_(self.linear4.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.bn1(self.linear1(inputs)))
        x = self.activation2(self.bn2(self.linear2(x)))
        x = self.activation3(self.bn3(self.linear3(x)))
        #x = self.activation4(self.linear4(x))
        x = self.output_layer(x)
        #x = torch.nn.functional.normalize(x,dim=1)
        return x

class Set_Encoder(nn.Module):
    def __init__(self,r_dim,h1_dim,h2_dim,z_dim):
        super(Set_Encoder, self).__init__()

        self.r_dim = r_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z_dim = z_dim
        self.x_to_r = mlp(2,h1_dim,r_dim)
        self.r_to_z = mlp(r_dim,h2_dim,z_dim)
    def forward(self,inputs):
        #pdb.set_trace()
        batch_size = inputs.shape[0]
        x_transpose = torch.transpose(inputs.reshape(batch_size,2,1000),1,2)
        batch_size, num_kmers, _ = x_transpose.size()
        x_flat = x_transpose.reshape(batch_size * num_kmers, 2)
        r_i_flat = self.x_to_r(x_flat)
        r_i = r_i_flat.reshape(batch_size, num_kmers, self.r_dim)
        r_mean = torch.mean(r_i, dim=1)
        z = self.r_to_z(r_mean)
        z = torch.nn.functional.normalize(z,dim=1)
        return z


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats

def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = torch.nn.Linear(input_size, 3072)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3072, 1024)
        self.activation2 = torch.nn.ReLU()
        #self.linear3 = torch.nn.Linear(1024, 512)
        #self.activation3 = torch.nn.ReLU()
        #self.linear4 = torch.nn.Linear(1024, 512)
        #self.activation4 = torch.nn.ReLU()


        self.output_layer = torch.nn.Linear(1024, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        #torch.nn.init.xavier_uniform_(self.linear3.weight)
        #torch.nn.init.xavier_uniform_(self.linear4.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        #x = self.activation3(self.linear3(x))
        #x = self.activation4(self.linear4(x))
        x = self.output_layer(x)
        x = torch.nn.functional.normalize(x,dim=1)
        return x

class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.activation4 = torch.nn.ReLU()
        #self.linear5 = torch.nn.Linear(hidden_size, hidden_size)
        #self.bn5 = nn.BatchNorm1d(hidden_size)
        #self.activation5 = torch.nn.ReLU()
        

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        #torch.nn.init.xavier_uniform_(self.linear5.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
    def forward(self,inputs):
        x = self.activation1(self.bn1(self.linear1(inputs)))
        x = self.activation2(self.bn2(self.linear2(x)))
        x = self.activation3(self.bn3(self.linear3(x)))
        x = self.activation4(self.bn4(self.linear4(x)))
        #x = self.activation5(self.bn5(self.linear5(x)))
        x = self.output_layer(x)
        x = torch.nn.functional.normalize(x,dim=1)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self,temperature=0.07,mode="out",lmda=1,beta=0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.lmda = lmda
        self.beta = beta
    def forward(self,features,sim_mat):
        #pdb.set_trace()
        assert len(features.shape) == 2
        assert features.shape[0] == sim_mat.shape[0]
        batch_size = features.shape[0]
        diagonal_mask = (torch.ones(batch_size,batch_size) - torch.eye(batch_size)).to(dtype=dtype,device=device)
        relu_sim_mat = torch.nn.functional.relu(sim_mat)
        mask = relu_sim_mat.clone().detach()
        mask[mask>0] = 1.0
        abs_sim_mat = torch.abs(sim_mat)


        feature_dot_contrast=torch.div(torch.matmul(features, features.T),self.temperature)

        if(self.mode=="out"):
            #pdb.set_trace()
            relu_logits = relu_sim_mat * feature_dot_contrast

            #abs_logits = (abs_sim_mat+self.beta) * feature_dot_contrast
            f_Threshold = nn.Threshold(self.beta,self.beta)
            abs_logits = f_Threshold(abs_sim_mat) * feature_dot_contrast
            #abs_logits = feature_dot_contrast
            #for numerical stability
            abs_logits_max,_ = torch.max(abs_logits,dim=1,keepdim=True)
            abs_logits_2 = abs_logits - abs_logits_max.detach()
            denominator_part = torch.exp(abs_logits_2) * diagonal_mask
            denominator_part = self.lmda * (torch.log(denominator_part.sum(1, keepdim=True)) + abs_logits_max.detach())
            #log_prob = relu_logits - denominator_part

            mean_log_prob = (mask*relu_logits).sum(1)/(mask.sum(1)+1e-7) - denominator_part

        if(self.mode=="in"):
            #pdb.set_trace()
            logits_max, _ = torch.max(feature_dot_contrast,dim=1,keepdim=True)
            logits = feature_dot_contrast - logits_max.detach()
            exp_logits = torch.exp(logits)
            relu_logits = relu_sim_mat * exp_logits
            abs_logits = abs_sim_mat * exp_logits
            mean_log_prob = torch.log(relu_logits.sum(1,keepdim=True)+1e-7) - torch.log(abs_logits.sum(1,keepdim=True)+1e-7)
        
        if(self.mode=="naive"):
            logits = sim_mat * feature_dot_contrast
            mean_log_prob = (diagonal_mask*logits).sum(1)/(diagonal_mask.sum(1))
        
        if(self.mode=="l2"):
            #pdb.set_trace()
            #l2_mat = (self.temperature*feature_dot_contrast*diagonal_mask-sim_mat*diagonal_mask)**2
            l2_mat = (self.temperature*feature_dot_contrast-sim_mat)**2
            loss = l2_mat.mean()
            return loss
        
        loss = -mean_log_prob.mean()

        return loss

def save_model(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state





