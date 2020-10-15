#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os,sys,inspect
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm
#from torch.nn.utils import cc
import torch.autograd as ag
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math 
import matplotlib.pyplot as plt
import numpy as np
import gc
from Training.tools_learning import get_right_indexes, criterion_pearson_no_reduction

def memReport(all = False):
    """
    :param all: wheter to detail all size obj
    :return: n objects
    In case of memory troubles call this function
    """
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()
def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out
def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up


class SpeakerEncoder(nn.Module):
    def __init__(self, c_in=512, c_h=128, c_out=128, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, n_dense_blocks=6, 
            subsample=[1,2,1,2,1,2], act='relu', dropout_rate=0):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])#先pad,再卷积
            y = self.act(y)#relu
            y = self.dropout_layer(y)#dropout
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)#池化
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)#relu
            y = self.dropout_layer(y)#dropout
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out

class ContentEncoder(nn.Module):
    def __init__(self, c_in=512, c_h=128, c_out=128, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, subsample=[1,2,1,2,1,2], 
            act='relu', dropout_rate=0):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, 
            c_in=128, c_cond=128, c_h=128, c_out=512, 
            kernel_size=5,
            n_conv_blocks=6, upsample=[2,1,2,1,2,1], act='relu', sn=False, dropout_rate=0):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
                [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, cond):
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            #y = pad_layer(y, self.second_conv_layers[l])
            #if self.upsample[l] > 1:
            #    y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            #y = self.norm_layer(y)
            #y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
            #y = self.act(y)
            #y = self.dropout_layer(y)
            #if self.upsample[l] > 1:
            #    out = y + upsample(out, scale_factor=self.upsample[l]) 
            #else:
            out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out

class ac2art_model(torch.nn.Module):
    """
    pytorch implementation of neural network
    """
    def __init__(self,  batch_size = 6,name_file="", sampling_rate=100,
            cuda_avail =True, filter_type=1, batch_norma=False):
        """
        :param hidden_dim: int, hidden dimension of lstm (usually 300)
        :param input_dim: int, input dimension of the acoustic features for 1 frame mfcc (usually 429)
        :param output_dim: int, # of trajectories to predict (usually 18)
        :param batch_size:  int, usually 10
        :param name_file: str, name of the model
        :param sampling_rate: int, sampling rate of the ema data for the smoothing (usually 100)
        :param cutoff: int, intial cutoff frequency for the smoothing, usually 10Hz
        :param cuda_avail: bool, whether gpu is available
        :param filter type: str, "out": filter outside the nn, "fix" : weights are FIXED,
        "unfix" : weights are updated during the training
        :param batch_norma: bool, whether to add batch normalization after the lstm layers
        """
        super(ac2art_model, self).__init__()
        self.speakerEncoder = SpeakerEncoder(c_in=39, c_h=128, c_out=128, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, n_dense_blocks=6, 
            subsample=[1,2,1,2,1,2], act='relu', dropout_rate=0)
        self.contentEncoder = ContentEncoder(c_in=39, c_h=128, c_out=128, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=1, subsample=[1,2,1,2,1,2], 
            act='relu', dropout_rate=0)
        self.decoder = Decoder(c_in=128, c_cond=128, c_h=128, c_out=39, 
            kernel_size=5,
            n_conv_blocks=6, upsample=[2,1,2,1,2,1], act='relu', sn=False, dropout_rate=0)


        self.sampling_rate = sampling_rate
        self.N = None
        self.min_valid_error = 100000
        self.all_training_loss = []
        self.all_validation_loss = []
        self.all_test_loss = []
        self.name_file = name_file


        self.cuda_avail = cuda_avail

        self.epoch_ref = 0
        self.batch_norma = batch_norma
        if cuda_avail :
            self.device = torch.device("cuda")
        else:
            self.device = None

    def prepare_batch(self, x, y):
        """
        :param x: list of B(batchsize) acoustic trajectories of variable lenghts,
        each element of the list is an array (K,18)  (K not always the same)
        :param y: list of B(batchsize) articulatory features,
        each element of the list is an array (K,429) (K not always the same)
        :return: 2 np array of sizes (B, K_max, 18) and (B, K_max, 429
        x,y initially data of the batch with different sizes . the script zeropad the acoustic and
        articulatory sequences so that all element in the batch have the same size
        """

        #max_length = np.max([len(phrase) for phrase in x])
        max_length = 300
        B = len(x)  # often batch size but not for validation
        new_x = torch.zeros((B, max_length, 39), dtype=torch.double)
        new_y = torch.zeros((B, max_length, 12), dtype=torch.double)
        for j in range(B):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
        x = new_x.view((B, max_length, 39))
        y = new_y.view((B, max_length, 12))
        return x, y

    def forward(self, x, filter_output=None):
        #print(x.shape)
        x = x.permute(0,2,1)
        emb = self.speakerEncoder(x)
        #print(emb.shape)
        mu, log_sigma = self.contentEncoder(x)
        #print(mu.shape)
        #print(log_sigma.shape)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        dec = dec.permute(0,2,1)
        #print(dec.shape)
        return mu, log_sigma, emb, dec

    

    def plot_results(self, y_target = None, y_pred_smoothed=None, y_pred_not_smoothed= None, to_cons=[]):
        """
        :param y: one TRUE arti trajectory
        :param y_pred_not_smoothed: one predicted arti trajectory not smoothed (forward with filtered=False)
        :param y_pred_smoothed:  one predicted arti trajectory (smoothed)
        :param to_cons:  articulations available to consider (list of 0/1)
        save the graph of each available trajectory predicted and true.
        If y_pred is given, also plot the non smoothed pred
        [future work : change filename and title of the graph]
        """
        print("you chose to plot")
        plt.figure()
        articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
         'ul_x', 'ul_y', 'll_x', 'll_y', 'la','lp','ttcl','tbcl','v_x', 'v_y']
        idx_to_cons = [k for k in range(len(to_cons)) if to_cons[k]]
        for j in idx_to_cons:
            plt.figure()

            plt.plot(y_target[:, j])
            plt.plot(y_pred_smoothed[:, j])
            if y_pred_not_smoothed is not None:
                plt.plot(y_pred_not_smoothed[:, j], alpha=0.6)
            plt.title("{0}_{1}.png".format(self.name_file, articulators[j]))
            if y_pred_not_smoothed is not None:
                plt.legend(["target", "pred smoothed", "pred not smoothed"])
            else:
                plt.legend(["target", "pred smoothed"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_{1}.png".format(self.name_file, articulators[j]))
            plt.savefig(save_pics_path)
            plt.close('all')

    def evaluate_on_test(self, X_test):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        all_diff = np.zeros((1, 39))
        all_pearson = np.zeros((1, 39))
        for i in range(len(X_test)):
            L = len(X_test[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, 39)  #x (1,L,429)
            #if self.cuda_avail:
            x_torch = x_torch.cuda().double()
            mu, log_sigma, emb, dec = self(x_torch)
            if self.cuda_avail:
                dec = dec.cpu().double()
            dec = dec.detach().numpy().reshape((L, 39))  # y_pred (L,13)
            x_torch=x_torch.cpu().detach().numpy().reshape((L,39))
            rmse = np.sqrt(np.mean(np.square(x_torch-dec), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, 39))

            all_diff = np.concatenate((all_diff, rmse))
            pearson = [0]*39
            for k in range(39):
                pearson[k] = np.corrcoef(x_torch[:, k].T, dec[:, k].T)[0, 1]
            pearson = np.array(pearson).reshape((1, 39))
            all_pearson = np.concatenate((all_pearson, pearson))
        all_pearson = all_pearson[1:]
        all_diff = all_diff[1:]
        all_pearson[np.isnan(all_pearson)] = 0

        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean != 0]))
        print("rmse mean per arti : \n", rmse_per_arti_mean)
        print("pearson final : ", np.mean(pearson_per_arti_mean[pearson_per_arti_mean != 0]))
        print("pearson mean per arti : \n", pearson_per_arti_mean)

        return rmse_per_arti_mean, pearson_per_arti_mean


