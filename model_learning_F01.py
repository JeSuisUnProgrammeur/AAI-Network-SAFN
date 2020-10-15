#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os,sys,inspect
from torch.autograd import Variable
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gc
from Training.tools_learning import get_right_indexes, criterion_pearson_no_reduction
from Training.model_learning import ac2art_model
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

class Fusion(nn.Module):

    def __init__(self):
        super(Fusion, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(39,20),nn.LeakyReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(6,20),nn.LeakyReLU(True))
    def forward(self, x):
        a = self.fc1(x[:,:,0:39])
        b = self.fc2(x[:,:,39:45])
        return torch.cat((a,b),2)
class gen(nn.Module):

    def __init__(self, input_size=39, hidden_size=100, num_layer=3, output_size=12):
        super(gen, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.fusion = Fusion()
        self.lstm_net = nn.LSTM(40,100,num_layer, batch_first=True, bidirectional=True)
        self.layer2 = nn.Linear(200, output_size)

    def forward(self, x):
        feature_fusion = self.fusion(x)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(feature_fusion)
        return self.layer2(output)

class ext(nn.Module):

    def __init__(self, input_size=39, hidden_size=100, num_layer=3, output_size=12):
        super(ext, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.lstm_net = nn.LSTM(input_size,hidden_size,num_layer, batch_first=True, bidirectional=True)
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_size*2, output_size)
    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        return self.layer2(output)

class my_ac2art_model(torch.nn.Module):
    """
    pytorch implementation of neural network
    """
    def __init__(self, hidden_dim, input_dim, output_dim, batch_size,name_file="", sampling_rate=100,cutoff=10,cuda_avail =False, filter_type=1, batch_norma=False):
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
        super(my_ac2art_model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        #self.num_layer = num_layer
        self.extractor = ext(input_dim,hidden_dim,3,output_dim)
        self.generator = gen(input_dim+output_dim,hidden_dim,3,output_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(256,78),
            nn.ReLU(inplace=True)
        )
        #####

        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.filter_type=filter_type
        self.softmax = torch.nn.Softmax(dim=output_dim)
        self.tanh = torch.nn.Tanh()
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.N = None
        self.min_valid_error = 100000
        self.all_training_loss = []
        self.all_validation_loss = []
        self.all_test_loss = []
        self.name_file = name_file
        self.lowpass = None
        self.init_filter_layer()
        self.cuda_avail = cuda_avail

        self.epoch_ref = 0
        self.batch_norma = batch_norma
        #self.attention_layer = torch.nn.Sequential(
        #        torch.nn.Linear(self.hidden_dim*2,self.hidden_dim*2),
        #        torch.nn.ReLU(inplace=True)
        #        )
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
        new_x = torch.zeros((B, max_length, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((B, max_length, 12), dtype=torch.double)
        #new_phone = torch.zeros((B,max_length,61),dtype=torch.double)
        for j in range(B):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
            #new_phone[j] = zeropad(torch.from_numpy(phone[j])).double()
        x = new_x.view((B, max_length, 78))
        y = new_y.view((B, max_length, 12))
        #phone = new_phone.view((B,max_length,61))

        return x, y
    #def attention_net_w(self, lstm_out, lstm_hidden):
    #    lstm_tmp_out = torch.chunk(lstm_out,2,-1)
    #    h = lstm_out[0]+lstm_out[1]
    #    lstm_hidden = torch.sum(lstm_hidden,dim = 1)
    #    lstm_hidden = lstm_hidden.unsqueeze(1)
    #    atten_w = self.attention_layer(lstm_hidden)
    #    m = torch.nn.Tanh()(h)
    #    atten_context = torch.bmm(atten_w,m.transpose(1,2))
    #    softmax_w = torch.nn.Functional.softmax(atten_context,dim = -1)
    #    context = torch.bmm(softmax_w,h).squeeze(1)
    #    return context

    def forward(self, x, filter_output=None):
        """
        :param x: (Batchsize,K,429)  acoustic features corresponding to batch size
        :param filter_output: whether or not to pass throught the convolutional layer
        :return: the articulatory prediction (Batchsize, K,18) based on the current weights
        """
        if filter_output is None :
            filter_output = (self.filter_type != "out")
        #print(x.size())
        x = x.permute(0,2,1)
        #print("sa")
        #print(x.size())
        x = self.fc1(x)
        #print("as")
        #print(x.size())
        lip = self.extractor(x)
        #acoustic_lip = torch.cat((x,lip),2)
        #return self.generator(acoustic_lip)
        return lip


    def get_filter_weights(self):
        """
        :return: low pass filter weights based on calculus exclusively using tensors so pytorch compatible
        """
        cutoff = torch.tensor(self.cutoff, dtype=torch.float64,requires_grad=True).view(1, 1)
        fc = torch.div(cutoff,
              self.sampling_rate)  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if fc > 0.5:
            raise Exception("cutoff frequency must be at least twice sampling rate")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd .
        self.N = N

        n = torch.arange(N).double()
        alpha = torch.mul(fc, 2 * (n - (N - 1) / 2)).double()
        minim = torch.tensor(0.01, dtype=torch.float64) #utile ?
        alpha = torch.max(alpha,minim)#utile ?
        h = torch.div(torch.sin(alpha), alpha)
        beta = n * 2 * math.pi / (N - 1)
        w = 0.5 * (1 - torch.cos(beta))  # Compute hanning window.
        h = torch.mul(h, w)  # Multiply sinc filter with window.
        h = torch.div(h, torch.sum(h))
        return h

    def get_filter_weights_en_dur(self):
        """
        :return:  low pass filter weights using classical primitive (not on tensor)
        """
        fc = self.cutoff / self.sampling_rate
        if fc > 0.5:
            raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        self.N = N
        n = np.arange(N)
        h = np.sinc(fc * 2 * (n - (N - 1) / 2))
        w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1)))  # Compute hanning window.
        h = h * w
        h = h / np.sum(h)
        return torch.tensor(h)

    def init_filter_layer(self):
        """
        intialize the weights of the convolution in the NN,  so that it acts as a lowpass filter.
        the typefilter determines if the weights will be updated during the optim of the NN
        """


        # maybe the two functions do exactly the same...

        if self.filter_type in ["out","fix"] :
            weight_init = self.get_filter_weights_en_dur()
        elif self.filter_type == "unfix":
            weight_init = self.get_filter_weights()
        C_in = 1
        stride = 1
        must_be_5 = 5
        padding = int(0.5 * ((C_in - 1) * stride - C_in + must_be_5)) + 23
        weight_init = weight_init.view((1, 1, -1))
        lowpass = torch.nn.Conv1d(C_in, self.output_dim, self.N, stride=1, padding=padding, bias=False)

        if self.filter_type == "unfix":  # we let the weights move
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=True)

        else :  # "out" we don't care the filter won't be applied, or "fix" the wieghts are fixed
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=False)

        lowpass = lowpass.double()
        self.lowpass = lowpass

    def filter_layer(self, y):
        """
        :param y: (B,L,18) articulatory prediction not smoothed
        :return:  smoothed articulatory prediction
        apply the convolution to each articulation, maybe not the best solution (changing n_channel of the conv layer ?)
        """
        B = len(y)
        L = len(y[0])
        y = y.double()
        y_smoothed = torch.zeros(B, L, self.output_dim)
        for i in range(self.output_dim):
            traj_arti = y[:, :, i].view(B, 1, L)
            traj_arti_smoothed = self.lowpass(traj_arti)
            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

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

    def evaluate_on_test(self, X_test, Y_test, std_speaker, to_plot=False, to_consider=None, verbose=True, index_common = [], no_std = False):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        if to_plot:
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        model2 = ac2art_model()
        model2= model2.double().cuda()
        #file_weights = "saved_models/F01_spec_loss_90_filter_fix_bn_False_32.pt"
        file_weights = "saved_models/F03_indep_onfi_loss_90_filter_fix_bn_False_0.pt"
        loaded_state = torch.load(file_weights,map_location=self.device)
        model2.load_state_dict(loaded_state)
        for i in range(len(X_test)):
            L = len(X_test[i])
            #L = len(phone[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, 39)  #x (1,L,429)
            #phone_torch = torch.from_numpy(phone[i]).view(1,L,61)
            y = Y_test[i].reshape((L, 12))
            y = y[:,0:12]
            if index_common != []:
                y = get_right_indexes(y, index_common, shape = 2)
            if self.cuda_avail:
                x_torch = x_torch.to(device=self.device)
            mu,log_sigma,emb,dec = model2(x_torch)
            eps = log_sigma.new(*log_sigma.size()).normal_(0,1)
            x_torch = (mu+torch.exp(log_sigma/2)*eps)
            emb = emb.unsqueeze(2)
            new_input = torch.cat([emb for i in range(0,L)],2)
            input = torch.cat((x_torch,new_input),1)
                #phone_torch = phone_torch.to(device = self.device)
            #Input = torch.cat((x_torch,phone_torch),2)
            #y_pred_not_smoothed = self(x_torch, False).double() #output y_pred (1,L,13)
            y_pred_smoothed = self(input,False).double()
            if self.cuda_avail:
                #y_pred_not_smoothed = y_pred_not_smoothed.cpu()
                y_pred_smoothed = y_pred_smoothed.cpu()
            #y_pred_not_smoothed = y_pred_not_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            y_pred_smoothed = y_pred_smoothed.detach().numpy().reshape((L, 12))  # y_pred (L,13)
            #if to_plot:
            #    if i in indices_to_plot:
            #        self.plot_results(y_target = y, y_pred_smoothed = y_pred_smoothed,
            #                          y_pred_not_smoothed = y_pred_not_smoothed, to_cons = to_consider)
            rmse = np.sqrt(np.mean(np.square(y - y_pred_smoothed), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, 12))

            std_to_modify = std_speaker
            if index_common != [] and not no_std:
                std_to_modify = get_right_indexes(std_to_modify, index_common, shape=1)
            #by longxuan#
            std_to_modify = torch.from_numpy(std_to_modify)
            std_to_modify = std_to_modify[0:12]
            std_to_modify = std_to_modify.detach().numpy()
            #############
            rmse = rmse*std_to_modify  # unormalize
            all_diff = np.concatenate((all_diff, rmse))
            pearson = [0]*self.output_dim
            for k in range(self.output_dim):
                pearson[k] = np.corrcoef(y[:, k].T, y_pred_smoothed[:, k].T)[0, 1]
            pearson = np.array(pearson).reshape((1, self.output_dim))
            all_pearson = np.concatenate((all_pearson, pearson))
        all_pearson = all_pearson[1:]
        #if index_common == []:
        #    all_pearson[:, idx_to_ignore] = 0
        all_diff = all_diff[1:]
        #if index_common == []:
        #    all_diff[:, idx_to_ignore] = 0
        all_pearson[np.isnan(all_pearson)] = 0

        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        if verbose:
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean != 0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[pearson_per_arti_mean != 0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)

        return rmse_per_arti_mean, pearson_per_arti_mean

    def evaluate_on_test_modified(self,X_test, Y_test, std_speaker, to_plot=False, to_consider=None, verbose=True, index_common = [], no_std = False):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_diff_without_std = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        if to_plot:
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        for i in range(len(X_test)):
            L = len(X_test[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, self.input_dim)  #x (1,L,429)
            y = Y_test[i].reshape((L, 18))                     #y (L,13)
            if index_common != []:
                y = get_right_indexes(y, index_common, shape = 2)
            if self.cuda_avail:
                x_torch = x_torch.to(device=self.device)
            y_pred_not_smoothed = self(x_torch, False).double() #output y_pred (1,L,13)
            y_pred_smoothed = self(x_torch, True).double()
            if self.cuda_avail:
                y_pred_not_smoothed = y_pred_not_smoothed.cpu()
                y_pred_smoothed = y_pred_smoothed.cpu()
            y_pred_not_smoothed = y_pred_not_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            y_pred_smoothed = y_pred_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            if to_plot:
                if i in indices_to_plot:
                    self.plot_results(y_target = y, y_pred_smoothed = y_pred_smoothed,
                                      y_pred_not_smoothed = y_pred_not_smoothed, to_cons = to_consider)
            rmse = np.sqrt(np.mean(np.square(y - y_pred_smoothed), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, self.output_dim))

            std_to_modify = std_speaker
            if index_common != [] and not no_std:
                std_to_modify = get_right_indexes(std_to_modify, index_common, shape=1)

            rmse_with_std = rmse*std_to_modify  # unormalize

            all_diff = np.concatenate((all_diff, rmse_with_std))
            all_diff_without_std = np.concatenate((all_diff_without_std, rmse))
            y_pred_smoothed = y_pred_smoothed.reshape(1,L, self.output_dim)
            y_pred_smoothed = torch.from_numpy(y_pred_smoothed)
            y = torch.from_numpy(y.reshape(1, L, self.output_dim))
            pearson_per_art = criterion_pearson_no_reduction(y, y_pred_smoothed, cuda_avail=self.cuda_avail, device=self.device) # (1,1,18)
            pearson_per_art = pearson_per_art.reshape(1, self.output_dim)
            all_pearson = np.concatenate((all_pearson, pearson_per_art))
        all_pearson = all_pearson[1:]
        if index_common == []:
            all_pearson[:, idx_to_ignore] = 0
        all_diff = all_diff[1:]
        if index_common == []:
            all_diff[:, idx_to_ignore] = 0
        all_pearson[np.isnan(all_pearson)] = 0

        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        rmse_per_art_mean_without_std = np.mean(all_diff_without_std, axis=0)
        if verbose:
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean != 0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[pearson_per_arti_mean != 0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)

        return rmse_per_arti_mean, rmse_per_art_mean_without_std, pearson_per_arti_mean



