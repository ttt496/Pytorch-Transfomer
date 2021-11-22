import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.stats
from sklearn import preprocessing
from sklearn import metrics

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m , -1)

def attention(Q, K, V):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K) #(batch_size, dim_attn, seq_length)
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        x = self.fc(a)
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
        #self.fc2 = nn.Linear(5, dim_val)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn   
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):  
        x = self.fc1(x)
        #print(x.shape)
        #x = self.fc2(x) 
        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x     
    
class Takens:

    ### constant
    tau_max = 100
    ### initializer
    def __init__(self, data):
        self.data           = data
        self.tau, self.nmi  = self.__search_tau()

    ### reconstruct data by using searched tau
    def reconstruct(self):
        _data1 = self.data[:-2]
        _data2 = np.roll(self.data, -1 * self.tau)[:-2]
        _data3 = np.roll(self.data, -2 * self.tau)[:-2]
        return _data1, _data2, _data3

    ### find tau to use Takens' Embedding Theorem
    def __search_tau(self):
        # Create a discrete signal from the continunous dynamics
        hist, bin_edges = np.histogram(self.data, bins=200, density=True)
        bin_indices = np.digitize(self.data, bin_edges)
        data_discrete = self.data[bin_indices]

        # find usable time delay via mutual information
        before     = 1
        nmi        = []
        res        = None

        for tau in range(1, self.tau_max):
            unlagged = data_discrete[:-tau]
            lagged = np.roll(data_discrete, -tau)[:-tau]
            nmi.append(metrics.normalized_mutual_info_score(unlagged, lagged))
            if res is None and len(nmi) > 1 and nmi[-2] < nmi[-1]:
                res = tau - 1
        if res is None:
            res = 50
        return res, nmi

### Original method for 1D sigmoid toy model
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length    
    t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t   
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

# Custom method for dealing with arbitrary shape data. This method is used in Alter_Dataset (below).
def getdata_multiIO_infer_revise(X_data, Y_data , in_seq_len, out_seq_len):
    if  in_seq_len < out_seq_len:
      print(" out_seq_len should be longer than in_seq_len.")

    X_data, Y_data = np.array(X_data), np.array(Y_data)

    if X_data.ndim==1:
      whole_time_len = len(X_data)
    else:
      whole_time_len = len(X_data[0,:])  

    if X_data.ndim == 1:
      n_X_features = 1
    else:
      n_X_features = X_data.shape[0]

    if Y_data.ndim == 1:
      n_Y_features = 1
    else:
      n_Y_features = Y_data.shape[0]

    n_seq = int( (whole_time_len -  in_seq_len) / out_seq_len ) +1
    
    X, Y = np.zeros((n_seq, in_seq_len, n_X_features)), np.zeros((n_seq, out_seq_len, n_Y_features) )
    random.seed(0) #乱数のシード固定
    #sequnenceのスタートする時間indexを out_seq_lenずつずらす
    all_start_point_list = [i for i in range(0 , whole_time_len - in_seq_len +1,  out_seq_len)]
    random.shuffle(all_start_point_list)
    #print("all_strart_points",all_strart_points)
 
    i = 0
    if X_data.ndim == 1 and Y_data.ndim == 1:  
      for start_point in all_start_point_list :
        X[i] += np.transpose( [ X_data[ start_point : start_point + in_seq_len] ] )
        Y[i] += np.transpose( [ Y_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )

    elif X_data.ndim != 1 and Y_data.ndim==1:  
      for start_point in all_start_point_list :
        X[i] += np.transpose( X_data[ : , start_point : start_point + in_seq_len] )
        Y[i] += np.transpose( [ Y_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )
        i += 1

    elif X_data.ndim==1 and Y_data.ndim !=1: 
      for start_point in all_start_point_list :
        X[i] += np.transpose( [ X_data[ start_point : start_point + in_seq_len] ] )
        Y[i] += np.transpose( Y_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        i += 1

    else: 
      for start_point in all_start_point_list :
        X[i] += np.transpose( X_data[ : , start_point : start_point + in_seq_len] )
        Y[i] += np.transpose( Y_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        i += 1

    #return torch.tensor(X_train).float(), torch.tensor(Y_train).float(), torch.tensor(X_test).float(), torch.tensor(Y_test).float(), train_start_point_list, test_start_point_list
    return torch.tensor(X).float(), torch.tensor(Y).float()

def getdata_multiIO_infer_revise_w_Ymean(X_data, Y_data, Y_mean_data, in_seq_len, out_seq_len):
    if  in_seq_len < out_seq_len:
      print(" out_seq_len should be longer than in_seq_len.")

    X_data, Y_data, Y_mean_data = np.array(X_data), np.array(Y_data), np.array(Y_mean_data)

    if X_data.ndim==1:
      whole_time_len = len(X_data)
    else:
      whole_time_len = len(X_data[0,:])  

    if X_data.ndim == 1:
      n_X_features = 1
    else:
      n_X_features = X_data.shape[0]

    if Y_data.ndim == 1:
      n_Y_features = 1
    else:
      n_Y_features = Y_data.shape[0]

    n_seq = int( (whole_time_len -  in_seq_len) / out_seq_len ) +1
    
    X, Y, Y_mean = np.zeros((n_seq, in_seq_len, n_X_features)), np.zeros((n_seq, out_seq_len, n_Y_features)), np.zeros((n_seq, out_seq_len, n_Y_features))
    random.seed(0) #乱数のシード固定
    #sequnenceのスタートする時間indexを out_seq_lenずつずらす
    all_start_point_list = [i for i in range(0 , whole_time_len - in_seq_len +1,  out_seq_len)]
    random.shuffle(all_start_point_list)
    #print("all_strart_points",all_strart_points)
 
    i = 0
    if X_data.ndim == 1 and Y_data.ndim == 1:  
      for start_point in all_start_point_list :
        X[i] += np.transpose( [ X_data[ start_point : start_point + in_seq_len] ] )
        Y[i] += np.transpose( [ Y_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )
        Y_mean[i] += np.transpose( [ Y_mean_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )
        i += 1

    elif X_data.ndim != 1 and Y_data.ndim == 1:  
      for start_point in all_start_point_list :
        X[i] += np.transpose( X_data[ : , start_point : start_point + in_seq_len] )
        Y[i] += np.transpose( [ Y_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )
        Y_mean[i] += np.transpose( [ Y_mean_data[ start_point + in_seq_len - out_seq_len : start_point + in_seq_len] ] )
        i += 1

    elif X_data.ndim == 1 and Y_data.ndim != 1: 
      for start_point in all_start_point_list :
        X[i] += np.transpose( [ X_data[ start_point : start_point + in_seq_len] ] )
        Y[i] += np.transpose( Y_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        Y_mean[i] += np.transpose( Y_mean_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        i += 1

    else: 
      for start_point in all_start_point_list :
        X[i] += np.transpose( X_data[ : , start_point : start_point + in_seq_len] )
        Y[i] += np.transpose( Y_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        Y_mean[i] += np.transpose( Y_mean_data[ : , start_point + in_seq_len - out_seq_len : start_point + in_seq_len] )
        i += 1

    #return torch.tensor(X_train).float(), torch.tensor(Y_train).float(), torch.tensor(X_test).float(), torch.tensor(Y_test).float(), train_start_point_list, test_start_point_list
    return torch.tensor(X).float(), torch.tensor(Y).float() , torch.tensor(Y_mean).float()


###Dataset for sigle dimension output
class Alter_Dataset_1dimOut(Dataset):
    def __init__(self, path_list, axis_number_x_strat , axis_number_x_end, axis_number_y,  in_seq_len, out_seq_len, test_size, is_test=False):
        i = 0
        for path in path_list:
            data = np.transpose(np.loadtxt(path, skiprows=1, delimiter=','  ))
            x_data = data[ axis_number_x_strat : axis_number_x_end+1, :]
            y_data = data[ axis_number_y, :] 
            if i==0:
                x, y = getdata_multiIO_infer_revise(x_data, y_data , in_seq_len, out_seq_len)
            else:
                x_tmp, y_tmp = getdata_multiIO_infer_revise(x_data, y_data , in_seq_len, out_seq_len)
                x = torch.cat([x, x_tmp], dim=0)
                y = torch.cat([y, y_tmp], dim=0)
            i += 1
        test_number = int( test_size * np.shape(x)[0] ) 
        if is_test==True:
            self.X = x[:test_number]
            self.Y = y[:test_number]
        else:
            self.X = x[test_number:]
            self.Y = y[test_number:]
  
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        # x[index]は (sequence_len x piezo_num)
        # y[index]は (output_sequence_len x get_axis_num) もしくは、(get_axie_num)
        return self.X[index, : , : ], self.Y[index, : , :]  

###Dataset for multi-dimension output
class Alter_Dataset_multidimOut(Dataset):
    def __init__(self, path_list, axis_number_x_start , axis_number_x_end, axis_number_y_start, axis_number_y_end,  in_seq_len, out_seq_len, test_size, is_test=False):
        
        i = 0
        for path in path_list:
            data = np.transpose(np.loadtxt(path, skiprows=1, delimiter=','  ))
            x_data = data[ axis_number_x_start : axis_number_x_end , : ]
            y_data = data[ axis_number_y_start : axis_number_y_end , : ] 
            if i==0:
                x, y = getdata_multiIO_infer_revise(x_data, y_data , in_seq_len, out_seq_len)
            else:
                x_tmp, y_tmp = getdata_multiIO_infer_revise(x_data, y_data , in_seq_len, out_seq_len)
                x = torch.cat([x, x_tmp], dim=0)
                y = torch.cat([y, y_tmp], dim=0)
            i += 1
        test_number = int( test_size * np.shape(x)[0] ) 
        if is_test==True:
            self.X = x[:test_number]
            self.Y = y[:test_number]
        else:
            self.X = x[test_number:]
            self.Y = y[test_number:]
  
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        # x[index]は (sequence_len x piezo_num)
        # y[index]は (output_sequence_len x get_axis_num) もしくは、(get_axie_num)
        return self.X[index, : , : ], self.Y[index, : , :]  


class Alter_Dataset_multidimOut_from_dict(Dataset):
    def __init__(self, data_dict, axis_number_x_start, axis_number_x_end, axis_number_y_start, axis_number_y_end, in_seq_len, out_seq_len, test_size, is_test=False):
        ##data shape suppoused to be (n_dataset, n_features, time )
        i = 0
        for n in range(len(data_dict)):
            x_data = data_dict[n][ axis_number_x_start : axis_number_x_end , :]
            y_data = data_dict[n][ axis_number_y_start : axis_number_y_end , :] 
            time_len = len(x_data[0,:])
            y_mean_data = np.array( [ [np.mean(y_data[i, : ])] * time_len for i in range(len(y_data[:,0])) ] ) 
            print("n",n,"\n x_data",x_data,"\n y_data",y_data,"\n y_mean",y_mean_data)
            if i==0:
                x, y, y_mean = getdata_multiIO_infer_revise_w_Ymean(x_data, y_data , y_mean_data, in_seq_len, out_seq_len)
            else:
                x_tmp, y_tmp, y_mean_tmp = getdata_multiIO_infer_revise_w_Ymean(x_data, y_data, y_mean_data, in_seq_len, out_seq_len)
                x = torch.cat([x, x_tmp], dim=0)
                y = torch.cat([y, y_tmp], dim=0)
                y_mean = torch.cat([y_mean, y_mean_tmp], dim=0)
                
            i += 1
        test_number = int( test_size * np.shape(x)[0] ) 
        if is_test==True:
            self.X = x[:test_number]
            self.Y = y[:test_number]
            self.Y_mean = y_mean[:test_number]
        else:
            self.X = x[test_number:]
            self.Y = y[test_number:]
            self.Y_mean = y_mean[test_number:]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        # x[index]は (sequence_len x piezo_num)
        # y[index]は (output_sequence_len x get_axis_num) もしくは、(get_axie_num)
        return self.X[index, : , : ], self.Y[index, : , :], self.Y_mean[index, : , :]

def alter_preprocess(path):

    df0 = pd.read_csv(path, na_values=['None'])
    df = df0.bfill()

    a=8
    b=25
    c=29
    d=38
    e=42
    f=44
    g=51
    h=68
    i=72
    j=81
    k=85
    l=87
    m=135
    n=145
    w=1000
    z=1000

    pd.set_option('display.max_columns', 100)
    setaxisA = df.iloc[w:,a:b]
    setaxisB = df.iloc[w:,c:d]
    setaxisC = df.iloc[w:,e:f]
    setaxis1 = pd.concat([setaxisA, setaxisB], axis=1)
    setaxis = pd.concat([setaxis1, setaxisC], axis=1)

    getaxisA = df.iloc[w:,g:h]
    getaxisB = df.iloc[w:,i:j]
    getaxisC = df.iloc[w:,k:l]

    getaxis1 = pd.concat([getaxisA,getaxisB], axis = 1)
    getaxis = pd.concat([getaxis1,getaxisC], axis = 1)

    piezo = df.iloc[w:,l:m]
    ninevec = df.iloc[w:,m:]

    ss = preprocessing.StandardScaler()
    getaxis_f = getaxis.bfill()
    set_get = pd.concat([setaxis,getaxis_f], axis = 1)
    set_get_piezo = pd.concat([set_get,piezo], axis = 1)
    samples_scaler_pd = (set_get_piezo - set_get_piezo.mean()) / set_get_piezo.std(ddof=0)
    set_get_piezo_minmax = preprocessing.minmax_scale(set_get_piezo)

    return pd.DataFrame(set_get_piezo_minmax).to_numpy() 
 
