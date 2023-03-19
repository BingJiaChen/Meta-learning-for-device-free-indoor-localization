import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import torch


def single_minmaxscale(data, scale_range):
    def minmaxscale(data, scale_range):
        scaler = MinMaxScaler(scale_range)
        scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized

    X = []
    for i in data:
        X.append(minmaxscale(i.reshape(-1,1), scale_range))
    return np.array(X,dtype=object)     


def data_preproc(dataset, scale_range = (-1, 1)):
    data = dataset.item().get('data')
    label = dataset.item().get('label')
    data = single_minmaxscale(data, scale_range)
    
    data = data.astype('float32')
    data = data.reshape(-1,1,120)
    
    return data, label

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num

def count_acc(logits, label):
    pred = torch.argmax(F.softmax(logits), dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()





