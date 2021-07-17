#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:29:51 2019

@author: anikat
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# multivariate output stacked lstm example
import numpy as np
#from numpy import array
#from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate train sequence into samples
def split_train_sequences(sequences, n_steps, segLen):
	X, y = list(), list()
	for i in range(sequences.shape[0]):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > sequences.shape[0]-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_test_sequences(sequences, num_outputs, input_len):
    X=[]
    y_expected = []
    start_seq=max(0,sequences.shape[0]-num_outputs-1)
    end_seq=sequences.shape[0]
    for i in range(start_seq,end_seq):
        start_ix = max(0,i - input_len)
        seq_x, seq_y = sequences[start_ix:start_ix+input_len, :], sequences[i, :]
        X.append(seq_x)
        y_expected.append(seq_y)
    return np.array(X), np.array(y_expected)

def get_forecast(dataset_train,dataset_test,n_steps,num_outputs,segLen):
    #print(dataset_train.shape,dataset_test.shape)
    n_features = dataset_train.shape[1]
    x_train,y_train=split_train_sequences(dataset_train,n_steps,segLen)
    print('x_train:',x_train.shape)
    #print(x_train)
    print('y_train:',len(y_train))
    #print(y_train)
    #'''
    # define model
    print('starting model training..')
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(x_train, y_train, epochs=10, verbose=0)
    print('model train ends')
    # demonstrate prediction
    #'''
    x_test,y_expected = split_test_sequences(dataset_test, num_outputs, n_steps)
    print('x_test:',x_test.shape)
    print('y_test:',len(y_expected))
    #'''
    #x_input = x_input.reshape((num_outputs+1, n_steps, n_features))
    yhat = model.predict(x_test, verbose=0)
    error=np.abs(yhat-y_expected)
    forecast_error=np.mean(error,axis=0)
    print('predicted:')
    print(yhat)
    print('expected:')
    print(y_expected)
    print(forecast_error)
    print(np.argsort(-forecast_error))
    #'''
    return forecast_error,np.argsort(-forecast_error)
    
def main(data,segs,savefile):
    start=0
    timestamp=data.shape[0]
    timeseries=data.shape[1]
    forecast_error=[]
    forecast_ts=[]
    for i in range(len(segs)):
        s=segs[i]
        print('seg ',s)
        segLen=s-start
        end=timestamp
        if i<len(segs)-1:
            end=segs[i+1]
        n_steps=int(.05*segLen)
        num_output=int(.05*(end-s))
        print('n_steps:',n_steps,' num_outputs:',num_output)
        error,sorted_ts=get_forecast(data[start:s,:],data[start:s+num_output+1,:],n_steps,num_output, segLen)
        forecast_error.append(error)
        forecast_ts.append(sorted_ts)
        start=s
    forecast_error=np.transpose(forecast_error)
    forecast_ts=np.transpose(forecast_ts)
    np.savetxt(savefile+'_forecast_exp.csv',forecast_error,delimiter=',')
    np.savetxt(savefile+'_forecast_ei.csv',forecast_ts,fmt='%d',delimiter=',')

        
    
    
savefile='insect'
filename='../../data/sudden_cardiac_7000_ts2.csv'
segfile='../segment/sudden_cardiac_sampled7k_gt_seg.txt'

data=np.loadtxt(filename,delimiter=',',dtype=float)
#data=data[:,:4]
segs= np.loadtxt(segfile,delimiter=',',dtype=int)

main(data,segs,savefile)
    