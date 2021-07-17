# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:43:53 2019

@author: Anika
"""

import numpy as np
import sys
import os
import subprocess
import pandas as pd

def get_features(node_name, data):
    seg = node_name.split('-')
    begin_time = int(seg[0])
    end_time = int(seg[1])
    m= data.shape[1] #num_timeseries
    num_feature=4
    if begin_time==end_time:
        return np.zeros((m,num_feature))
    mean=[]
    mini=[]
    maxi=[]
    #median=[]
    var=[]
    for i in range(m):
        segment_data = data[begin_time:end_time, i]
        mean.append(np.mean(segment_data))
        #median.append(np.median(segment_data))
        mini.append(np.min(segment_data))
        maxi.append(np.max(segment_data))
        var.append(np.var(segment_data))
    features = np.column_stack((mean,mini,maxi,var))
    return features

def read_segment_data(segmentData, timestamp):
    PB =[]
    for i in range(origSeg.size):
        osz1Prev = 1
        if origSeg.size>1:
            osz1 = origSeg[i]+1
        else:
            if origSeg==-1:
                node_label= str(osz1Prev)+'-'+str(timestamp)
                PB.append(node_label)
                break  
            osz1=origSeg+1
        if i!=0:
            osz1Prev = origSeg[i-1]+1
        node_label= str(osz1Prev)+'-'+str(osz1) 
        PB.append(node_label)
        if i==(origSeg.size-1):
            node_label= str(osz1)+'-'+str(timestamp)
            PB .append(node_label)
    return PB 

def find_local_rationalization(K,num_timeseries,data,PB,save_dir,save_file):
    num_cp= len(PB)-1
    local_rat = np.zeros((num_timeseries,num_cp))
    Ei=[]
    selected_localK =[]
    for i in range(len(PB)-1):
        feature_node1= get_features(PB[i], data)
        feature_node2= get_features(PB[i+1], data)
        local_rat[:,i]=np.linalg.norm(np.subtract(feature_node1,feature_node2),axis=1)
    #local_rat_norm_ttl =local_rat
    local_rat_norm_ttl =local_rat/(np.max(local_rat)-np.min(local_rat))
    for i in range(len(PB)-1):    
        local_rat_norm = local_rat_norm_ttl[:,i]
        ei = np.argsort(-local_rat_norm)
        Ei.append(ei)
        ttl=0
        localK =[]
        for j in range(num_timeseries):
            localK.append(ei[j])
            ttl+=local_rat_norm[ei[j]]
            if ttl>=K:
                break
        selected_localK.append(localK)
        print('local seg %s:%s'%(PB[i],PB[i+1]))
        print(localK)
    selected_localK=np.array(selected_localK)
    np.savetxt(save_dir+save_file+'_expl.csv',local_rat,delimiter=',')
    #np.savetxt(save_dir+save_file+'_topKlocal.csv',selected_localK,fmt='%d',
     #          delimiter=',')
    return local_rat
    
def find_global_rationalization(thresh, num_timeseries, local_rat, save_dir, save_file):
    global_rat = np.zeros(num_timeseries)
    num_segs= local_rat.shape[1]
    for i in range(num_timeseries):
        for j in range(num_segs):
            global_rat[i]+= local_rat[i][j]
        global_rat[i]/=num_segs
    np.savetxt(save_dir+save_file+'_global_weight.csv',global_rat,delimiter=',')
    global_rat_norm = global_rat/(np.max(global_rat)-np.min(global_rat))
    global_timeseries = np.argsort(-global_rat)
    selected_globalK = []
    ttl=0
    for i in range(num_timeseries):
        selected_globalK.append(global_timeseries[i])
        ttl+=global_rat_norm[global_timeseries[i]]
        if ttl>=thresh:
            break
    print('globalK')
    print(selected_globalK)
    np.savetxt(save_dir+save_file+'_topKglobal.csv',selected_globalK,fmt='%d',delimiter=',')
    
def main(data, origSeg, K, save_dir,save_file):
    timestamp = data.shape[0]
    num_timeseries = data.shape[1]
    segments = read_segment_data(origSeg,timestamp)
    #print(segments)
    local_rat = find_local_rationalization(K,num_timeseries,data,segments,save_dir,save_file)
    #find_global_rationalization(K, num_timeseries, local_rat, save_dir, save_file)
    
if __name__ == '__main__':
    #datadir= '../data/Expl_synthetic.csv'
    #datadir= '../data/sudden_cardiac_sampled7k.csv'
    datadir= '../data/chicken21_original_normalized.csv'
    #datadir= sys.argv[1]
    #datadir= '../data/chicken21.csv'#sys.argv[1]
    #segFile = sys.argv[2]
    #segFile = '../segment/chicken21_gt'
    #segFile = '../segment/global_expl_gt'
    #segFile = '../segment/sudden_cardiac_sampled7k_gt'
    name='chicken21_original_gt'
    segFile = '../segment/'+name
    # Here K means the threshold or entropy of weight to choose
    K = 0.9#sys.argv[3] 
    save_file= name+'_topdiff'#sys.argv[4]#'cnr_syn_ticc'
    normalized = 'false'#sys.argv[5].lower()
    data = np.loadtxt(datadir,delimiter=',',dtype=float)
    if normalized=='true':
        data = (data-np.min(data))/(np.max(data)-np.min(data))
    save_dir= '../result_baseline/topdiff/'
    origSeg = np.loadtxt(segFile+'_seg'+'.txt',delimiter=',',dtype=int)
    #k is the threshold weight how many timeseries to choose
    main(data, origSeg, K, save_dir,save_file)

