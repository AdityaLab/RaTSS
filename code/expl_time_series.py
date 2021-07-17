# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 00:40:42 2019

@author: Anika
"""

import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import reduce
import sys
import os
import subprocess
import scipy
import time
import sharedmem as shm
import pickle
import math
from decimal import Decimal
from decimal import *

global data
global num_timeseries
global timestamp
global EPS
	
def read_local_exp(PB,alpha):
    CP=[]
    Ei=[]
    Feat=[]
    print(np.argsort(-alpha))
    for i in range(1,len(PB)-1):
        if PB[i]=='s' or PB[i+1]=='t':
            continue
        node1=PB[i]
        t=str.split(node1,'-')
        f1 = get_features(int(t[0]),int(t[1]))
        node2=PB[i+1]
        t=str.split(node2,'-')
        #print(t)		
        f2 = get_features(int(t[0]),int(t[1]))
        cp_weight= find_weight_matrix(f1,f2)
        feat_c=np.zeros(cp_weight.shape[0])
        for j in range(cp_weight.shape[0]):
            feat_c[j]=cp_weight[j]
            cp_weight[j]= np.abs(alpha[j]*cp_weight[j])
        ei=np.argsort(-cp_weight)
        #print(node1+':'+node2)
        #print(cp_weight)
        #print(ei)
        CP.append(cp_weight)
        Ei.append(ei)
        Feat.append(feat_c)
    CP = np.transpose(CP)
    Ei=np.transpose(Ei)
    Feat=np.transpose(Feat)
    #print(CP)
    np.savetxt(save_dir+save_file+'_exp.csv',CP,delimiter=',')
    np.savetxt(save_dir+save_file+'_ei.csv',Ei,fmt='%d',delimiter=',')
    np.savetxt(save_dir+save_file+'_F.csv',Feat,delimiter=',')

def is_edge_in_pb(PB, node, succ):
    if (node in PB) and (succ in PB):
        return 1
    else:
        return 0

def get_features(begin_time, end_time):
    m=num_timeseries#data.shape[1] #
    num_feature=3#5
    if begin_time==end_time:
        return np.zeros((m,num_feature))
    mean=[]
    min=[]
    max=[]
    var=[]
    #time_peaks=[]
    for i in range(m):
        segment_data=data[begin_time:end_time,i]
        mean.append(np.mean(segment_data))
        min.append(np.min(segment_data))
        max.append(np.max(segment_data))
        var.append(np.var(segment_data))
    if num_feature==1:
        features = np.column_stack((min,max,var)) #remove mean
    elif num_feature==2:
        features = np.column_stack((mean,min,max)) #remove variance
    elif num_feature==3:
        features = np.column_stack((mean,max,var)) #remove min
    elif num_feature==5:
        features = np.column_stack((mean,min,var)) #remove max
    else:
        features = np.column_stack((mean,min,max,var)) #all
    #features = np.column_stack((mean,min,max,var))
    #print(features)
    return features

def find_total_path_with_edge(s1, e1, e2, beginTime, endTime):
    if s1==beginTime and e2==endTime:
        return 1.0
    elif s1==beginTime:
        return 2**(endTime-e2-1)
    elif e2==endTime:
        return 2**(s1-beginTime-1)
    else:
        return 2**((s1-beginTime-1)+(endTime-e2-1))

def find_weight_value(f1, f2, idx):
    return np.sum(np.square(np.subtract(f1[idx,:],f2[idx,:])))
	
def find_bound(segLen,diff_l,diff_h):
    if segLen<diff_l:
        diff_l=segLen
    if segLen>diff_h:
        diff_h=segLen
    return diff_l,diff_h

def read_segment_data(segmentData):
    PB=['s']
    diff_l=timestamp
    diff_h=0
    for i in range(origSeg.size):
        osz1Prev = 1
        if origSeg.size>1:
            osz1=origSeg[i]+1
        else:
            if origSeg==-1:
                node_label= str(osz1Prev)+'-'+str(timestamp)
                PB.append(node_label)
                diff_l,diff_h=find_bound(timestamp-osz1Prev,diff_l,diff_h)
                break
            osz1=origSeg+1
        if i!=0:
            osz1Prev = origSeg[i-1]+1
        node_label= str(osz1Prev)+'-'+str(osz1)		
        PB .append(node_label)
        diff_l,diff_h=find_bound(osz1-osz1Prev,diff_l,diff_h)
        if i==(origSeg.size-1):
            node_label= str(osz1)+'-'+str(timestamp)
            diff_l,diff_h=find_bound(timestamp-osz1,diff_l,diff_h)
            PB .append(node_label)
    PB.append('t')
    #print('The black-box path')
    #print(PB)
    print(diff_l,diff_h)
    return PB,diff_l,diff_h 

def is_in_bound(node_start,node_end,seglen,diff_l,diff_h):
    if seglen<=diff_l or seglen>=diff_h:
        return False
    return True
	
def get_gloal_exp(PB,ttl_path,graph_nodes,B2,m,beginTime, endTime, matlab_path, save_dir,save_file):
    B1 = np.zeros(m) #saves total cost for blackbox path
    for i in range(len(PB)-1):
        node1 = PB[i]
        node2 = PB[i+1]
        endSeg1= int(str.split(node1,'-')[1])
    np.savetxt(save_dir+save_file+'_B.txt',allB,delimiter=',')
    max_it=3000
    learning_rate=0.7
    matlab_function = "grad_desc('"+ save_dir+ "', '"+save_file+"', '"+str(max_it) + "', '"+str(learning_rate)+"')"
    subprocess.call([matlab_path, "-nosplash", "-nodisplay", "-r", matlab_function])
    filename= save_dir+save_file+'_alpha.txt'
    alpha = np.loadtxt(filename,delimiter=',')
    return alpha,B1

def find_weight_matrix(f1,f2):
    return np.linalg.norm(np.subtract(f1,f2),axis=1)

def write_output_file(save_dir,save_file,PB,B2,ttl_path):
    #getcontext().prec = 6
    #B1 = np.array([Decimal(0)]*num_timeseries,dtype=np.dtype(Decimal)) #saves total cost for blackbox path
    B1 = np.zeros(num_timeseries,dtype=np.float128)
    for i in range(len(PB)-1):
        if PB[i]=='s' or PB[i+1]=='t':
            continue
        node1 = PB[i]
        t=str.split(node1,'-')
        f1 = get_features(int(t[0]),int(t[1]))
        node2 = PB[i+1]
        t=str.split(node2,'-')
        #print(t)		
        f2 = get_features(int(t[0]),int(t[1]))
        #B1 = B1 + (ttl_path-1)*find_weight_matrix(f1,f2)
        B1 = np.add(B1,find_weight_matrix(f1,f2))
    B = np.subtract(B1,B2)
    allB = np.column_stack((B,B1,B2))
    np.savetxt(save_dir+save_file+'_B.txt',allB,delimiter=',')

def create_graph_path(key,node_start,node_end,PB,Prest,idx,timestamp,lock1,lock3,K):
    #getcontext().prec = 6
    #eps=np.float64(10**-320)
    for j in range(1,node_start):
        #if is_in_bound(1,j,node_start-j,diff_l,diff_h):
        #lock3.acquire()
        node1 = str(j)+'-'+str(node_start)
        #if not is_edge_in_pb(PB, node1, key):
        path_in_edge=find_total_path_with_edge(j, node_start, node_end, 1, timestamp)
        path_in_edge-=is_edge_in_pb(PB, node1, key)
        node1f= get_features(j,node_start)
        t=str.split(key,'-')
        node2f= get_features(int(t[0]),int(t[1]))
        #edgew = path_in_edge*find_weight_matrix(node1f,node2f)
        edgew = find_weight_matrix(node1f,node2f)
        for i in range(num_timeseries):
            pijk = Decimal(path_in_edge)/K
            edge_cost= np.float128(pijk)*np.float128(edgew[i])
            if pijk<EPS:
                edge_cost=np.float128(0)
            #edge_cost = float(format(edge_cost,'.5f'))
            Prest[i]=Prest[i]+edge_cost
            #print('Prest')
        #lock3[i].release()
        #lock3.release()

def generate_graph(PB,diff_l,diff_h):
    print('generating graph...')
    #getcontext().prec = 6
    #ttl_path=Decimal(0.0)
    ttl_path=Decimal((2**(timestamp-2)-1))
	#eps_edges=np.float64(0.0)
    '''    
    for idx in range(2,timestamp+1):
        #if is_in_bound(1,idx,idx-1,diff_l,diff_h):
        ttl_path+= Decimal(find_total_path_with_edge(1, 1, idx, 1, timestamp))
    '''
    lock1 = mp.Lock()
    lock2 = mp.Lock()
    lock3 = mp.Lock()
    #for i in range(num_timeseries):
     #   lock3.append(mp.Lock())
    manager = mp.Manager()
    #ttl_Prest = np.array([Decimal(0)]*num_timeseries,dtype=np.dtype(Decimal))
    ttl_Prest = np.zeros(num_timeseries,dtype=np.float128)
    #print('total prest type')	
    #print(type(ttl_Prest[0]))
    #print(ttl_Prest.shape)
    #'''
    ttl_nodes = timestamp*timestamp
    num_cores = mp.cpu_count()
    if num_cores > ttl_nodes:
        n_jobs = ttl_nodes
    else: 
        n_jobs = num_cores
    batch_size = int(ttl_nodes/n_jobs)
    node_num=1 	
    start = time.time()
    for batch in range(batch_size):
        job=[]
        num_path = mp.Value('i',0)
        Prest = shm.full(num_timeseries,0,dtype=np.float128)
        #Prest = shm.full(num_timeseries,Decimal(0),dtype=np.dtype(Decimal))
        #print('prest type when created:')
        #print(type(Prest[0]))
        for idx in range(node_num,min(node_num+n_jobs,ttl_nodes)):	
            node_start= int(idx/timestamp+1)
            node_end= int(idx%timestamp+1)
            if node_start>=node_end: #or not is_in_bound(node_start,node_end,node_end-node_start,diff_l,diff_h):
                continue
            #firstend=node_start+diff_l
            #print(node_start,node_end,firstend,diff_l)
            #if (node_end-firstend)<diff_l:
             #   continue
            key= str(node_start)+'-'+str(node_end)
            print(key)
            p = mp.Process(target=create_graph_path, args=(key,node_start,node_end,PB,Prest,idx,timestamp,lock1,lock3,ttl_path)) 
            job.append(p)
            p.start()
        node_num+=n_jobs	
        _ = [p.terminate() for p in job]
        _ = [p.join() for p in job]
        #ttl_path+=num_path.value
        ttl_Prest = np.add(ttl_Prest,Prest)
        #print('process Prest:')
        #print(Prest)
    #'''
    stop= time.time()
    comp_time=stop-start
    print('time: %.4f'%comp_time)
    #print('Prest:')
    #print(ttl_Prest)
    #print('nodes')
    #print(graph_nodes.keys())
    return ttl_Prest,ttl_path,comp_time

def main(segData, matlab_path, save_dir,save_file):
    PB,diff_l,diff_h = read_segment_data(segData) #blackbox path list
    diff_l=1
    diff_h=timestamp
    
    Prest,ttl_path,ttime = generate_graph(PB,diff_l,diff_h)
    np.savetxt(save_dir+save_file+'_Prest.txt',Prest,delimiter=',')
    f2 = open(save_dir+save_file+'_ttlpath.txt', 'w+')
    f2.write('%.18e\n'%ttl_path)
    f2.write('%.6f\n'%ttime)
    f2.close()
    write_output_file(save_dir,save_file,PB,Prest,ttl_path)
    #print('# paths:')
    #print(ttl_path)
    #'''
    #alpha = np.loadtxt(save_dir+save_file+'_alpha.txt',delimiter=',')
    B2 = np.array(Prest)    
    alpha,B1= get_gloal_exp(PB,ttl_path,graph_nodes,B2,num_timeseries,1,timestamp, matlab_path, save_dir,save_file) 
    print('global explanation weight:')
    print(alpha)
    read_local_exp(PB,alpha)
    #'''
    
if __name__ == '__main__':
    datadir=sys.argv[1]
    data = np.loadtxt(datadir,delimiter=',')
    data=data[:,:3000]
    timestamp = data.shape[0]
    num_timeseries = data.shape[1] #no. of timeseris
    EPS=Decimal(10**-320)
    segFile = sys.argv[2]#'segment/toy'
    save_file=sys.argv[3]#'cnr_syn_ticc'
    matlab_path=sys.argv[4]
    save_dir= sys.argv[5]
    normalized = sys.argv[6].lower()
    if normalized=='true':
        #data = (data-data.min(0))/(data.max(0)-data.min(0))#for groundtruth datasets of floss
        data = (data-np.min(data))/(np.max(data)-np.min(data)) #for wiki, hurricane datasets
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    #save_dir= 'result_check_X/'
    print(data.shape)
    origSeg = np.loadtxt(segFile+'_seg'+'.txt',delimiter=',',dtype=int)
    #f=4 #mean, median, variance for testing
    main(origSeg, matlab_path,save_dir,save_file)

