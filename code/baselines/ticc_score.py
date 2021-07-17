# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:55:35 2019

@author: Anika
"""

from TICC_solver import TICC
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Visualization_function import visualize
import networkx as nx

def get_column_names(num_sensor):
    columns=[]
    for j in range(num_sensor):
       columns.append('S'+str(j)) 
    return columns

def get_importance_score(cluster_MRFs,num_sensor,num_cluster,save_file):
    #names = get_column_names(num_sensor)
    cluster = 0
    cluster_score=[]
    threshold=2e-5
    for cluster in range(num_cluster):
        out = np.zeros(cluster_MRFs[0].shape, dtype = np.int)
        A = cluster_MRFs[cluster]
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if np.abs(A[i,j]) > threshold:
                    out[i,j] = 1
        G = nx.from_numpy_matrix(out)
        #importance = nx.degree_centrality(G)
        importance = nx.betweenness_centrality(G)
        print('cluster '+str(cluster))
        print(importance)
        cluster_score.append(importance)
        in_cov_file= save_file+'_clusterMRF'+str(cluster)+'.txt'
        np.savetxt(in_cov_file, A, fmt='%.5f', delimiter=',')
        '''
        file_name = "Cross time graphs"+str(cluster)+"maxClusters="+str(maxClusters-1)+".jpg"
        out2 = out[(num_stacked-1)*n:num_stacked*n,]
        if write_out_file:
            visualize(out2,-1,num_stacked,names,file_name)
        '''
        return cluster_score

def run_ticc(data,save_file):
    num_cluster= 4 #chickendance
    #num_cluster= 3 #sudden_cardiac, synthetic
    n= data.shape[1]
    ticc = TICC(window_size=5, number_of_clusters=num_cluster, lambda_parameter=11e-2, beta=600, maxIters=100, threshold=2e-15,
            write_out_file=True, prefix_string="ration_folder/", num_proc=1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=data,rf=-1,rl=-1,rational=True)
    cluster_score= get_importance_score(cluster_MRFs,n,num_cluster,save_file)
    return np.array(cluster_score),cluster_assignment

def get_tp_segments(gt_cuts,thrsh,predicted_cuts,cluster_assignment):
    gt_cuts_copy=gt_cuts[:]
    tp_cuts=[]
    tp_cluster_prev=[]
    tp_cluster_next=[]
    for cut in predicted_cuts:
        temp=[match(cut,i,thrsh) for i in gt_cuts_copy]
        for i in range(len(temp)):
            if temp[i]==1:
                seg=gt_cuts_copy[i]
                tp_cluster_prev.append(cluster_assignment[seg-2])
                tp_cluster_next.append(cluster_assignment[seg+2])
                tp_cuts.append(gt_cuts_copy[i])
                
        for i in range(len(gt_cuts_copy)):
            if temp[i]==1:
                del gt_cuts_copy[i]
                break
    return tp_cuts, tp_cluster_prev,tp_cluster_next
      
def match(cut,i,trsh):
    if cut<i+trsh and cut>i-trsh:
        return 1
    else:
        return 0
   
def main(data, gt_cuts,save_file):
    cluster_score,cluster_assignment=run_ticc(data,save_file)
    predicted_cuts=[]
    thrsh=int(data.shape[0]*0.05)
    for i in range(1,len(cluster_assignment)):
        if cluster_assignment[i]!=cluster_assignment[i-1]:
            predicted_cuts.append(i)
    orig_tp_seg,tp_cluster_prev,tp_cluster_next= get_tp_segments(gt_cuts,thrsh,predicted_cuts,cluster_assignment)
    print('Local rationalization:')
    for seg in gt_cuts:
        print('seg: %d'%seg)
        try:
            idx = orig_tp_seg.index(seg)
            diff_cluster = np.abs(cluster_score[:,tp_cluster_prev[idx]]-cluster_score[:,tp_cluster_next[idx]])
            print('difference of cluster score:')
            print(diff_cluster)
            local_rat = np.argsort(-diff_cluster)
            print(local_rat)
        except ValueError:
            print("-1")

    
if __name__ == '__main__':
    fname = "blackbox_data/chicken21.csv" #sys.argv[1]
    gt_segfile='segment/chicken21_gt_seg.txt' #sys.srgv[2]
    gt_cuts= np.loadtxt(gt_segfile,delimiter=',',dtype=int)
    data = np.loadtxt(fname,delimiter=',')
    save_file= 'result_ticc_score/'
    main(data,gt_cuts,save_file)
