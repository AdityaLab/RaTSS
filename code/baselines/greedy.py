# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:02:13 2019

@author: Anika
"""
import numpy as np
from TICC_solver import TICC
import numpy as np
import sys
from get_f1_f import calculate_f1
"""
The greedy algorithm: 
1. calculate the f1 score of the segmentation 
run by segmentation algorithm
2. now one by one remove each time series run segmentation algorithm,
and calculate f1-score 
3. Get the top K timeseries which if removed  lose true positive segments
"""
def run_ticc(data):
    ticc = TICC(window_size=1, number_of_clusters=2, lambda_parameter=11e-2, beta=600, maxIters=50, threshold=2e-15,
            write_out_file=True, prefix_string="ration_folder/", num_proc=1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=data,rf=-1,rl=-1,rational=True)
    cuts=[]
    for i in range(1,len(cluster_assignment)):
        if cluster_assignment[i]!=cluster_assignment[i-1]:
            cuts.append(i)
    return cuts

def find_local_rat(full_tp_segs,no_timeseries,seg):
    local=[]
    for ts in range(no_timeseries):
        tp_ts = full_tp_segs[ts,:]
        if seg not in tp_ts:
            local.append(ts)
    return np.array(local)
    
def find_rationalization(data,gt_cuts):
    full_predicted_cuts = run_ticc(data)
    thrsh=int(data.shape[0]*0.05)
    orig_tp_seg,full_f1= calculate_f1(gt_cuts,thrsh,full_predicted_cuts)
    f1_score=[]
    full_tp_segs=[]
    no_timeseries = data.shape[1]
    for i in range(no_timeseries):
        p_data=np.delete(data,i,axis=1)
        p_cuts = run_ticc(p_data)
        tp_seg,p_f1= calculate_f1(gt_cuts,thrsh,p_cuts)
        f1_score.append(full_f1-p_f1)
        full_tp_segs.append(tp_seg)
    f1_score = np.array(f1_score)
    global_rat = np.argsort(f1_score)
    print('global rationalization:')
    print(global_rat)
    return orig_tp_seg,full_tp_segs

def main(data,gt_cuts):
    orig_tp_seg, full_tp_segs= find_rationalization(data,gt_cuts)
    no_timeseries = data.shape[1]
    for seg in gt_cuts:
        print('seg: %d'%seg)
        if seg in orig_tp_seg:
            local_rat = find_local_rat(full_tp_segs,no_timeseries,seg)
            print(local_rat)
        else:
            print("-1")

if __name__ == '__main__':
    fname = "blackbox_data/chicken21.csv" #sys.argv[1]
    gt_segfile='segment/chicken21_gt_seg.txt' #sys.srgv[2]
    gt_cuts= np.loadtxt(gt_segfile,delimiter=',',dtype=int)
    data = np.loadtxt(fname,delimiter=',')
    main(data,gt_cuts)
    
    
    
    