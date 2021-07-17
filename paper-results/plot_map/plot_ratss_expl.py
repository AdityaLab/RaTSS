import os
import sys
import networkx as nx
import ast
import random
import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
def plot_result(data_file, E_file, segfile, save_dir, savefile, thres, hurricane,chooseTopK):
    """
    plot the network behind the time series for each cut point, highlight the important time series in the explanation accordingly.
    also plot the time series data and the segmentation
    """
    with open(data_file)as f:
        lines = f.readlines()
    counties = lines[0].strip().split(',')
    #lines = lines[1:]
    data = []
    for line in lines:
        data.append([float(x) for x in line.strip().split(',')])
    data = np.array(data)
    if hurricane=='wiki':
        data = data[:,:3000]
        #data=np.delete(data,653,1)
        counties=np.arange(1,3001)
        #counties=counties[:1000]
        #counties=counties.pop(653)
        print data.shape
        print len(counties)
   
    l = len(lines)
    print 'reading the segmentation result'
    with open(segfile) as f:
        line = f.readline()
    #S = ast.literal_eval(line.strip())
    S = [int(x) for x in line.strip().split(',')]
    print 'reading the explanation'
    E = np.loadtxt(E_file,dtype=float,delimiter=',')
    if hurricane=='harvey':
        E=E[:,1:]
    E=abs(E)    
    #E = (E - E.min(0)) / (E.max(0) - E.min(0))
    E = E/ E.sum(0)
    print E.shape
    colormap = plt.cm.gist_ncar
    county_index = np.linspace(0, 0.9, len(counties))
    #random.shuffle(county_index)

    #plt.gca().set_color_cycle([colormap(i) for i in county_index])
    plt.gca().set_prop_cycle('color',[colormap(i) for i in county_index])
    print 'plotting the segmentation'
    for i in range(len(counties)):
        plt.plot(range(l), data[:, i])
    for s in range(len(S)):
        if hurricane=='irma' and s==1:
            continue
        if hurricane=='harvey' and s==0:
            continue
        plt.axvline(x=float(S[s]), linestyle = 'dashed', color = 'k')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15, rotation = 45)
    if hurricane=='diptheria':
        #plt.xlim(0, 40)
        plt.ylim(0,1800)
    if hurricane=='covid':
        plt.xlim(40, 104)
        plt.ylim(0,5200)
    plt.savefig(save_dir + savefile +'.png')
    plt.clf()
    print 'plotting the explanations'
    segments = []
    bd = 0
    start=0
    #if hurricane=='harvey':
    #    start=1
    for i in range(start,len(S)):
        #if hurricane =='irma' and i==1
        if i - 1 >= start:
            lc = S[i - 1] + bd
        else:
            lc = 0 + bd 
        if i + 1 < len(S):
            rc = S[i + 1] - bd
        else:
            rc = len(data) - bd
        segments.append([lc, rc])
    sf = open(save_dir + savefile + '_cut' + '_impC.txt', 'wb')
    for i in range(len(segments)):
        imp_c = get_imp_c(E, i, thres, chooseTopK) 
        sf.write('cut ' + str(i) + '\n')
        for ic in imp_c:
            sf.write(str(counties[ic]) + '\t' + str(E[ic, i]) + '\n')
        sf.write('\n')
        seg = segments[i]
        for c in imp_c:
            #print seg[0], seg[1], c
            plt.plot(range(seg[0], seg[1]), data[seg[0]:seg[1], c])
    sf.close()
    for s in S:
        plt.axvline(x=float(s), linestyle = 'dashed', color = 'k')
    
    if hurricane==covid:
        plt.xlim(40, 104)
        plt.ylim(0,5200)
    else:
        plt.xlim(-10, 804)
        plt.ylim(0,1500000)
    plt.savefig(save_dir + savefile +'_exp.png')
    plt.clf()
    for i in range(len(segments)):
        imp_c = get_imp_c(E, i, thres, chooseTopK) 
        seg = segments[i]
        if hurricane == 'harvey':
            ym = 120000
            xm = 300
        elif hurricane == 'irma':
            ym = 900000
            xm = 180
        elif hurricane == 'flu':
            ym = 16
            xm = 55#250
        elif hurricane == 'chicken21':
            ym=1
            xm=330
        elif hurricane == 'diptheria' or hurricane=='tb':
            ym=1800
            xm=40 
        elif hurricane=='wiki':
            ym=1500000
            xm=800
        elif hurricane=='covid':
            ym=5200
            xm=104
        for c in imp_c:
            #print seg[0], seg[1], c
            plt.plot(range(seg[0], seg[1]), data[seg[0]:seg[1], c], color = colormap(county_index[c]))
        for j in range(len(S)):
            s = S[j]
            if abs(j - i) <= 1: 
                plt.axvline(x=float(s), linestyle = 'dashed', color = 'k')
        if hurricane=='covid':
            plt.xlim(40, xm)
        else:
            plt.xlim(-10, xm)
	#plt.xlim(right=xm)
        plt.ylim(0, ym)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=15, rotation = 45)
        plt.savefig(save_dir + savefile + '_exp_' + str(i) + '.png')
        plt.clf()

def get_imp_c(E, i, thres,chooseTopK):
    #given the explanations, return for the ith cut point, the time series index that account for more than thres of the importance
    j=i
    #if i==1:
     #   j=2
    items = zip(range(len(E)), E[:, j])
    items = sorted(items, reverse = True, key = lambda x:x[1])
    imp_c = []
    #thres = 0.8
    s = 0.0
    for item in items:
        if item[1]>=0:
            if chooseTopK=='true':
                s+=1
            else:
                s += item[1]
            imp_c.append(item[0])
        if s > thres:
            break
    return imp_c

if __name__ == '__main__':
    #data_dir = '../data/non_intrusive_load_monitoring/'
    #filename = 'non_intrusive_load_monitoring_dataset_3.csv'
    #segfile = 'osc_segment_indices_lambda_1_0.5_lambda_2_1000_numiter_300.csv'
    #E_file = 'E_23.txt.npy'
    #save_dir = '../result/nilm3/'
    #non = 'normalized' 

    #data_dir = '../data/'
    #filename = 'Irma_60min_sample.csv'
    #save_dir = '../result/OSC_Hurricane_Results/irma/' + non + '/'
    #segfile = 'segmentation.txt'
    #adj_file = 'Irma.adjlist'
    #E_file = 'E_1.txt.npy'

    #data_file = '../data/chicken21.csv'
    #save_dir = 'plot_results/'
    #segfile = '../segment/chicken21_floss_seg.txt'
    #E_file = '../result/not-absolute/chicken21_floss_exp.csv'
    #savefile = 'chicken21_floss'
    #thres = 0.6
    #hurricane= 'chicken21'
	
    #data_file = '../data/Irma_60min_sample.csv'
    #save_dir = 'plot_results_new/'
    #segfile = '../segment/irma_ratss_seg.txt'
    #E_file = '../result/not-absolute/irma_exp.csv'
    #savefile = 'irma'
    #thres = 0.78
    #hurricane= 'irma'
	
    #data_file = '../data/Harvey_60min_sample.csv'
    #save_dir = 'plot_results_new/'
    #segfile = '../segment/harvey_new_seg.txt'
    #E_file = '../result/not-absolute/harvey_exp.csv'
    #E_file = '../result_check_X/harvey_exp_new.csv'
    #savefile = 'harvey'
    #thres = 0.95 #10
    #hurricane= 'harvey'
    	
    #data_file = '../data/TB_data_sampled_missing.csv'
    #save_dir = 'plot_results/'
    #segfile = '../segment/TB_ticc_seg.txt'
    #E_file = '../result/not-absolute/TB_ticc_exp.csv'
    #savefile = 'TB'
    #thres = 0.95
    #hurricane= 'tb'

	
    #data_file = '../data/Diptheria_data_sampled_missing.csv'
    #save_dir = 'plot_results_new/'
    #segfile = '../segment/TB_ticc_seg.txt'
    #E_file = '../result/not-absolute/diptheria_ticc_exp.csv'
    #savefile = 'diptheria'
    #thres = 0.85
    #hurricane= 'diptheria'
    
    #data_file = '../data/not_normalized_train2_5000.csv'   
    #save_dir = 'plot_results/'
    #segfile = '../segment/wiki_train2_3000_seg.txt'
    #E_file = '../result_parallel_without_prune/wiki_train2_3000_exp.csv'
    #savefile = 'wiki_train2_3000'
    #thres = 0.85
    #hurricane= 'wiki'    

    #data_file = '../data/covid_nytimes.csv'
    #save_dir = 'plot_results_new/'
    #segfile = '../segment/covid_enacted_all_seg.txt'
    #E_file = '../result/not-absolute/covid_enacted_all_exp.csv'
    #savefile = 'covid_enacted_all'
    #thres = 0.98
    #hurricane= 'covid'

    data_file = '../data/covid_times_remove_ny_nj.csv'
    save_dir = 'plot_results_new/'
    segfile = '../segment/covid_enacted_all_2week_seg.txt'
    E_file = '../result/not-absolute/covid_enacted_all_2week_exp.csv'
    savefile = 'covid_enacted_2week'
    thres = 0.95
    hurricane= 'covid'

    chooseTopK='false'
    plot_result(data_file, E_file, segfile, save_dir, savefile, thres, hurricane,chooseTopK)
