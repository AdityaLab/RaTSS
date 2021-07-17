import os
import sys
import ast

def main(file_):
	print file_
	if 'Chicken19' in file_:
		gt_cuts=[200,350,520,800,900,1100,1250]
		trsh=77
		mode='test_19_10'
	elif 'Chickendance21' in file_:
		gt_cuts=[45,83,123,175,208,243,284]
		trsh=17
		mode='test_21_10'
	elif 'HAR' in file_:
		gt_cuts=[3000,6000,9000,12000]
		trsh=500
		mode='HAR'
	elif 'NewNDSSL' in file_:
		gt_cuts=[0.2]
		trsh=0.04
		mode='NewNDSSL'
	elif 'nilm1' in file_:
		gt_cuts=[85,185,215,252,265,305,360,390,500,525,550,600,660,690]
		trsh=22
		mode='nilm1'
	elif 'nilm3' in file_:
		gt_cuts=[85,185,215,305,360,390,500,525,550,600,660,690]
		trsh=22
		mode='nilm3'	
	elif 'WJR' in file_:
		#gt_cuts=[100,200]
		#gt_cuts=[100] #100 for wjr_199
		gt_cuts=[310,700,800] #for wjr_floss
		trsh=50#15
		mode='wjr'
	elif 'syn' in file_:
		gt_cuts=[150,250,500,700,820,920]
		trsh = 50
		mode='syn'
	elif 'GrandMal' in file_:
		gt_cuts=[300,600]
		trsh = 45
		mode='grandmal'
	with open(file_) as f:
		lines=f.readlines()
	line=lines[0]
	items=line.strip().split(',')
	items=[float(i) for i in items]
	print items
	f1=calculate_f1(gt_cuts,trsh,items)
	sf=open('f1_f.txt','a')
	sf.write(file_+'\n')
	sf.write(str(f1)+'\n')
	sf.close()
def calculate_f1(gt_cuts,trsh,cuts):
	tp=0
	fp=0
	fn=0
	l=len(gt_cuts)
	gt_cuts_copy=gt_cuts[:]
	tp_cuts=[]
	for cut in cuts:
		temp=[match(cut,i,trsh) for i in gt_cuts_copy]
		#print temp
		if sum(temp)>0:
			tp+=1
			for i in range(len(temp)):
				if temp[i]==1:
					tp_cuts.append(gt_cuts_copy[i])
		else:
			fp+=1
		#print temp
		#print gt_cuts
		for i in range(len(gt_cuts_copy)):
			if temp[i]==1:
				del gt_cuts_copy[i]
				break
	fn=l-tp
	pr=tp*1.0/(tp+fp)
	#print tp,fp,fn
	rec=tp*1.0/(tp+fn)
	if pr==0 and rec==0:
		return 0
	return tp_cuts,2*pr*rec/(pr+rec)
		
def match(cut,i,trsh):
	if cut<i+trsh and cut>i-trsh:
		return 1
	else:
		return 0

if __name__=='__main__':
	main('Chicken19_60minsample_dyn_cuts.txt')
	main('Chickendance21Sub_Normalized_321_dyn_cuts.txt')
	main('nilm1_60min_sample_dyn_cuts.txt')
	main('nilm3_60min_sample_dyn_cuts.txt')
	main('WJR_floss_dyn_cuts.txt')
	#main('WJRSub_60min_sample_199_dyn_cuts.txt')
	main('syn4_dyn_cuts.txt')
	main('GrandMalSeizuresSub_dyn_cuts.txt')
	#main('HAR_cuts.txt')
	#main('NewNDSSL_cuts.txt')
	
