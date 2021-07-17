# 'Actionable Insights in Multivariate Time-series for Urban Analytics', Anika Tabassum, Supriya Chinthavali, Varisara Tansakul, and B. Aditya Prakash

==========================================================================
Paper:
-------------
Both paper and supplementary readings are added here. Check paper.pdf and appendix.pdf
-------------
Usage:
Note: You need to set the correct MATLAB_path in the makefile (Including the MATLAB executable).
```
- Example:
    MATLAB_path = '/usr/local/bin/./matlab'
```
To run Ratss for sample data do as follows,
```
>> make demo  
```
'make demo' will run for the sample data (Covid-19 interventions data in the paper) in data/ directory.

Output:
-------
 #located in result_file directory
 -- covid_interventions_exp.csv: nXs contains the rationalization weight of n time-series in s segments in the segment file.
 
 -- covid_interventions_ei.csv: nXs, each column contains the time-series index ranked by rationalization weights in _exp.csv for each segment in the segment file.
 
 -- covid_interventions_Prest.txt: file of n, contains cost Prest of n time-series 
 
 -- covid_interventions_ttl_path.txt: contains value of ttl path of the constructed segment graph, i.e., K
 
 -- covid_interventions_B.txt: file of nX3, contains cost K*PB-PRest,PB,Prest of n time-series 

Citations:
------------
This paper is under creative common license. If you use our code and paper use the following citations.

@article{tabassum2021actionable,
  title={Actionable Insights in Multivariate Time-series for Urban Analytics},
  author={Tabassum, Anika and Chinthavali, Supriya and Tansakul, Varisara and Prakash, B Aditya},
  journal={7th International Workshop of Mining and Learning Time Series in ACM SigKDD},
  year={2021}
}
