# -*- coding: utf-8 -*-
"""
Created on Thu May 17 23:43:24 2018

@author: Anika
"""

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import string
import numpy as np

def preprocess():
    
    #fileName='clusters_U_lam1_0.1_lam2_0.1lam3_0.1_clusV_3_clusU_2.csv'
    
    lines=[]
    with open('Harvey_60min_sample.csv') as f: #subcountyname
        lines = f.read().splitlines()
    #countyName = map(lambda it: it.strip('\"\t\n\r'), lines)
    countyName = lines[0].strip().split(',')
    lines = lines[1:]
    #print clustercounty
    #print countyName
    E = np.loadtxt('harvey_exp.csv',delimiter=',',usecols=range(4)) #E1
    #print E
    return countyName,E
    
def draw_us_map(counties, ei):
    # Set the lower left and upper right limits of the bounding box:
    lllon = -106.65
    urlon = -93.51
    lllat = 25.84
    urlat = 36.5
    # and calculate a centerpoint, needed for the projection:
    centerlon = float(lllon + urlon) / 2.0
    centerlat = float(lllat + urlat) / 2.0

    m = Basemap(resolution='i',  # crude, low, intermediate, high, full
                llcrnrlon = lllon, urcrnrlon = urlon,
                lon_0 = centerlon,
                llcrnrlat = lllat, urcrnrlat = urlat,
                lat_0 = centerlat,
                projection='tmerc')
    #m.drawmapboundary(fill_color='white')
    # Read state boundaries.
    shp_infoS = m.readshapefile('states_21basic/states', 'states',
                               drawbounds=True, color='black')

    # Read county boundaries
    shp_infoC = m.readshapefile('us_counties_2016_500k/cb_2016_us_county_500k',
                               'counties',
                               drawbounds=False)
    MAXSTATEFP = 73
    states = [None] * MAXSTATEFP
    ax = plt.gca()
    '''
    for state in m.states_info:
        statefp = int(state["STATE_FIPS"])
        print state
        if not states[statefp]:
            states[statefp] = state["STATE_NAME"]
    '''
    
    for county,shape in zip(m.counties_info,m.counties):
        #print county
        #print county["NAME"] + ' ' + county["STATEFP"]
        stateId=int(county["STATEFP"])
        if stateId==48:
            x, y = zip(*shape) 
            m.plot(x, y, marker=None,color='black',linewidth=0.2)
    
    #print shp_infoC
    #print m.counties_info
    #color counties
    #county_colors = ['r','g','b','m']
    
    ttl =0
    for i, county in enumerate(m.counties_info):
        countyname = county["NAME"]
        stateId= int(county["STATEFP"])
        if stateId==48:
            countyname=countyname+'_TX'
        fc=(1, 1, 1, 0)
        if countyname in counties:
            idx = int(counties.index(countyname))            
            #if stateId==48 :
            fc = (1,0,0,ei[idx])
                #fc = (1,0,0)
            ttl+=1
            countyseg = m.counties[i]
            poly = Polygon(countyseg, facecolor=fc)  # edgecolor="white"
            ax.add_patch(poly)
        elif stateId==48:
            fc = (1,1,1,1)
            countyseg = m.counties[i]
            poly = Polygon(countyseg, facecolor=fc)  # edgecolor="white"
            ax.add_patch(poly)
                
    print(ttl)
if __name__ == "__main__":
    county,E=preprocess()
    #E_normed = (E-E.min(0))/(E.max(0)-E.min(0))
    #E_normed = abs(E)
    #s= E_normed.shape[1]
    s= E.shape[1]
    #print E_normed
    #draw_us_map(cluster,county,E[:,0])
    E[E<0]= 0
    #E = np.abs(E)
    E= (E- np.min(E))/(np.max(E)-np.min(E))
    print(E.shape)
    for i in range(s):
        m=1.0
        if i==0:
            E[:,i]=E[:,i]*5
        draw_us_map(county,E[:,i])
        plt.title('US Counties')
        # Get rid of some of the extraneous whitespace matplotlib loves to use.
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()
    