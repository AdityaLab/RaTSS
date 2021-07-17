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
    with open('flu_2017.csv') as f: #subcountyname
        lines = f.read().splitlines()
    #countyName = map(lambda it: it.strip('\"\t\n\r'), lines)
    countyName = lines[0].strip().split(',')
    lines = lines[1:]
    #print clustercounty
    #print countyName
    E = np.loadtxt('flu2017_floss_exp.csv',delimiter=',',usecols=range(2)) #E1
    #print E
    return countyName,E
    
def draw_us_map(counties, ei,ints):
    # Set the lower left and upper right limits of the bounding box:
    lllon = -119
    urlon = -64
    lllat = 20
    urlat = 49
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

    
    states = [] 
    ax = plt.gca()
    
    for state,shape in zip(m.states_info,m.states):
        statename= state['STATE_ABBR']
        #print state
        if statename not in states:
            states.append(statename)
        if statename in counties:     
            x, y = zip(*shape) 
            m.plot(x, y, marker=None,color='black',linewidth=0.3)
    
            
    #print(states)
    ttl =0
    
    for i, county in enumerate(m.states_info):
        statename = county["STATE_ABBR"]
        fc=(1, 1, 1, 0)
        if statename in counties:
            idx = int(counties.index(statename))            
            fc = (1,0,0,ei[idx]*ints)
            ttl+=1
            countyseg = m.states[i]
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
    #E= (E- np.min(E))/(np.max(E)-np.min(E))
    print(E.shape)
    ints=1
    for i in range(s):
        m=1.0
        #if i==0:
         #   E[:,i]=E[:,i]*5
        draw_us_map(county,E[:,i],ints)
        plt.title('US Map')
        # Get rid of some of the extraneous whitespace matplotlib loves to use.
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()
    