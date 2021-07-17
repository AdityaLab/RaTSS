# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 12:01:01 2018

@author: Anika
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:57:26 2018

@author: Anika
"""

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
    lines=[]

    with open('Irma_60min_sample.csv') as f:
        lines = f.read().splitlines()
    #countyName = map(lambda it: it.strip('\"\t\n\r'), lines)
    counties = lines[0].strip().split(',')
    lines = lines[1:]
    #print countyName
    E = np.loadtxt('irma_exp.csv',delimiter=',',usecols=range(5))
    #print E
    return counties,E
    
def draw_us_map(counties, ei,ints):
    # Set the lower left and upper right limits of the bounding box:
    lllon = -87.63
    #urlon = -75.61
    urlon = -74.61
    lllat = 24.4
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
    #print m.states_info
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
        countyname = county["NAME"]
        if stateId==12:
            countyname=countyname+'_FL'
        elif stateId==13:
            countyname=countyname+'_GA'
        elif stateId==37:
            countyname=countyname+'_NC'
        elif stateId==45:
            countyname=countyname+'_SC'
        if countyname in counties:    
            x, y = zip(*shape) 
            m.plot(x, y, marker=None,color='black',linewidth=0.2)
    
    ttl =0
    for i, county in enumerate(m.counties_info):
        countyname = county["NAME"]
        stateId= int(county["STATEFP"])
        if stateId==12:
            countyname=countyname+'_FL'
        elif stateId==13:
            countyname=countyname+'_GA'
        elif stateId==37:
            countyname=countyname+'_NC'
        elif stateId==45:
            countyname=countyname+'_SC'
        #fc=(0.827, 0.827, 0.827,0.5)
        fc=(1,1,1,1)
        if countyname in counties:
            idx = int(counties.index(countyname))            
            #if stateId==48 :#and idx< len(cluster)
                #coloridx = int(cluster[idx])
                #print (countyname + ' %.4f'%ei[idx])
                #color=county_colors[coloridx-1]
            #print countyname   
            fc = (1,0,0,ei[idx]*ints)
                #fc = (1,0,0)
            ttl+=1
        countyseg = m.counties[i]
        poly = Polygon(countyseg, facecolor=fc)  # edgecolor="white"
        ax.add_patch(poly)        
    print ttl
if __name__ == "__main__":
    county,E=preprocess()
    #E_normed = (E-E.min(0))/(E.max(0)-E.min(0))
    #E_normed = abs(E)
    #E= E[:,1:]
    s= E.shape[1]
    #print E_normed
    ints=1 #Matthew
    #draw_us_map(county,E[:,0],ints)
    E[E<0]=0
    #E= (E- np.min(E))/(np.max(E)-np.min(E))
    for i in range(s):
        if i==1 or i==3 or i==4:
            ints=2
        if i==2:
            ints=10
        #if i>2:
         #   ints=15
        draw_us_map(county,E[:,i],ints)
        plt.title('US Counties')
        # Get rid of some of the extraneous whitespace matplotlib loves to use.
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()
    
    