# -*- coding: utf-8 -*-
"""
Created on Mon May 06 17:15:13 2019

@author: Anika
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np

def preprocess(datafile,mapfile,col):
    lines=[]

    with open(datafile) as f:
        lines = f.read().splitlines()
        #countyName = map(lambda it: it.strip('\"\t\n\r'), lines)
        cities = lines[0].strip().split(',')
        cities = cities[1:]
        cities = [x.lower() for x in cities]
        lines = lines[1:]
        #print cities
    E = np.loadtxt(mapfile,delimiter=',',usecols=range(col))
    #print E
    return cities,E


def draw_us_map(cities, ei,ints):
    # Set the lower left and upper right limits of the bounding box:
    lllon = -119
    #urlon = -75.61
    urlon = -64
    lllat = 22
    urlat = 49
    # and calculate a centerpoint, needed for the projection:
    centerlon = float(lllon + urlon) / 2.0
    centerlat = float(lllat + urlat) / 2.0

    m = Basemap(resolution='i',  # crude, low, intermediate, high, full
                llcrnrlon = lllon, urcrnrlon = urlon,
                lon_0 = -95,lat_1=32,lat_2=45,
                llcrnrlat = lllat, urcrnrlat = urlat,
                lat_0 = centerlat,
                projection='lcc')
    shp_infoS = m.readshapefile('states_21basic/states', 'states',
                               drawbounds=True, color='black')
    geolocator = Nominatim()
    idx=0
    for city in cities:
        #print(idx,city)
        loc = geolocator.geocode(city)
        x, y = m(loc.longitude, loc.latitude)
        if (ei[idx]*ints)>1:
            sc=1
        else:
            sc=ei[idx]*ints
        fc = (1,0,0,sc)
        m.plot(x,y,marker='o',color='Red',markersize=int(ei[idx]*ints))
        idx+=1
    # Get rid of some of the extraneous whitespace matplotlib loves to use.
    #plt.title('US Cities')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()
    
if __name__ == "__main__":
    file = 'data-to-plot/TB_data_sampled.csv'
    mapfile = 'data-to-plot/TB_ticc_alpha.txt'
    isglobal=1
    if isglobal:
        col=1
        s= 1
    else:
        col=4
        s=4
    cities,E=preprocess(file,mapfile,col)
    #E_normed = abs(E)
    #E_normed = (E-np.min(E))/(np.max(E)-np.min(E))
    #s= E.shape[1]
    #print E_normed
    ints=50 #Matthew
    #draw_us_map(county,E[:,0],ints)
    #E[E<0]=0
    E= (E- np.min(E))/(np.max(E)-np.min(E))
    for i in range(s):
        if isglobal:
            draw_us_map(cities,E,ints)
        else:
            draw_us_map(cities,E[:,i],ints)
        