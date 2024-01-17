#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: JiM as a subset of make_model_gif.py

Modified by JiM in Winter 2023-24 to add emolt obs and ptown wind
Stored in Github as "fvcom_viz" repository
See hardcodes at top of code where there is, for example, "start_date" and "ndays" 
Note: You may need to adjust the "clevs" to get good range of colors.
Note: You may want to adjust the Basemap resolution to "c" for crude in experiments and "f" for full resolution
Note: You may need to "conda install -c conda-forge basemap-data-hires" in order to use the higher resolution coastlines
Note: 
"""
#hardcodes########################
#area='NEC'#'SNW'#'NorthShore'#'SNW'#'GBANK_RING'#'Gloucester'
area='inside_CCBAY'
start_date='2024-01-16'#'2013-04-01'
time_str='2024-01-16 0000UTC'
clevs=[65.,80.,.5]#gb ring surf June
clevs=[50.,72.,.5]#ns bottom June
clevs=[52.,78.,.5]#gb ring June
clevs=[58.,74.,.5]#SNE-W in July
clevs=[58.,80.,.5]#SNE in August
clevs=[56.,75.,.5]#NEC in December
clevs=[43.,52.,.2]#CCBay in Dec
clevs=[32.,50.,.2]#CCBay in late Dec
dtime=[]
units='degF'
surf_or_bot=0#0 0 for surface 
subsample=5
maxvel = 1.0 # for quiver legend
#########
import os,imageio
import conda
import pandas as pd
from pylab import *
from scipy.interpolate import griddata as gd
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ['PROJ_LIB'] = '/home/user/anaconda3/pkgs/proj-8.2.1-h277dcde_0/share'
from mpl_toolkits.basemap import Basemap
import matplotlib.tri as Tri
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,timezone
import time
import zlconversions as zl
from conversions import mps2knots
import sys
import warnings
warnings.filterwarnings("ignore") # gets rid of warnings at runtime but you may want to comment this out to see things
import netCDF4 # for wind access
from math import sqrt
try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import ssl # added this in Oct 2021 when I got a "certificate_verify_failed" message
ssl._create_default_https_context = ssl._create_unverified_context

def getgbox(area):
  # gets geographic box based on area
  if area=='SNE':
    gbox=[-71.,-67.,39.5,42.] # for SNE shelf east
  elif area=='SNW':
    gbox=[-71.5,-69.5,40.,41.75] # for SNw shelf west
  elif area=='MABN':
    gbox=[-73.,-68.,39.,42.] # for SNw shelf west  
  elif area=='OOI':
    gbox=[-71.5,-69.,39.5,41.6] # for OOI
  elif area=='GBANK':
    gbox=[-71.,-66.,40.,42.] # for GBANK
    labint=1.0
    dept_clevs=[50,100,150]  
    #plt.text(x,y,' Georges Bank',fontsize=16, rotation=3
  elif area=='GBANK_RING':
    gbox=[-71.,-65.,39.,42.5] # for typical GBANK Ring 
    labint=1.0
    dept_clevs=[50,100,150]
  elif area=='GS':           
    gbox=[-71.,-63.,38.,42.5] # for Gulf Stream
  elif area=='NorthShore':
    gbox=[-71.,-69.5,41.75,43.25] # for north shore
    labint=.50
    dept_clevs=[50,100,150] 
  elif area=='Gloucester':
    gbox=[-71.,-70.,42.25,43.] # for north shore
  elif area=='IpswichBay':
    gbox=[-71.,-70.,42.5,43.] # for IpswitchBay
    labint=0.2
    dept_clevs=[30,50,100, 150]
  elif area=='CCBAY':
    gbox=[-70.75,-69.8,41.5,42.23] # CCBAY
    labint=0.5
    dept_clevs=[30,50,100]
  elif area=='inside_CCBAY':
    gbox=[-70.75,-70.,41.7,42.15] # inside CCBAY
    labint=0.5
    dept_clevs=[30,50,100]
  elif area=='NEC':
    gbox=[-68.,-63.,38.,43.5] # NE Channel
  elif area=='NE':
    gbox=[-76.,-66.,35.,44.5] # NE Shelf
    labint=1.0
    dept_clevs=[50,100,1000]
  else:
    gbox=[-76.,-66.,35.,44.5]
    labint=1.0
    dept_clevs=[30,50,100, 150,300,1000]

  return gbox,labint,dept_clevs
    

if clevs[1]<32.:
    units='degC'
else:
    units='degF'

[gb,labint,dept_clevs]=getgbox(area)
clevs=np.arange(clevs[0],clevs[1],clevs[2])  #for all year:np.arange(34,84,1) or np.arange(34,68,1)
dtthis=datetime(int(time_str[0:4]),int(time_str[5:7]),int(time_str[8:10]),int(time_str[11:13]),int(time_str[13:15]))

if dtthis<datetime(2020,6,1,0,0,0):
    url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
elif dtthis>datetime.now()-timedelta(days=2):    
    url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
    url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_GOM7_FORECAST.nc'
else:
    url='http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_NORTHEAST_FORECAST.nc'
nc = netCDF4.Dataset(url).variables
time_var = nc['time']
itime = netCDF4.date2index(dtthis,time_var,select='nearest')
# Get lon,lat coordinates for nodes (depth)
lats = nc['lat'][:]
lons = nc['lon'][:]
figure(figsize=(12,8))
subplot(111,aspect=(1.0/cos(mean(lats)*pi/180.0)))
# Get lon,lat coordinates for cell centers (depth)
latc = nc['latc'][:]
lonc = nc['lonc'][:]
# Get Connectivity array
nv = nc['nv'][:].T - 1 
# Get depth
depth = nc['h'][:]  # depth
dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystr = dtime.strftime('%Y-%b-%d %H:%M')

#x, y = m(lons, lats) # compute map proj coordinates.

if surf_or_bot==3:# vertically averaged
    temp = nc['temp'][itime,0,:]#if len(clevs)==0:
    u = nc['ua'][itime,:]# surface fields
    v = nc['va'][itime,:]
else:    
    temp = nc['temp'][itime, surf_or_bot, :]*1.8+32
    u = nc['u'][itime,surf_or_bot,:]
    v = nc['v'][itime,surf_or_bot,:]
if len(clevs)==0:
	clevs=np.arange(int(min(temp)),int(max(temp)+1),1)
# find velocity points in bounding box
idv = np.argwhere((lonc >= gb[0]) & (lonc <= gb[1]) & (latc >= gb[2]) & (latc <= gb[3]))
idv=idv[0::subsample]
tri = Tri.Triangulation(lons,lats, triangles=nv)
tricontourf(tri, temp,levels=clevs,shading='faceted',cmap=plt.cm.rainbow,zorder=0)
axis(gb)

gca().patch.set_facecolor('0.5')
cbar=colorbar()
cbar.set_label('SST (degF)', rotation=-90)
Q = quiver(lonc[idv],latc[idv],u[idv],v[idv],scale=20)
maxstr='%3.1f m/s (~2 knots)' % maxvel
qk = quiverkey(Q,0.9,0.08,maxvel,maxstr,labelpos='W')

print(dtime)
#cbar = m.colorbar(cs,location='right',pad="2%",size="5%")
#cbar.set_label(units,fontsize=25)


# add title
clayer=''# default current layer
if (surf_or_bot==-1):# & (model=='FVCOM'):
    layer='bottom'
elif (surf_or_bot==0):# & (model=='FVCOM'):
    layer='surface'
elif surf_or_bot==3:
    clayer='VA_'
    layer='surface'
plt.title('FVCOM '+layer+' model temp (color) & current (black) '+clayer+' depth (m) ',fontsize=10,fontweight='bold')
plt.suptitle(time_str, fontsize=18) 
