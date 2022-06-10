# -*- coding: utf-8 -*-
# Originally by V Sheremet
# Modified by JiM in Feb 2021 to plot instananeous FVCOM field
# Modified by JiM in Jun 2022 to plot mean monthly FVCOM field & clean

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime as dt
import pandas as pd

###### HARDCODES ###################

area='cc'
#area='gs'#gulf stream
case='hindcast'# vertically ave
#case='forecast'
flow='s' # 's'  for surface and 'a' for vertically averaged

if case[0:4]=='mean':
    FNU='c:/users/james.manning/downloads/PT/gom3_u'+flow+'_'+case+'.npy' # vitalii's output
    FNV='c:/users/james.manning/downloads/PT/gom3_v'+flow+'_'+case+'.npy'
else:
    dtime=dt(2015,6,7,6,0,0)
    FNU = dtime.strftime('%Y-%b-%d_%H:%M')
    layer='surface' 
    if dtime<dt(2017,1,1,0,0,0):
        url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
    else:
        url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'

if area=='gom':
    LON1=-71.;LON2=-64.+1.e-6;LAT1=39.0;LAT2=44.+1.e-6
    loni=np.arange(LON1,LON2,0.02)
    lati=np.arange(LAT1,LAT2,0.02)
    lonii,latii=np.meshgrid(loni,lati)
    # starting points
    lons=np.arange(-69.25,-65.2501,0.25)
    lats=np.arange( 39.75, 42.7501,0.25)
else: #'cc'
    LON1=-70.7;LON2=-69.9+1.e-6;LAT1=41.65;LAT2=42.2+1.e-6
    loni=np.arange(LON1,LON2,0.01)
    lati=np.arange(LAT1,LAT2,0.01)
    lonii,latii=np.meshgrid(loni,lati)
    # starting points
    lons=np.arange(-70.7,-70.1,0.2)
    lats=np.arange( 41.6, 42.1,0.2)
###################  END  OF HARDCODES ##########################    
    
def nearxy(x,y,xp,yp):
    """
    i=nearxy(x,y,xp,yp)
    find the closest node in the array (x,y) to a point (xp,yp)
    input:
        x,y - np.arrays of the grid nodes, cartesian coordinates
        xp,yp - point on a plane
    output:
        i - index of the closest node
        min_dist - the distance to the closest node
    For coordinates on a sphere use function nearlonlat

    Vitalii Sheremet, FATE Project
    """
    dx=x-xp
    dy=y-yp
    dist2=dx*dx+dy*dy
    # dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    return i

def nearlonlat(lon,lat,lonp,latp):
    """
    i=nearlonlat(lon,lat,lonp,latp)
    find the closest node in the array (lon,lat) to a point (lonp,latp)
    input:
        lon,lat - np.arrays of the grid nodes, spherical coordinates, degrees
        lonp,latp - point on a sphere
    output:
        i - index of the closest node
        min_dist - the distance to the closest node, degrees
    For coordinates on a plane use function nearxy

    Vitalii Sheremet, FATE Project
"""
    cp=np.cos(latp*np.pi/180.)
    # approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
    # dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    #    min_dist=np.sqrt(dist2[i])
    return i 

def find_kf(Grid,xp,yp):
    """
    kf,lamb0,lamb1,lamb2=find_kf(Grid,xp,yp)

    find to which triangle a point (xp,yp) belongs
    input:
        Grid - triangular grid info
        xp,yp - point on a plane
    output:
        kf - index of the the triangle
        lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

    Vitalii Sheremet, FATE Project
    """

# coordinates of the vertices
    kvf=Grid['kvf']
    x=Grid['x'][kvf];y=Grid['y'][kvf]  
    # calculate baricentric trilinear coordinates
    A012=((x[1,:]-x[0,:])*(y[2,:]-y[0,:])-(x[2,:]-x[0,:])*(y[1,:]-y[0,:])) 
    # A012 is twice the area of the whole triangle,
    # or the determinant of the linear system above.
    # When xc,yc is the baricenter, the three terms in the sum are equal.
    # Note the cyclic permutation of the indices
    lamb0=((x[1,:]-xp)*(y[2,:]-yp)-(x[2,:]-xp)*(y[1,:]-yp))/A012
    lamb1=((x[2,:]-xp)*(y[0,:]-yp)-(x[0,:]-xp)*(y[2,:]-yp))/A012
    lamb2=((x[0,:]-xp)*(y[1,:]-yp)-(x[1,:]-xp)*(y[0,:]-yp))/A012
    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    #    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
    #    kf,=np.where((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    return kf,lamb0[kf],lamb1[kf],lamb2[kf]

def find_kf_lonlat(Grid,lonp,latp):
    """
    kf,lamb0,lamb1,lamb2=find_kf(Grid,lonp,latp)

    find to which triangle a point (lonp,latp) belongs
    input:
        Grid - triangular grid info
        lonp,latp - point on a plane
    output:
        kf - index of the the triangle
        lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

    This method is approximate, valid only for small spherical triangles.
    The metric coefficient is evaluated at P.

    derived from find_kf

    Vitalii Sheremet, FATE Project
    """
    cp=np.cos(latp*np.pi/180.)
    xp=lonp*cp;yp=latp
    # coordinates of the vertices
    kvf=Grid['kvf']
    x=Grid['lon'][kvf]*cp;y=Grid['lat'][kvf]  
    # calculate baricentric trilinear coordinates
    A012=((x[1,:]-x[0,:])*(y[2,:]-y[0,:])-(x[2,:]-x[0,:])*(y[1,:]-y[0,:])) 
    # A012 is twice the area of the whole triangle,
    # or the determinant of the linear system above.
    # When xc,yc is the baricenter, the three terms in the sum are equal.
    # Note the cyclic permutation of the indices
    lamb0=((x[1,:]-xp)*(y[2,:]-yp)-(x[2,:]-xp)*(y[1,:]-yp))/A012
    lamb1=((x[2,:]-xp)*(y[0,:]-yp)-(x[0,:]-xp)*(y[2,:]-yp))/A012
    lamb2=((x[0,:]-xp)*(y[1,:]-yp)-(x[1,:]-xp)*(y[0,:]-yp))/A012
    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    #    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
    #    kf,=np.where((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    return kf,lamb0[kf],lamb1[kf],lamb2[kf]

def find_kf2(Grid,xp,yp):
    """
    kf,lamb0,lamb1,lamb2=find_kf(Grid,xp,yp)

    find to which triangle a point (xp,yp) belongs
    input:
        Grid - triangular grid info
        xp,yp - point on a plane
    output:
        kf - index of the the triangle
        lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

    Faster version than find_kf. Find the closest vertex first
    and then check lamb condition only for neighboring triangles.

    Vitalii Sheremet, FATE Project
    """
    # find the nearest vertex    
    kv=nearxy(Grid['x'],Grid['y'],xp,yp)
    # list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]

    # sometimes this fails
    # append the list with the nearest barycenter 
    kf=nearxy(Grid['xc'],Grid['yc'],xp,yp)
    #    kkf=np.concatenate((kfv,np.array([kf])))
    # and the triangles surrounding the nearest barycenter    
    kff=Grid['kff'][:,kf]
    kkf=np.concatenate((kfv,np.array([kf]),kff))

    # coordinates of the vertices
    kvf=Grid['kvf'][:,kkf]
    x=Grid['x'][kvf];y=Grid['y'][kvf]  
    # calculate baricentric trilinear coordinates
    A012=((x[1,:]-x[0,:])*(y[2,:]-y[0,:])-(x[2,:]-x[0,:])*(y[1,:]-y[0,:])) 
    # A012 is twice the area of the whole triangle,
    # or the determinant of the linear system above.
    # When xc,yc is the baricenter, the three terms in the sum are equal.
    # Note the cyclic permutation of the indices
    lamb0=((x[1,:]-xp)*(y[2,:]-yp)-(x[2,:]-xp)*(y[1,:]-yp))/A012
    lamb1=((x[2,:]-xp)*(y[0,:]-yp)-(x[0,:]-xp)*(y[2,:]-yp))/A012
    lamb2=((x[0,:]-xp)*(y[1,:]-yp)-(x[1,:]-xp)*(y[0,:]-yp))/A012
    #    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    #    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
    #    kf,=np.where((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    # kf is an index in the short list of triangles surrounding the vertex
    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
    # select only the first entry, the same triangle may enter twice
    # since we appended the closest barycenter triangle
    kf=kf[0]
    # return the index in the full grid     
    return kkf[kf],lamb0[kf],lamb1[kf],lamb2[kf]
    
    
def polygonal_barycentric_coordinates_old(xp,yp,xv,yv):
    """
    Calculate generalized barycentric coordinates within an N-sided polygon.

    w=polygonal_barycentric_coordinates(xp,yp,xv,yv)
    
    xp,yp - a point within an N-sided polygon
    xv,yv - vertices of the N-sided polygon, length N
    w     - polygonal baricentric coordinates, length N,
            normalized w.sum()=1
   
    Used for function interpolation:
    fp=(fv*w).sum()
    where fv - function values at vertices,
    fp the interpolated function at the point (xp,yp)
    
    Vitalii Sheremet, FATE Project    
    """
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
    jb=(j-1)%N # previous vertex in the sequence
    # area of the chord triangle j-1,j,j+1
    Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T) 
    # area of triangle p,j,j+1
    Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,np.array([xv[ja]-xp,yv[ja]-yp]).T)  

    # In FVCOM A is O(1.e7 m2) .prod() may result in inf
    # to avoid this scale A
    AScale=max(abs(Aj))
    Aj=Aj/AScale
    Ajab=Ajab/AScale
    
    w=xv*0.
    j2=np.arange(N-2)
    
    for j in range(N):
        # (j2+j+1)%N - list of triangles except the two adjacent to the edge pj
        # For hexagon N=6 j2=0,1,2,3; if j=3  (j2+j+1)%N=4,5,0,1
        w[j]=Ajab[j]*Aj[(j2+j+1)%N].prod()
        # timing [s] per step:  1.1976 1.478
        # timing [s] per step:  1.2048 1.4508 
        
    
    #    w=np.array([Ajab[j]*Aj[(j2+j+1)%N].prod() for j in range(N)])
    # timing [s] per step:  1.2192 1.4572
    # list comprehension does not affect speed

    # normalize w so that sum(w)=1       
    w=w/w.sum() 
       
    return w,Aj

def polygonal_barycentric_coordinates(xp,yp,xv,yv):
    """
    Calculate generalized barycentric coordinates within an N-sided polygon.

    w=polygonal_barycentric_coordinates(xp,yp,xv,yv)
    
    xp,yp - a point within an N-sided polygon
    xv,yv - vertices of the N-sided polygon, length N
    w     - polygonal baricentric coordinates, length N,
            normalized w.sum()=1
   
    Used for function interpolation:
    fp=(fv*w).sum()
    where fv - function values at vertices,
    fp the interpolated function at the point (xp,yp)
    
    N=2 -> lenear interpolation
    N=1 -> fixed value w=1
    
    Vitalii Sheremet, FATE Project    
    """
    N=len(xv)
    if N>2:
        j=np.arange(N)
        ja=(j+1)%N # next vertex in the sequence 
        jb=(j-1)%N # previous vertex in the sequence
        # area of the chord triangle j-1,j,j+1
        Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T) 
        # area of triangle p,j,j+1
        Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,np.array([xv[ja]-xp,yv[ja]-yp]).T)  
    
        # In FVCOM A is O(1.e7 m2) .prod() may result in inf
        # to avoid this scale A
        AScale=max(abs(Aj))
        Aj=Aj/AScale
        Ajab=Ajab/AScale
        
        w=xv*0.
        j2=np.arange(N-2)
        
        for j in range(N):
            # (j2+j+1)%N - list of triangles except the two adjacent to the edge pj
            # For hexagon N=6 j2=0,1,2,3; if j=3  (j2+j+1)%N=4,5,0,1
            w[j]=Ajab[j]*Aj[(j2+j+1)%N].prod()
            # timing [s] per step:  1.1976 1.478
            # timing [s] per step:  1.2048 1.4508 
            
            #    w=np.array([Ajab[j]*Aj[(j2+j+1)%N].prod() for j in range(N)])
            # timing [s] per step:  1.2192 1.4572
            # list comprehension does not affect speed
        w=w/w.sum() 

    # for areas close to boundary
    elif N==2:
        w=xv*0.
        w[0]=np.dot(np.array([xv[1]-xp,yv[1]-yp]).T,np.array([xv[1]-xv[0],yv[1]-yv[0]]).T)    
        w[1]=np.dot(np.array([xp-xv[0],yp-yv[0]]).T,np.array([xv[1]-xv[0],yv[1]-yv[0]]).T)
        # normalize w so that sum(w)=1       
        w=w/w.sum()
        Aj=w*0.

    elif N==1:
        w=xv*0.+1.
        Aj=w*0.
       
    return w,Aj

    
def Veli(x,y,Grid,u,v):
    """
    Velocity interpolatin function

    ui,vi=Veli(x,y,Grid,u,v)
    
    x,y - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    """
    # 1 fastest, 
    # find nearest barycenter
    kf=nearxy(Grid['xc'],Grid['yc'],x,y)
    # but the point may be in the neighboring triangle 
    #timing [s] per step:  0.0493136494444 0.0309618651389

    # 2 slower     
    # find the triangle to which point x,y truely belongs
    #    kf,lamb0,lamb1,lamb2=find_kf(Grid,x,y)
    # by means of calculating baricentric coordinates for all triangles in the grid
    #timing [s] per step:  0.482606426944 0.148569285694

    # 3 fasterthan 2
    # find the closest vertex and closest barycenter
    # and calculate barycentric coordinates 
    # in the small neighborhood of those points
    #    kf,lamb0,lamb1,lamb2=find_kf2(Grid,x,y)
    #timing [s] per step:  0.0725187981944 0.0322402066667


    # nearest neighbor interpolation    
    ui=u[kf]
    vi=v[kf]
    
    return ui,vi
    
def Veli2(xp,yp,Grid,u,v):
    """
    Velocity interpolatin function

    ui,vi=Veli(x,y,Grid,u,v)
    
    xp,yp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    """
    
    # find the nearest vertex    
    kv=nearxy(Grid['x'],Grid['y'],xp,yp)
    #    print kv
    # list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
    #    print kfv
    xv=Grid['xc'][kfv];yv=Grid['yc'][kfv]
    w=polygonal_barycentric_coordinates(xp,yp,xv,yv)
    #    print w

    # interpolation within polygon, w - normalized weights: w.sum()=1.    
    ui=(u[kfv]*w).sum()
    vi=(v[kfv]*w).sum()
        
    return ui,vi 

def VelInterp_lonlat2(lonp,latp,Grid,u,v):
    """
    Velocity interpolating function

    urci,vi=VelInterp_lonlat(lonp,latp,Grid,u,v)
    ui,vi,urci=VelInterp_lonlat2(lonp,latp,Grid,u,v)
    
    lonp,latp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    urci - interpolated u/cos(lat)
    vi   - interpolated v
    The Lame coefficient cos(lat) of the spherical coordinate system
    is needed for RungeKutta4_lonlat: dlon = u/cos(lat)*tau, dlat = vi*tau

    
    """
    
    # find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
    #    print kv
    # list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
    #    print kfv
    # coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
    lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv] 
    w,Aj=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
    # baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check! 

    if Aj.sum()==0.:
        w=w*0.
    else:    
        # Check whether any Aj are negative, which would mean that a point is outside the polygon.
        # Otherwise, the polygonal interpolation will not be continous.
        # This check is not needed if the triangular mesh and its dual polygonal mesh
        # are Delaunay - Voronoi. 
        
        # normalize subareas by the total area 
        # because the area sign depends on the mesh orientation.    
        Aj=Aj/Aj.sum()
        if np.any(Aj<0.): #np.argwhere(Aj<0).flatten().size>0: 
            # if point is outside the polygon try neighboring polygons
            #print (kv,kfv,Aj)
            for kv1 in Grid['kvv'][0:Grid['nvv'][kv],kv]:
                kfv1=Grid['kfv'][0:Grid['nfv'][kv1],kv1]
                lonv1=Grid['lonc'][kfv1];latv1=Grid['latc'][kfv1] 
                w1,Aj1=polygonal_barycentric_coordinates(lonp,latp,lonv1,latv1)
                Aj1=Aj1/Aj1.sum()
                if np.any(np.isinf(Aj1)): print ('kv1',kv1)
                if np.all(Aj1>=0.): #np.argwhere(Aj1<0).flatten().size==0:
                    w=w1*1.;kfv=kfv1*1;kv=kv1*1;Aj=Aj1*1.
                    #print (kv,kfv,Aj)
    
        # Now there should be no negative w
        # unless the point is outside the triangular mesh
        if np.any(w<0.): #np.argwhere(w<0).flatten().size>0:
            #print (kv,kfv,w)
            
            # set w=0 -> velocity=0 for points outside 
            w=w*0.        

    # interpolation within polygon, w - normalized weights: w.sum()=1.    
    # use precalculated Lame coefficients for the spherical coordinates
    # coslatc[kfv] at the polygon vertices
    # essentially interpolate u/cos(latitude)
    # this is needed for RungeKutta_lonlat: dlon = u/cos(lat)*tau, dlat = vi*tau

    # In this version the resulting interpolated field is continuous, C0.
    cv=Grid['coslatc'][kfv]    
    urci=(u[kfv]/cv*w).sum()
    vi=(v[kfv]*w).sum()
    ui=urci*Grid['coslat'][kv]    
    return ui,vi,urci

    
def ingom3(lonp,latp,Grid):
    """
    check if point is inside GOM3 mesh

    i=ingom3(lonp,latp,Grid)
    
    lonp,latp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid

    i - boolean, True if lonp,latp inside GOM3, False otherwise
    
    """

    # find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
    #    print kv
    # list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
    #    print kfv
    # coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
    lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv] 
    w,Aj=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
    # baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check! 

    # Check whether any Aj are negative, which would mean that a point is outside the polygon.
    # Otherwise, the polygonal interpolation will not be continous.
    # This check is not needed if the triangular mesh and its dual polygonal mesh
    # are Delaunay - Voronoi. 

    # normalize subareas by the total area 
    # because the area sign depends on the mesh orientation.    
    Aj=Aj/Aj.sum()
    if np.argwhere(Aj<0).flatten().size>0:
        # if point is outside the polygon try neighboring polygons
        #        print kv,kfv,Aj
        for kv1 in Grid['kvv'][0:Grid['nvv'][kv],kv]:
            kfv1=Grid['kfv'][0:Grid['nfv'][kv1],kv1]
            lonv1=Grid['lonc'][kfv1];latv1=Grid['latc'][kfv1] 
            w1,Aj1=polygonal_barycentric_coordinates(lonp,latp,lonv1,latv1)
            Aj1=Aj1/Aj1.sum()
            if np.argwhere(Aj1<0).flatten().size==0:
                w=w1;kfv=kfv1;kv=kv1;Aj=Aj1
                #                print kv,kfv,Aj

    # Now there should be no negative w
    # unless the point is outside the triangular mesh
    i=(w>=0.).all()
        
    return i
  
def inconvexpolygon(xp,yp,xv,yv):
    """
    check if point is inside a convex polygon

    i=inconvexpolygon(xp,yp,xv,yv)
    
    xp,yp - arrays of points to be tested
    xv,yv - vertices of the convex polygon

    i - boolean, True if xp,yp inside the polygon, False otherwise
    
    """    
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
    #    jb=(j-1)%N # previous vertex in the sequence
    
    NP=len(xp)
    i=np.zeros(NP,dtype=bool)
    for k in range(NP):
        # area of triangle p,j,j+1
        Aj=np.cross(np.array([xv[j]-xp[k],yv[j]-yp[k]]).T,np.array([xv[ja]-xp[k],yv[ja]-yp[k]]).T) 
        # if a point is inside the convect polygon all these Areas should be positive 
        # (assuming the area of polygon is positive, counterclockwise contour)
        Aj /= Aj.sum()
        # Now there should be no negative Aj
        # unless the point is outside the triangular mesh
        i[k]=(Aj>0.).all()
        
    return i

def inpolygon(xp,yp,xv,yv):
    """
    check if point is inside a polygon

    i=inconvexpolygon(xp,yp,xv,yv)
    
    xp,yp - arrays of points to be tested
    xv,yv - vertices of the convex polygon

    i - boolean, True if xp,yp inside the polygon, False otherwise
    
    """    
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
    #    jb=(j-1)%N # previous vertex in the sequence
    
    NP=len(xp)
    i=np.zeros(NP,dtype=bool)
    for k in range(NP):
        # area of triangle p,j,j+1
        Aj=np.cross(np.array([xv[j]-xp[k],yv[j]-yp[k]]).T,np.array([xv[ja]-xp[k],yv[ja]-yp[k]]).T) 
        # if a point is inside the convect polygon all these Areas should be positive 
        # (assuming the area of polygon is positive, counterclockwise contour)
        Aj /= Aj.sum()
        # Now there should be no negative Aj
        # unless the point is outside the triangular mesh
        i[k]=(Aj>0.).all()
        
    return i
  
###############  MAIN CODE #####################

Grid=np.load('Grid.npy',allow_pickle=True).item() # JiM added "allow_pickle=True"
NV=Grid['lon'].shape[0]
NF=Grid['lonc'].shape[0]
kvv=Grid['kvv'];nvv=Grid['nvv'];
kfv=Grid['kfv'];nfv=Grid['nfv'];



# JiM wants to plot a hourly field from say April 7th, 2013

if case[0:4]=='mean':
    ua=np.load(FNU,mmap_mode='r')# reading Vitalii's mean field
    va=np.load(FNV,mmap_mode='r')
    u=ua*1.
    v=va*1.
else:
    nc = netCDF4.Dataset(url).variables
    time_var = nc['time']
    itime = netCDF4.date2index(dtime,time_var,select='nearest')
    u = nc['u'][itime,0,:]# surface fields
    v = nc['v'][itime,0,:]




#FIGURE 1 - quiver at FVCOM nodes
kfGB=np.argwhere((Grid['lonc']>LON1)&(Grid['lonc']<LON2)&(Grid['latc']>LAT1)&(Grid['latc']<LAT2)).flatten() # Georges Bank
i=kfGB*1
fig=plt.figure() 
#subplot(111,aspect=(1.0/cos(mean(lat)*pi/180.0)))
plt.subplot(111,aspect=1.3)
plt.quiver(Grid['lonc'][i], Grid['latc'][i], u[i], v[i],color='k')
plt.axis([LON1,LON2,LAT1,LAT2])
# add coastline
coastfilename='c:/users/james.manning/Downloads/basemaps/capecod_coastline_detail.csv'
df=pd.read_csv(coastfilename)
plt.plot(df.lon,df.lat,'k.',markersize=1)
plt.title(FNU)
plt.show()
fig.savefig(case+'_'+flow+'_'+area+'_quivers.png')

#FIGURE 2 - streamplot 
lonss,latss=np.meshgrid(lons,lats)
sp=np.vstack((lonss.flatten(),latss.flatten())).T
uii=lonii*0.;urcii=lonii*0.;vii=lonii*0.
NX,NY=lonii.shape
for i in range(NX):
    for j in range(NY):
        uii[i,j],vii[i,j],urcii[i,j]=VelInterp_lonlat2(lonii[i,j],latii[i,j],Grid,u,v)
fig=plt.figure()
A=1./np.cos(41*np.pi/180.)
plt.tricontour(Grid['lon'],Grid['lat'],Grid['h'],[30.,70.,100.,200.],colors='k')  
strm=plt.streamplot(lonii,latii,urcii,vii,density=4,color=np.sqrt(uii*uii+vii*vii),cmap='GnBu')
fig.colorbar(strm.lines)
coastfilename='c:/users/james.manning/Downloads/basemaps/capecod_coastline_detail.csv'
df=pd.read_csv(coastfilename)
plt.plot(df.lon,df.lat,'k.',markersize=1)
plt.axis([LON1,LON2,LAT1,LAT2])
ax=plt.gca()
ax.set_aspect(A)
plt.title(case+'_'+flow+'_'+area+'_streamlines')
plt.show()
fig.savefig(case+'_'+flow+'_'+area+'_streamlines.png')


