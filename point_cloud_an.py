#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:24:39 2019

@author: giacomo
"""

import open3d as o3d
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import animation

# read coordinates point cloud file
def readCoord():
    coord_0 = genfromtxt('t1.xyz', delimiter=' ')
    coord_0 = np.delete(coord_0, np.s_[3:], 1)

    coord_1=genfromtxt('t3.xyz', delimiter=' ')
    coord_1 = np.delete(coord_1, np.s_[3:], 1)
    
    return(coord_0,coord_1)

# create point cloud 
def createPointCloud(coord_0,coord_1):

    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(coord_0)
    downpcd_0 = pcd_0.voxel_down_sample(voxel_size=0.26)
    downpcd_0.estimate_normals()
    norm=np.asarray(downpcd_0.normals)
    #print(downpcd)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(coord_1)
    downpcd_1 = pcd_1.voxel_down_sample(voxel_size=0.26)
    #print(downpcd_1)

    coord_0_down=np.asarray(downpcd_0.points)
    coord_0_down=coord_0_down.round(decimals=3)

    coord_1_down=np.asarray(downpcd_1.points)
    coord_1_down=coord_1_down.round(decimals=3)
    
    return(coord_0_down,coord_1_down,norm)
    
# plot points as 3d scatter plot
def plotPoints(coord_0_down,coord_1_down):
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coord_0_down[:,0], coord_0_down[:,1], coord_0_down[:,2])
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coord_1_down[:,0], coord_1_down[:,1], coord_1_down[:,2])
    plt.show()

# calculate displacements array
def calcDisp(coord_0_down,coord_1_down,norm):
    
    l_disp=[]
    magnitude=[]

    for j in range(0,len(coord_0_down)):
        row_0=coord_0_down[j]
        dist=[]
        for row_1 in coord_1_down:
            dist.append(((row_1[0]-row_0[0])**2+(row_1[1]-row_0[1])**2+(row_1[2]-row_0[2])**2)**(0.5))
    
        min_dist=min(dist)
        i=dist.index(min_dist)
        u=coord_1_down[i][0]-row_0[0]
        v=coord_1_down[i][1]-row_0[1]
        w=coord_1_down[i][2]-row_0[2]
        alpha=(u*norm[i][0]+v*norm[i][1]+w*norm[i][2])/(norm[i][0]**2+norm[i][1]**2+norm[i][2]**2)
        l_disp.append([alpha*norm[j][0],alpha*norm[j][1],alpha*norm[j][2]])
        magnitude_i=math.sqrt(u**2+v**2+w**2)
        magnitude.append(magnitude_i)
 
    disp=np.array(l_disp)
    
    return(disp,magnitude)

# find outliers from calculated displacements   
def outliersRemoval(coord_0_down,disp,magnitude):
    #plt.hist(magnitude, bins=200)
    magnitude_std = np.std(magnitude) # standard deviation
    magnitude_mean=np.mean(magnitude) # mean
    anomaly_cut_off = magnitude_std * 1
    lower_limit  = magnitude_mean - anomaly_cut_off 
    upper_limit = magnitude_mean + anomaly_cut_off
    
    anomalies = []
    anomalies_index=[]
    for i in range(0,len(magnitude)):
        if magnitude[i] > upper_limit or magnitude[i] < lower_limit:
            anomalies.append(magnitude[i])
            anomalies_index.append(i)
    
    magnitude_clean = [i for j, i in enumerate(magnitude) if j not in anomalies_index]
    disp_clean = np.delete(disp, anomalies_index, axis=0)
    coord_0_down_clean=np.delete(coord_0_down, anomalies_index, axis=0)
    
    return(coord_0_down_clean,disp_clean,magnitude_clean)
    
def plotNorm(coord_0_down,norm):
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.quiver(coord_0_down[:,0],coord_0_down[:,1], coord_0_down[:,2], norm[:,0],
              norm[:,1],norm[:,2], length=0.5, normalize=True,linewidth=0.5)
    plt.show()

def plotDisp(coord_0_down,disp,magnitude):
    
    fig = plt.figure()
    ax = Axes3D(fig)
    coord_0_down_clean=outliersRemoval(coord_0_down,disp,magnitude)[0]
    disp_clean=outliersRemoval(coord_0_down,disp,magnitude)[1]
    magnitude_clean =outliersRemoval(coord_0_down,disp,magnitude)[2]

    c = magnitude_clean # Color by magnitude
    c = np.concatenate((c, np.repeat(c, 2))) # Repeat for each body line and two head lines
    c = plt.cm.hsv(c) # Colormap

    ax.quiver(coord_0_down_clean[:,0],coord_0_down_clean[:,1],
              coord_0_down_clean[:,2], disp_clean[:,0], disp_clean[:,1],
              disp_clean[:,2],color=c, length=20, normalize=False,
              linewidth=0.5)
    
    plt.show()

'''
def rotate(angle):
    ax.view_init(azim=angle)
    
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,30),interval=1000)
rot_animation.save('rotation.gif', dpi=200, writer='imagemagick')
plt.show()
'''

def main():
    coord_0=readCoord()[0]
    coord_1=readCoord()[1]
    coord_0_down=createPointCloud(coord_0,coord_1)[0]
    coord_1_down=createPointCloud(coord_0,coord_1)[1]
    norm=createPointCloud(coord_0,coord_1)[2]
    disp=calcDisp(coord_0_down,coord_1_down,norm)[0]
    magnitude=calcDisp(coord_0_down,coord_1_down,norm)[1]
    plotDisp(coord_0_down,disp,magnitude)
    plotNorm(coord_0_down,norm)
    
# call the main function
if __name__ == "__main__":
    main()
