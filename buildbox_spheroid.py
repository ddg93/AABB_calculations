#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2022
@author: davide
"""

import numpy as np
import os

def align_z_to_vector(n):
    """
    Construct rotation matrix aligning z-axis to vector n (unit vector).
    """
    z_axis = np.array([0, 0, 1.0])
    v = np.cross(z_axis, n)
    c = np.dot(z_axis, n)
    if np.allclose(v, 0):  # n is parallel to z
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return R


def euler_xyz_to_rotation_matrix(yaw, pitch, roll):
    """
    Constructs a rotation matrix from XYZ Euler angles [rads].
    Order: R = Rz * Ry * Rx
    """
    #terms
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    #build matrices
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx

def compute_aabb_extents(a, b, c, R):
    #spheroid representation in the particle reference
    Q_body = np.diag([a**2, b**2, c**2])
    #spheroid representation in the world reference
    Q_world = R @ Q_body @ R.T
    #get the extents of the projections
    extents = []
    for u in np.eye(3):
        half_extent = np.sqrt(u @ Q_world @ u)
        extents.append(2 * half_extent)
    return extents
# =============================================================================
# Main script
# =============================================================================
home = os.getcwd()
destination = ''
### set the radius
r = 0.5     
### set the aspect ratio
ar = 0.56   
#ellipsoid semi-axes squared
a, b, c = r, r, (r*ar)
###set the number of considered particle orientations
N = 20000
###generate the particle orientation vectors, ensure norm-2 equal to 1 (redundant)
phi = np.random.uniform(0.0, np.pi/2.0, N)
costheta = np.random.uniform(0, 1, N)
theta = np.arccos(costheta)
x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)
n_vectors = np.column_stack((x,y,z))
for i in range(3):
    n_vectors[:,i] = n_vectors[:,i] / np.linalg.norm(n_vectors,axis=1)
###Loop
results = []
for n in n_vectors:
    #Align the particle to the given n
    R = align_z_to_vector(n)
    #Compute AABB extents
    extents = compute_aabb_extents(a, b, c, R)
    #Save results
    row = list((phi[i],theta[i],0.0)) + list(extents) + list(n)
    results.append(row)
results = np.array(results)
np.savetxt(home+destination+'boxlist.txt',results)

###here you can visualize the considered orientations
if False:
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  
  # plot
  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x,y,z, s=5, alpha=0.6)
  
  # make axes equal
  ax.set_box_aspect([1, 1, 1])
  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([0, 1])  # hemisphere only
  
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  plt.show()
