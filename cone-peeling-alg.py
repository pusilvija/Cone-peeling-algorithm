#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:12:36 2019

@author: si
"""


import torch
import numpy as onp
import jax.numpy as np
import jax.random as random
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from Bio.PDB import *
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import time
import multiprocessing



### IMPLEMENTING CONE-PEELING ALGORITHM ### we have: native coordinates

from sklearn.metrics import pairwise_distances

## 1. Sort nodes in descending order by degree

# Create a list of each node pw distances, delete distances that are > 9 or < 1 Angstrem

n_dists_clear = []
for i in native_coords:
    s = []
    coord = [i]*len(native_coords)
    coord_a = onp.array(coord)
    output = pairwise_distances(coord_a, onp.array(native_coords), metric='euclidean')
    tmp = []
    s.append(output)
    for i in s[0][0]:
        if 1 < i < 9:
            tmp.append(i)
    n_dists_clear.append(tmp)    
    
    
# Create a list of order of nodes

order_nodes = []
for i in n_dists_clear:
    tmp = len(i)
    order_nodes.append(tmp)


# Create a dict_distances: {node_nr: [distances]}

node_nr = list(range(1,51))
dict_distances = dict(zip(node_nr, order_nodes))

# Sort nodes in descending order

sorted_n_d = {k: v for k, v in sorted(dict_distances.items(), key=lambda item: item[1], reverse=True)}
sorted_n = list(sorted_n_d.keys())

# Sort coordinates
sorted_coords = []
for i in range(len(native_coords)):
    tmp = native_coords[sorted_n[i]-1]
    sorted_coords.append(tmp)

## 2. For every node u, get egdes incident on u

# dict where for each node we have a list of PW distance and the node that it goes to             ##### PATIKRINTI !!!!!

nodes_edges = []
for i in range(len(sorted_coords)):
    edges = []
    coord = [sorted_coords[i]]*len(native_coords)
    coord_a = onp.array(coord)
    output = pairwise_distances(coord_a, onp.array(native_coords), metric='euclidean')
    for j in range(len(output[0])):
        if 1 < output[0][j] < 9:
            tmp = (output[0][j], j)
            edges.append(tmp)
    
    
    nodes_edges.append(edges)

# to check the above:
    # distance between x and y CA
#x = 40 # from sorted_n list
#nr = 0                         # change this to change node to which distance is measured
#y = nodes_edges[x][nr][1]       
#y
    # this should be the same as
#nodes_edges[x][nr][0]
    #this
#torch.dist(native_coords_t[sorted_n[x]-1], native_coords_t[y])
#distanc = torch.dist(native_coords_t[sorted_n[x]-1], native_coords_t[y])
#print("Distance between", sorted_n[x], "and", y+1, "nodes is:", distanc.item(), "Angstrem")

#sorted_n[x]-1 
#native_coords_t[y]
    
    
# make a list of edges for every node according to sorted_n

all_edges_nr = []
for i in range(len(nodes_edges)): # in node
    edges_nr = []
    for edge in nodes_edges[i]:
        tmp = (sorted_n[i], edge[1])
        edges_nr.append(tmp)
    all_edges_nr.append(edges_nr)


#torch.dist(native_coords_t[2-1], native_coords_t[0])


## 3. where A(u) is the list of edges incident on u
    # For every edge in A(u) get Cnbsize and sort in ascending order

# CNb size function

def get_CNb_size(edge_ij, overall_edges):
    # set edge nodes
    node_i = edge_ij[0]
    node_j = edge_ij[1]
    
    # create a list of edges with node_i
    i_edges = []
    for i in overall_edges:
        if i[0][0] == node_i:
            for y in i:
                i_edges.append(y)
    n_ei = []
    for i in i_edges:
        n_ei.append(i[1])

    # create a list of edges with node_j
    j_edges = []
    for i in overall_edges:
        if i[0][0] == node_j:
            for y in i:
                j_edges.append(y)   
    n_ej = []
    for i in j_edges:
        n_ej.append(i[1])
    
    # find common elements from n_ei and n_ej
    CNb = list(set(n_ei).intersection(n_ej))
    
    # find CNb size
    CNb_size = len(CNb)

    
    #print("i edges:", i_edges, "\nj edges:", j_edges)
    return CNb, CNb_size

# example    
#a = get_CNb_size((11,9), all_edges_nr)


# find CNb sizes for all edges and delete when the sequence-range of either of the neighbour is <= 3

all_cnbsizes_unsorted = []
for specific_node in all_edges_nr:
    #cnbsize_node = []
    for edge in specific_node:
        #print(edge)
        cnbsize = get_CNb_size(edge, all_edges_nr)
        if abs(edge[0]-edge[1]) > 3:
            #print(cnbsize[1])
            tmp = [edge, cnbsize[1]]
            #print(tmp)
            all_cnbsizes_unsorted.append(tmp)
    
all_cnbsizes = sorted(all_cnbsizes_unsorted, key=lambda x: x[1], reverse=True)

# Get CNB edges list
CNBedges_list = []
for i in all_cnbsizes:
    CNBedges_list.append(i[0])

# find  distances according to edges
dist_nr = []
for p in CNBedges_list:
    d=torch.dist(native_coords_t[p[0]-1],native_coords_t[p[1]])
    dist_nr.append(d)



torch.dist(native_coords_t[13-1], native_coords_t[44])


CNBedges_list[0]
dist_nr[0]










