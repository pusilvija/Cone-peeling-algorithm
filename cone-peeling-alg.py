#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:12:36 2019

@author: si
"""

import os
os.chdir('/home/si/Desktop/project3')
os.getcwd()


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


def save_M(M, f_out):
    """
    Save CA trace of M in PDB file f_out.
    """
    _ATOM = '%s%5i  %-4s%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f %4s%2s%2s\n'

    def get_ATOM_line(atom_i, name, resid, x, y, z, aa_type):
        """
        Write PDB ATOM line.
        """
        args=('ATOM  ', atom_i, name, aa_type, 'A', resid, ' ', x, y, z, 0.0, 0.0, 'X', ' ', ' ')
        s = _ATOM % args
        return s

    fp = open(f_out, 'w')
    for i in range(0, M.shape[0]):
        x, y, z = M[i]
        s = get_ATOM_line(i, 'CA', i, x, y, z, 'ALA') 
        fp.write(s)
    fp.close()

def get_samples(posterior, name):
    """
    Extracts samples from a posterior object.
    """
    marginal = posterior.marginal(sites=[name])
    marginal_tensor = marginal.support()[name]
    return marginal_tensor

def get_CA_coords(protein_name, n):
    """
    Gets coordinates of nth CA atom. 
    """
    # Get protein structure
    parser = PDBParser()
    struct = parser.get_structure(protein_name, protein_name + '.pdb')
    # Get the coordinates of CA atoms
    coords = []
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        XYZ = atom.get_coord()
                        coords.append(XYZ)
    # Get the nth atom coordinates
    nth_coord = coords[n]
    return nth_coord



## SMALL protein ### 
n=50


# Creating a list of 50 coordintes
native_coords = [] 
for i in range(n):
    native_coords.append(get_CA_coords('50aa', i))
native_coords_t = torch.tensor(native_coords)

native_coords_t


# Find first 3 coordinates of native_coords_t
def first3_coords(coords):
    first3 = torch.zeros([3,3])
    for i in range(3):
        first3[i] = coords[i]
    return np.array(first3)

M_first = first3_coords(native_coords_t)


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
    
n_dists_clear
    
# Create a list of order of nodes

order_nodes = []
for i in n_dists_clear:
    tmp = len(i)
    order_nodes.append(tmp)

order_nodes

# Create a dict_distances: {node_nr: [distances]}

node_nr = list(range(1,51))

dict_distances = dict(zip(node_nr, order_nodes))
dict_distances

# Sort nodes in descending order

sorted_n_d = {k: v for k, v in sorted(dict_distances.items(), key=lambda item: item[1], reverse=True)}
sorted_n = list(sorted_n_d.keys())
sorted_n

# Sort coordinates
sorted_coords = []
for i in range(len(native_coords)):
    tmp = native_coords[sorted_n[i]-1]
    sorted_coords.append(tmp)
    
sorted_coords

native_coords


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


nodes_edges

# to check the above:
    # distance between x and y CA
x = 40 # from sorted_n list
nr = 0                         # change this to change node to which distance is measured
y = nodes_edges[x][nr][1]       
y
    # this should be the same as
nodes_edges[x][nr][0]
    #this
torch.dist(native_coords_t[sorted_n[x]-1], native_coords_t[y])
distanc = torch.dist(native_coords_t[sorted_n[x]-1], native_coords_t[y])
print("Distance between", sorted_n[x], "and", y+1, "nodes is:", distanc.item(), "Angstrem")

sorted_n[x]-1 
native_coords_t[y]
    
    
# make a list of edges for every node according to sorted_n

all_edges_nr = []
for i in range(len(nodes_edges)): # in node
    edges_nr = []
    for edge in nodes_edges[i]:
        tmp = (sorted_n[i], edge[1])
        edges_nr.append(tmp)
    all_edges_nr.append(edges_nr)

all_edges_nr

torch.dist(native_coords_t[2-1], native_coords_t[0])


nodes_edges



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
a = get_CNb_size((11,9), all_edges_nr)
a

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
all_cnbsizes


# Get CNB edges list
CNBedges_list = []
for i in all_cnbsizes:
    CNBedges_list.append(i[0])
CNBedges_list

len(CNBedges_list)
# find  distances according to edges
dist_nr = []
for p in CNBedges_list:
    d=torch.dist(native_coords_t[p[0]-1],native_coords_t[p[1]])
    dist_nr.append(d)

len(dist_nr)



torch.dist(native_coords_t[13-1], native_coords_t[44])


CNBedges_list[0]
dist_nr[0]


def rmsd_dist_burn(burn,s_b,s_d,distances=233, target_accept_prob=0.4, noise=10):
    """
    The function runs NUTS sampler based on the specific model for sampling protein 
    structure with given pairwise distances.
    distance: number of random distances to be additionally restraint;
    burn: warm up size;
    target_accept_prob: target acceptance probability, NUTS Sampler parameter;
    
    Returns: all structure average RMSD and separate values, fixed 3 first coordinates 
    average RMSD and separate values, time that it took each iteration to run.
    """

    def model(N=50):
        plate1=numpyro.plate("aa", N-3, dim=-2)
        plate2=numpyro.plate("coord", 3, dim=-1)
        with plate1, plate2:
            M_last = numpyro.sample("M", dist.Normal(0, 20))      

        # Stack fixed and moving coordinates
        M=np.concatenate((M_first, M_last))
        
        # Make sure bond distances are around 3.8 Å       
        bonds=M[0:-1]- M[1:]        
        Bonds=(bonds[:,0]**2+bonds[:,1]**2+bonds[:,2]**2)**(1/2)
        
        sb = numpyro.sample("s_b", dist.HalfNormal(s_b))
        i=0       
        with numpyro.plate("Bonds",49):
            bond_obs = numpyro.sample("Bonds_%i" % i, dist.Normal(Bonds, sb), obs=3.8)
            i+=1    
 
        # Add a pairwise distance restraints
        for i in range(distances):
            sd = numpyro.sample("s_d%i" % i, dist.HalfNormal(s_d))
            D = (M[CNBedges_list[i][0]] - M[CNBedges_list[i][1]])
            d = (D[0]**2+D[1]**2+D[2]**2)**(1/2)
            d_obs=numpyro.sample('d%s_obs' % i, dist.Normal(d, sd), obs= numpyro.sample('noise_%i' % i, dist.Normal(dist_nr[i].item(), noise)))               
    
    # Nr samples
    S=1000
    # Nr samples burn-in
    B=burn
    
    start = time.time()
    
    # Do NUTS sampling
    nuts_kernel = NUTS(model, adapt_step_size=True, target_accept_prob=target_accept_prob)
    mcmc_sampler = MCMC(nuts_kernel,B, num_samples=S)
    
    j=onp.random.randint(10000)
    rng= random.PRNGKey(j)
    
    posterior = mcmc_sampler.run(rng)
    # Get the last sampled points
    M_last=mcmc_sampler.get_samples()
    M=np.concatenate((M_first, M_last['M'][-1]))

    # Compute running time
    end = time.time()
    #print(end)

    # or return samples for pdb file:
    return M

n=50

def model_check(M,distances=233):    
    M=torch.tensor(onp.array((M)))
    #Check that bound distance is 3.8 Å    
    bounds=[]
    for i in range(n-1):
        bound=torch.dist(M[i],M[i+1]).item()
        #print(bound)
        bounds.append(bound)
    rmsd_b=0
    for i in range(n-1):
        rmsd_b += (bounds[i]-3.8)**2
    rmsd_b=math.sqrt(rmsd_b/(n-1))
    
    rmsd_d=0
    N=0
    for i in range(n):
        for j in range(i+1,n):
            a,b=i,j
            d_M=round(torch.dist(M[a],M[b]).item(),1)
            d_n=round(torch.dist(native_coords_t[a],native_coords_t[b]).item(),1)
            #rmsd_d+=(d_M-d_n)**2
            N+=1
            if (a,b) in CNBedges_list[:distances] or (a,b) in CNBedges_list[:distances]:
                rmsd_d+=(d_M-d_n)**2
                print('Restricted distance ',str(a),'-',str(b),',difference ',(d_M-d_n),
                      ",    d_M=", d_M, "d_n=", d_n)
            else:
                pass
                #print('Not restricted distance ',str(a),',',str(b),',',(d_M-d_n))

    rmsd_d=math.sqrt(rmsd_d/(N))
    print(rmsd_b, rmsd_d)
    return rmsd_b,rmsd_d


x = rmsd_dist_burn(70, 0.1,0.1, distances = 200, noise=100)
model_check(x)

save_M(x,'x_d200_noise100.pdb')











