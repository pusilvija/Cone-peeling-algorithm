"""
Cone-peeling algorithm for getting the essential pairwise distances.
"""
#Imports
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import time
from sklearn.metrics import pairwise_distances
import multiprocessing
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import *
from sklearn.metrics import mean_squared_error


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

def get_structure(protein_name):
    """
    Returns the protein struture given the protein name.
    """
    parser = PDBParser()
    structure = parser.get_structure(protein_name, protein_name + '.pdb')

    return structure

## PP length ##
n = 42
## ##

## Creating a list of n COORDINATES ##
native_coords = [] 
for i in range(n):
    native_coords.append(get_CA_coords('42aa', i))
native_coords_t = torch.tensor(native_coords)
native_coords = np.array(native_coords)
## ##

### CONE-PEELING ALGORITHM IMPLEMENTATION ###

### 1. Sort nodes in descending order by degree ###

def sort_nodes_desc(coords):
    """
    Input: coords - numpy array of 3D coordinates
    Output:
        [0] - sorted coordinates
        [1] - sorted nodes
    """
    # Create a list (pw_distances) of each node (i) pw distances, delete distances that are > 9 or < 1 Angstrem
    pw_distances = []
    for i in coords:
        coord = [i]
        pwds = pairwise_distances(coord, native_coords, metric='euclidean') # a list of all pairwise distances of coordinate i
        selected_pwds = []
        for i in pwds[0]: # save pairwise distance (i) if it is 1<i<9
            if 1 < i < 9:
                selected_pwds.append(i)
        pw_distances.append(selected_pwds)    
        
    # Create a list of nodes orders
    nodes_orders = []
    for i in pw_distances:
        i_order = len(i)
        nodes_orders.append(i_order)

    # Create a distances_dict: {node_nr: [pw_distances]}
    node_nr = list(range(len(native_coords)))                             
    distances_dict = dict(zip(node_nr, nodes_orders))
    
    # Sort nodes in descending order by degree
    sorted_n_d = {k: v for k, v in sorted(distances_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_n = list(sorted_n_d.keys())
    
    # Sort coordinates
    sorted_coords = []
    for i in range(len(native_coords)):
        tmp = native_coords[sorted_n[i]]                             
        sorted_coords.append(tmp)
    sorted_coords = np.array(sorted_coords)

    return sorted_coords, sorted_n


### 2. For every node u, get egdes incident on u ###

def get_edges_indices(sorted_co, sorted_no, native_co):
    """
    Input: sorted_co - sorted coordinates from (1)
           sorted_no - sorted nodes by degree from (1)
           native_co - native coordinates
    Output: all_edges_nr - every nodes edges
    """
    # dict where for each node we have a list of PW distance and the node that it goes to           
    nodes_edges = []
    # make a list of edges for every node according to sorted_n
    all_edges_nr = []
    for i in range(len(sorted_co)):
        edges = []
        coord = [sorted_co[i]] # pirma coord is sorted
        pwds = pairwise_distances(coord, np.array(native_co), metric='euclidean')
        for j in range(len(native_co)):
            if 1 < pwds[0][j] < 9:
                tmp = (pwds[0][j], (sorted_no[i], j))
                tmp2 = (sorted_no[i], j)
                all_edges_nr.append(tmp2)
                edges.append(tmp)
        nodes_edges.append(edges)
             
    return all_edges_nr # to also see distance return nodes_edges
    
     
### 3. CNb size function ###

def get_CNb_size(edge_ij, overall_edges):
    """
    Input: edge_ij - edge in (i, j) format
           overall_edges - all edges
    Output: [0] CNb - common neighbourhood nodes
            [1] CNb_size - size of the common neighborhood
    """
    # set edge nodes
    node_i = edge_ij[0]
    node_j = edge_ij[1]
    
    # create a lists of nodes connected to node_i and to node_j
    nodes_to_i = []
    nodes_to_j = []
    for edge in overall_edges:
        if edge[0] == node_i:
          nodes_to_i.append(edge[1])
        elif edge[0] == node_j:
          nodes_to_j.append(edge[1])

    # find common elements from n_ei and n_ej
    CNb = list(set(nodes_to_i).intersection(nodes_to_j))
    
    # find CNb size
    CNb_size = len(CNb)

    return CNb, CNb_size


### 4. Find CNb sizes for all edges and delete when the sequence-range of either of the neighbour is <= 3 ###

def get_CNbs_edges(all_edges_nr):
    """
    Returns CNb edges for all edges.
    Deletes the edge when the sequence-range of either of the neighbour is <= 3.
    """
    all_cnbsizes_unsorted = []
    for edge in all_edges_nr:
        cnbsize = get_CNb_size(edge, all_edges_nr)
        if abs(edge[0]-edge[1]) > 3:
          tmp = [edge, cnbsize[1]]
          all_cnbsizes_unsorted.append(tmp)
        
    all_cnbsizes = sorted(all_cnbsizes_unsorted, key=lambda x: x[1], reverse=True)   

    # Get CNB edges list
    CNBedges_list = []
    for i in all_cnbsizes:
        CNBedges_list.append(i[0])

    # Check for the same reverse edges
    CNBedges_list_final = []
    for edge in CNBedges_list:
      if edge not in CNBedges_list_final:
        if edge[::-1] not in CNBedges_list_final:
          CNBedges_list_final.append(edge)

    return CNBedges_list_final


### Find pairwise distances according to the given edges ###

def find_dist(edges, native_co):
    """
    Returns pairwise distances according to the given edges and native coordinates.
    """
    pw_dists = []
    for e in edges:
        dist = torch.dist(native_co[e[0]],native_co[e[1]])
        pw_dists.append(dist)
    pw_dists = torch.tensor(pw_dists)
    return pw_dists


### Get essential pairwise distances

def cone_peeling_alg(native_co):
    """
    """
    ### 1 ### sorted nodes and coordinates in descenting order by degree
    sorted_coords = sort_nodes_desc(native_co)[0]
    sorted_nodes = sort_nodes_desc(native_co)[1]
    
    ### 2 ### for every node u get edges incident on u
    all_edges_n = get_edges_indices(sorted_coords, sorted_nodes, native_co)#[0]
    #nodes_edges = get_edges_indices(sorted_coords, sorted_nodes, native_coords)[1]
    
    ### 3 ### get essential edges
    CNb_edges = get_CNbs_edges(all_edges_n)
    
    ### 4 ### distances to restrict with according edges
    edges_dist = find_dist(CNb_edges, torch.tensor(native_co)) 
    
    return CNb_edges, edges_dist

edges = torch.tensor(cone_peeling_alg(native_coords)[0])
dists = torch.tensor(cone_peeling_alg(native_coords)[1])

print(len(edges), "", len(edges)/(n*(n-1)/2), "% of all possible pairwise distances")
print("Edges nodes:", edges)
print("Distances according to the edges:" dists)

