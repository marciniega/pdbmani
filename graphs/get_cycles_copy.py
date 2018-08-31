#!/usr/bin/env python
import sys
import numpy as np
from undirected_graph import *

def get_atom_lists(infile):
    """Three lists: 1) aromatic , 2) adjancency pairs and
       3) number of node """
    found_bounds = False
    ar1 = []
    ar2 = []
    alltyp1 = []
    alltyp2 = []
    nested_lst_ar= []
    nested_lst_all = []
    count = 0

    for line in infile:
        if '@<TRIPOS>MOLECULE' in line:
            count += 1
        if '@<TRIPOS>BOND' in line:
            found_bounds = True
            continue
        if '@<TRIPOS>SUBSTRUCTURE' in line:
            found_bounds = False
            continue
        if found_bounds:
            line = line.split('\n')[0]
            a = line.split()
            alltyp1.append(a[1])
            alltyp2.append(a[2])
            if 'ar' in a[3]:
               ar1.append(a[1])
               ar2.append(a[2])
        if count > 1:
            break

    nested_lst_ar.append(ar1)
    nested_lst_ar.append(ar2)
    ar_atms = np.swapaxes((np.array(nested_lst_ar)),0,1)
    ar_atms = ar_atms[np.argsort(ar_atms[:,0])].astype(int)
    armtic_atms = np.sort(np.unique(ar_atms))
    #print armtic_atms

    nested_lst_all.append(alltyp1)
    nested_lst_all.append(alltyp2)
    all_atms = np.swapaxes((np.array(nested_lst_all)),0,1)
    adj_list_all_atms = all_atms[np.argsort(all_atms[:,0])].astype(int)
    #print adj_list_all_atms
    num_nodes = len(np.unique(all_atms))
    return armtic_atms, adj_list_all_atms , num_nodes

def get_aros_cycles(file_name):
    input_file = open(file_name).readlines()
    aro , mat , n = get_atom_lists(input_file)
    g = Graph(n)
    g.BuildNet(mat)
    g.MarkAro(aro)
    cyc_6 = g.GetAroCycles(6)
    cyc_5 = g.GetAroCycles(5)
    return cyc_6 , cyc_5

if __name__ == '__main__':
    print get_aros_cycles(sys.argv[1])

