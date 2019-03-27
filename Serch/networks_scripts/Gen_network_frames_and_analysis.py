#!/usr/bin/env python

# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
import sys
sys.path.append("/home/serch/pdbmani/Serch/math_tricks/")
sys.path.append("/home/serch/pdbmani/Serch/")
import math_vect_tools as mymath
# herramientas para leer pdbs
import read_pdb_tools as rpt
# funciones de click generadas en pandas
import funciones_CLICK as fc
# cuenta tiempo de ejecucion
import datetime
time_all = datetime.datetime.now()
# networks
import networkx as nx
# filtro distancia minima
# from scipy.spatial.distance import euclidean
# por si no jala
import os
os.chdir('/home/serch/pdbmani/Serch')
# multiprocessing
# import multiprocessing
# from functools import partial

# lectura de archivo
file1 = '/home/serch/pdbmani/Serch/pdbs/trj_0.pdb'  # sys.argv[1]

# se define la estructura
trj1 = rpt.Trajectory(file1)

# se lee el pdb y se agrega al objeto
trj1.ReadTraj("%s" % file1)

trj11 = trj1.frames[:3]

# set center_mass
for pdbstruct in trj11:
    pdbstruct.set_center_mass()



def get_df_distancias(ref):
    """Funcion para obtener los enlaces de distancias de cada residuo
    Dudas en codigo pueden revisar fc.distancia_entre_atomos en ese se basa
    esta funcion, la diferencia es que se crea con el objeto residuo"""
    # se generan listas con coordenadas y numero de atomo

    # calcula distancia y regresa dataframe
    enlaces = []
    for res1 in ref[1:-1]:
        for res2 in ref[1:-1]:
            if res2.resi >= res1.resi:
                if mymath.distance(res2.GetAtom('CA').coord, res1.GetAtom('CA').coord) < 10:
                    enlaces.append([res1.resi, res2.resi])

    # se genera la matriz de adyacencias para la red
    return enlaces

pdbs = [pdb.GetResChain(chain='') for pdb in trj11]

enlaces = [get_df_distancias(residues) for residues in pdbs]
# aqui se filtran los enlaces




redes = [nx.Graph(net) for net in enlaces]

net1 = redes[0]
net2 = redes[1]
net3 = redes[2]

print(nx.info(net1))
print(nx.info(net2))
print(nx.info(net3))
