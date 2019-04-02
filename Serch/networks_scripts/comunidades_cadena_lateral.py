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
file1 = '/home/serch/pdbmani/Serch/pdbs/1xxa.pdb'  # sys.argv[1]
file2 = '/home/serch/pdbmani/Serch/pdbs/1tig.pdb'  # sys.argv[2]
# file1 = 'pdbs/2mhu.pdb'  # sys.argv[1]
# file2 = 'pdbs/2mrt.pdb'  # sys.argv[2]

# numero de cliques, preguntar en el software para generalizarlo INPUT...
number_elements_clique = 3

# se define la estructura
pdb1 = rpt.PdbStruct(file1)
pdb2 = rpt.PdbStruct(file2)

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)

#filtro SS
pdb1.Set_SS()
pdb2.Set_SS()

# filtro dihedral
pdb1.SetDiheMain()
pdb2.SetDiheMain()

# set center_mass
pdb1.set_center_mass()
pdb2.set_center_mass()

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

# creando tabla de estructura secundaria para filtro de SS
ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)


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
                if mymath.distance(res2.center_mass, res1.center_mass) < 10:
                    enlaces.append([res1.resi, res2.resi])

    # se genera la matriz de adyacencias para la red
    return enlaces


enlaces1 = (get_df_distancias(pdb11))
enlaces2 = (get_df_distancias(pdb22))

red1 = (nx.Graph(enlaces1))
red2 = (nx.Graph(enlaces2))

cliques_1, cliques_max_1 = fc.gen_cliques(red1, k=7)
cliques_2, cliques_max_2 = fc.gen_cliques(red2, k=7)
