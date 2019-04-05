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
# networks
import networkx as nx

import os
os.chdir('/home/serch/pdbmani/Serch')

# lectura de archivo
file1 = '/home/serch/Descargas/dimer_con_cofact.pdb'  # dime_out no es trayectoria es pdb

# se define la estructura
trj1 = rpt.Trajectory(file1)

# se lee el pdb y se agrega al objeto
trj1.ReadTraj("%s" % file1)

trj11 = trj1.frames

# set center_mass
for pdbstruct in trj11:
    pdbstruct.set_center_mass()

def get_df_distancias(ref):
    """Funcion para obtener los enlaces de distancias de cada residuo
    Dudas en codigo pueden revisar fc.distancia_entre_atomos en ese se basa
    esta funcion, la diferencia es que se crea con el objeto residuo"""
    # se generan listas con coordenadas y numero de atomo
    enlaces = [ [res1.resi, res2.resi] for res1 in ref[1:-1] for res2 in ref[1:-1]
               if res2.resi >= res1.resi if mymath.distance(res2.GetAtom('CA').coord, res1.GetAtom('CA').coord) < 10
              ]
    return enlaces

# elegir la cadena  o tal ves es ambas preguntar!!!!
pdbs = [pdb.GetResChain(chain='B') for pdb in trj11]

enlaces = [get_df_distancias(residues) for residues in pdbs]
# aqui se filtran los enlaces

from collections import Counter # obtener conteo de enlaces
a = Counter([str(j) for i in enlaces for j in i]).most_common()

import ast # para pasar de str a tupla o listas
red_promedio = nx.Graph([ast.literal_eval(i[0]) for i in a if (i[1] / len(enlaces)) > 0.7])

print(nx.info(red_promedio))

trj11[0].PrintPdbInfo()
