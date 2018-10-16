#!/usr/bin/env python
# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creagas
import sys
# herramientas para ller pdbs
import read_pdb_tools as rpt
# calculo de distancia
from scipy.spatial.distance import pdist, squareform
# libreria de tablas
import pandas as pd
# funciones de click generadas en pandas
import funciones_CLICK as fc

# sys.path.append("../math_tricks/")
# import math_vect_tools as mymath

# assert( len(sys.argv) > 1)
#lectura de archivo
infile = '1xxa.pdb' #sys.argv[1]
# outfile = open('hh_%s.txt'%infile.split('.')[0],'w')

#se define la estructura
ref1 = rpt.PdbStruct("first")

#se lee el pdb y se agrega al objeto
ref1.AddPdbData("%s"%infile)

#se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
ref1 = ref1.GetResChain()

# se generan listas con coordenadas y numero de atomo
coord1 = []
index = []
apilacoord = coord1.append
apilaindex = index.append
for res in ref1:
    apilacoord(res.GetAtom('CA').coord)
    apilaindex(res.GetAtom('CA').atom_number)

# calcula distancia y regresa dataframe
distancias = []
# se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
for v in coord1:
    distancia_un_atomo = []
    for av in coord1:
        distancia = pdist(np.array([v, av]), metric='euclidean').item()
        distancia_un_atomo.append(distancia)
    distancias.append(distancia_un_atomo)

# se genera la matriz de adyacencias para la red
df_da = pd.DataFrame(index=index, columns=index, data=distancias)
# se generan cliques, tte devuleve dataframe con cliques de 3 y la lista de cliques sin partir
df_lc1,cliques1 = fc.gen_3_cliques(df_da, dth = 10, k=3)
# se obtiene la estructura secundaria utilizando dssp
ss1 = fc.mini_dssp('1xxa.pdb', index)
# se le pega la estructura secundaria al dataframe de los cliques
df_lc1 = fc.get_SS(ss1,df_lc1)








# # while ref1.current < ref1.seqlength:
# # res = ref1.next()
# # if res.chain == "A":
# #     print(res)
#     #net = pdbnet.Network("first")
#     #net.set_nodes_on_frame(ref, mute=True)
#     #net.set_arists_on_nodes()
#     #line = "# frame: %-4s  Nodes: %-6s  Edges: %-10s"%(trj.current ,net.count_nodes() ,net.count_arists())
#     #print line
#     #outfile.write('%s\n# Edge   Distance[A]\n'%line)
#     #net.write_arist_short(outfile, trj.current, short=True)

