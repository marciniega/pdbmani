#!/usr/bin/env python
# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
import sys
# herramientas para leer pdbs
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
file1 = '1xxa.pdb' # sys.argv[1]
file2 = '1tig.pdb' # sys.argv[2]
# outfile = open('hh_%s.txt'%infile.split('.')[0],'w')
#se define la estructura
ref1 = rpt.PdbStruct("first")
ref2 = rpt.PdbStruct("second")

#se lee el pdb y se agrega al objeto
ref1.AddPdbData("%s"%file1)
ref2.AddPdbData("%s"%file2)

#se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
ref1 = ref1.GetResChain()
ref2 = ref2.GetResChain()

def get_df_distancias(ref):
    """Funcion para obtener el dataframe de distancias de cada proteina"""
    # se generan listas con coordenadas y numero de atomo
    coord = []
    index = []
    apilacoord = coord.append
    apilaindex = index.append
    for res in ref:
        apilacoord(res.GetAtom('CA').coord)
        apilaindex(res.GetAtom('CA').atom_number)

    # calcula distancia y regresa dataframe
    distancias = []
    # se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
    for v in coord:
        distancia_un_atomo = []
        for av in coord:
            distancia = pdist(np.array([v, av]), metric='euclidean').item()
            distancia_un_atomo.append(distancia)
        distancias.append(distancia_un_atomo)

    # se genera la matriz de adyacencias para la red
    df_da = pd.DataFrame(index=index, columns=index, data=distancias)
    return(df_da,index)

df_da1,index1 = get_df_distancias(ref1)
df_da2,index2 = get_df_distancias(ref2)

# se generan cliques, tte devuleve dataframe con cliques de 3 y la lista de cliques sin partir
df_lc1, cliques1 = fc.gen_3_cliques(df_da1, dth = 10, k=3)
print('**'*50)
df_lc2, cliques2 = fc.gen_3_cliques(df_da2, dth = 10, k=3)
print('**'*50)
def get_df_ca(ref):
    """Genera dataframe con la informacion necesaria para las siguientes funciones
    FALTA DOCUMENTAR ESTA COSA!!!!"""
    #crear df_ca
    atom_number = []
    atom_name = []
    residue_name = []
    residue_number = []
    coord = []
    for res in ref:
        for atom in res.atoms:
            atom_number.append(atom.atom_number)
            atom_name.append(atom.name)
            residue_name.append(res.resn)
            residue_number.append(res.resi)
            coord.append(atom.coord)

    df_ca = pd.DataFrame(columns=['atom_number', 'atom_name', 'residue_name',
                                   'residue_number', 'vector'])
    df_ca.atom_number = atom_number
    df_ca.atom_name = atom_name
    df_ca.residue_name = residue_name
    df_ca.residue_number = residue_number
    df_ca.vector = coord

    return(df_ca)
# CREAR DF_CA #
df_ca1 =  get_df_ca(ref1)
df_ca2 =  get_df_ca(ref2)

# se obtiene la estructura secundaria utilizando dssp
ss1 = fc.mini_dssp(file1, index1)
print('**'*50)
ss2 = fc.mini_dssp(file2, index2)

# se le pega la estructura secundaria al dataframe de los cliques
df_lc1 = fc.get_SS(ss1,df_lc1)
df_lc2 = fc.get_SS(ss2,df_lc2)
print('**'*50)
#get coords of cliques
df_lc1 = fc.get_coord_clique(df_ca1, df_lc1)
df_lc2 = fc.get_coord_clique(df_ca2, df_lc2)

#baricentro clique
df_lc1 = fc.baricenter_clique(df_lc1)
df_lc2 = fc.baricenter_clique(df_lc2)

#vectores gorro
df_lc1 = fc.center_vectors(df_lc1)
df_lc2 = fc.center_vectors(df_lc2)

prueba1 = df_lc1.values
prueba2 = df_lc2.values

for i in range(df_lc2.shape[0]):
    fc.calculate_rmsd_rot_trans(10, i)




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