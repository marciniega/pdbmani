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
#iteradores
import itertools as it

# sys.path.append("../math_tricks/")
# import math_vect_tools as mymath

# assert( len(sys.argv) > 1)
# lectura de archivo
file1 = '1xxa.pdb' # sys.argv[1]
file2 = '1tig.pdb' # sys.argv[2]
# outfile = open('hh_%s.txt'%infile.split('.')[0],'w')
# se define la estructura
pdb1 = rpt.PdbStruct("first")
pdb2 = rpt.PdbStruct("second")

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)


# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

ss1 = pdb1.Get_SS(file1)
ss2 = pdb1.Get_SS(file2)

# se crea atributo a cada residuo
for i, j in zip(pdb11, ss1.structure.values):
    setattr(i, 'structure', j)
for i, j in zip(pdb22, ss2.structure.values):
    setattr(i, 'structure', j)


def get_df_distancias(ref):
    """Funcion para obtener el dataframe de distancias de cada proteina"""
    # se generan listas con coordenadas y numero de atomo
    coord = []
    index = []
    apilacoord = coord.append
    apilaindex = index.append
    for res in ref:
        apilacoord(res.GetAtom('CA').coord)
        apilaindex(res.resi)

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
    return(df_da, index)


df_distancias1, index1 = get_df_distancias(pdb11)
df_distancias2, index2 = get_df_distancias(pdb22)

# print(index1)

# se generan cliques, tte devuleve dataframe con cliques de 3 y la lista de cliques sin partir
df_cliques1, cliques1 = fc.gen_3_cliques(df_distancias1, dth = 10, k=3)
print('**'*50)
df_cliques2, cliques2 = fc.gen_3_cliques(df_distancias2, dth = 10, k=3)
print('**'*50)


def get_df_ca(list_of_residues):
    """Genera dataframe con la informacion necesaria para las siguientes funciones
    FALTA DOCUMENTAR ESTA COSA!!!!"""
    #crear df_ca
    atom_number = []
    atom_name = []
    residue_name = []
    residue_number = []
    coord = []
    for res in list_of_residues:
        for atom in res.atoms:
            atom_number.append(atom.atom_number)
            atom_name.append(atom.name)
            residue_name.append(res.resn)
            residue_number.append(res.resi)
            coord.append(atom.coord)

    df_atoms = pd.DataFrame(columns=['atom_number', 'atom_name', 'residue_name',
                                   'residue_number', 'vector'])
    df_atoms.atom_number = atom_number
    df_atoms.atom_name = atom_name
    df_atoms.residue_name = residue_name
    df_atoms.residue_number = residue_number
    df_atoms.vector = coord

    return(df_atoms)


# CREAR DF_atomos_CA #
df_atoms1 = get_df_ca(pdb11)
df_atoms2 = get_df_ca(pdb22)

# se obtiene la estructura secundaria utilizando dssp
# ss1 = fc.mini_dssp(file1, index1)
# print('**'*50)
# ss2 = fc.mini_dssp(file2, index2)



# se le pega la estructura secundaria al dataframe de los cliques
#esto va a cambiar por que lo tiene que obtener del objeto residuo
df_cliques1 = fc.get_SS(ss1, df_cliques1)
df_cliques2 = fc.get_SS(ss2, df_cliques2)


#comparacion SSM
comp1 = df_cliques1[['ss_0','ss_1','ss_2']].values
comp2 = df_cliques2[['ss_0','ss_1','ss_2']].values


producto = it.product(df_cliques1.index.values, df_cliques2.index.values)
candidatos_ss = []
candidatosapila = candidatos_ss.append
for i, j in producto:
    score = (list(map(fc.SSM, comp1[i], comp2[j])))
    if 2 in score:
        continue
    else:
        candidatosapila((i, j))

# print(len(df_cliques1.index.values) * len(df_cliques2.index.values))
# print(len(candidatos_ss))
# print(candidatos_ss)


# exit()



#get coords of cliques
df_cliques1 = fc.get_coords_clique(df_atoms1, df_cliques1)
df_cliques2 = fc.get_coords_clique(df_atoms2, df_cliques2)

#baricentro clique
df_cliques1 = fc.baricenter_clique(df_cliques1)
df_cliques2 = fc.baricenter_clique(df_cliques2)

#vectores gorro
df_cliques1 = fc.center_vectors(df_cliques1)
df_cliques2 = fc.center_vectors(df_cliques2)

#se pasan a numpy arrays para mayor rapidez
array_df_cliques1 = df_cliques1.values
array_df_cliques2 = df_cliques2.values

#calculo del RMSD
candidatos = []
apilacandidatos = candidatos.append
calcularmsd = fc.calculate_rmsd_rot_trans

producto = it.product(df_cliques1.index.values, df_cliques2.index.values)
# print('total iteraciones:', len(list(producto)))
#
# for i,j in producto:
#     print(i, j)
import datetime

timenow = datetime.datetime.now()
for i, j in candidatos_ss:
    rmsd_final = calcularmsd(i, j, array_df_cliques1, array_df_cliques2)
    if rmsd_final <= 0.15:
        apilacandidatos([i, j])

time = datetime.datetime.now()
print(len(candidatos))


print(time - timenow)


# for i in range(df_lc1.shape[0]):
#     for j in range(df_lc2.shape[0]):
#         rmsd_final = calcularmsd(i, j, prueba1, prueba2)
#         if rmsd_final <= 0.15:
#             apilacandi([j,i])




# for j in range(df_lc1.shape[0]):
#     for i in range(df_lc2.shape[0]):
#         fc.calculate_rmsd_rot_trans(j,i,prueba1,prueba2)












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