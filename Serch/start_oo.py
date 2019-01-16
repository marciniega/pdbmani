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
# iteradores
import itertools as it
# cuenta tiempo de ejecucion
import datetime

# multiprocessing
import multiprocessing
from functools import partial
# filtro distancia
from scipy.spatial import distance

timenow_bueno = datetime.datetime.now()

timenow = datetime.datetime.now()

# sys.path.append("../math_tricks/")
# import math_vect_tools as mymath

# assert( len(sys.argv) > 1)
# lectura de archivo
file1 = 'pdbs/1xxa_prueba.pdb'  # sys.argv[1]
file2 = 'pdbs/1xxa.pdb'  # sys.argv[2]

# numero de cliques, preguntar en el software para generalizarlo...
number_elements_clique = 3

# outfile = open('hh_%s.txt'%infile.split('.')[0],'w')

# se define la estructura
pdb1 = rpt.PdbStruct(file1)
pdb2 = rpt.PdbStruct(file2)

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

pdb1.Set_SS()
pdb2.Set_SS()

ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)

def get_df_distancias(ref):
    """Funcion para obtener el dataframe de distancias de cada proteina"""
    # se generan listas con coordenadas y numero de atomo
    coord = [res.GetAtom('CA').coord for res in ref]
    index = [res.resi for res in ref]

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


# devuelve tabla e indices de el dataframe de distancias entre atomos de la misma proteina con dth < 10A
df_distancias1, index1 = get_df_distancias(pdb11)
df_distancias2, index2 = get_df_distancias(pdb22)

# se generan cliques, te devuleve dataframe con cliques de 3 y la lista de cliques maximales
df_cliques1, cliques1 = fc.gen_3_cliques(df_distancias1, file1[5:9], dth=10, k=number_elements_clique)
print('**'*50)
df_cliques2, cliques2 = fc.gen_3_cliques(df_distancias2, file2[5:9], dth=10, k=number_elements_clique)
print('**'*50)

# se agrega filtro de residuos que pertenecen a cliques de 7 elementos
mc1_7 = cliques1[cliques1.numero_elementos == 7].drop('numero_elementos',1)
mc2_7 = cliques2[cliques2.numero_elementos == 7].drop('numero_elementos',1)

residuos_unicos_1 = []
for i in [list(mc1_7[i].unique()) for i in mc1_7.dropna(1).columns]:
    for j in i:
        residuos_unicos_1.append(int(j))

residuos_unicos_2 = []
for i in [list(mc2_7[i].unique()) for i in mc2_7.dropna(1).columns]:
    for j in i:
        residuos_unicos_2.append(int(j))

residuos_unicos_1 = set(residuos_unicos_1)
residuos_unicos_2 = set(residuos_unicos_2)

for i in df_cliques1.columns:
    mask = np.where(df_cliques1[i].isin(residuos_unicos_1), True, False)
    df_cliques1 = df_cliques1[mask].reset_index(drop=True)

for i in df_cliques2.columns:
    mask = np.where(df_cliques2[i].isin(residuos_unicos_2), True, False)
    df_cliques2 = df_cliques2[mask].reset_index(drop=True)


# funcion para obtener las propiedades del residuo para los cliques agrupados
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

# se genera la columna de ss a la tabla de los cliques
df_cliques1 = fc.paste_SS(ss1, df_cliques1, num_cliques=number_elements_clique)
df_cliques2 = fc.paste_SS(ss2, df_cliques2, num_cliques=number_elements_clique)

# comparacion SSM #aqui se obtienen los candidatos posibles pasando el filtro de SS
candidatos_ss = fc.compare_SS(df_cliques1, df_cliques2, num_cliques=number_elements_clique)

# obtienes coordenadas de cada atomo de los cliques.
df_cliques1 = fc.get_coords_clique(df_atoms1, df_cliques1, number_elements_clique)
df_cliques2 = fc.get_coords_clique(df_atoms2, df_cliques2, number_elements_clique)

# calculo de baricentro baricentro clique
df_cliques1 = fc.baricenter_clique(df_cliques1, number_elements_clique)
df_cliques2 = fc.baricenter_clique(df_cliques2, number_elements_clique)

# calculo de vectores gorro
df_cliques1 = fc.center_vectors(df_cliques1, number_elements_clique)
df_cliques2 = fc.center_vectors(df_cliques2, number_elements_clique)

# para obtener el vector de columna de interes sin importar el numero de cliques.
idx_rmsd1, idx_rmsd2 = 3 * number_elements_clique, 4 * number_elements_clique + 3
# print(list(range(idx_rmsd1,idx_rmsd2)))
# se pasan a numpy arrays para mayor rapidez
array_df_cliques1 = df_cliques1.values[:, range(idx_rmsd1, idx_rmsd2)]  # del 9 al 15 columnas de interes
array_df_cliques2 = df_cliques2.values[:, range(idx_rmsd1, idx_rmsd2)]

# Se genera columna del calculo de distancia promedio para posteriormente filtrar por distancia promedio minima (dpm)
df_cliques1 = fc.get_distancia_promedio(number_elements_clique, df_cliques1)
df_cliques2 = fc.get_distancia_promedio(number_elements_clique, df_cliques2)

array_dist_promedio1 = df_cliques1.values[:, -1]  # el ultimo valor de distancia.
array_dist_promedio2 = df_cliques2.values[:, -1]

limite_distancia_minima = 0.45
if number_elements_clique == 4:
    limite_distancia_minima = 0.9
if number_elements_clique == 5:
    limite_distancia_minima = 1.8
if number_elements_clique == 6:
    limite_distancia_minima = 3.6
if number_elements_clique == 7:
    limite_distancia_minima = 4.5
if number_elements_clique == 8:
    limite_distancia_minima = 8.0

# filtro por distancia minima
candidatos_filter_dist = [(i, j) for i, j in candidatos_ss if (
        array_dist_promedio1[i] - array_dist_promedio2[j] >= -limite_distancia_minima) & (
        array_dist_promedio1[i] - array_dist_promedio2[j] <= limite_distancia_minima)]

print('num candidatos filtro SS', len(candidatos_ss))
print('num candidatos filtro distancia y ss', len(candidatos_filter_dist))

# calculo del RMSD

time = datetime.datetime.now()
print('tiempo pasado en filtros:', time - timenow)

timenow = datetime.datetime.now()

restriccion_rmsd = 0.15
if number_elements_clique == 4:
    restriccion_rmsd = 0.30
if number_elements_clique == 5:
    restriccion_rmsd = 0.60
if number_elements_clique == 6:
    restriccion_rmsd = 0.90
if number_elements_clique == 7:
    restriccion_rmsd = 1.50
if number_elements_clique == 8:
    restriccion_rmsd = 1.80

# calculo del RMSD
long_candidatos_ss = len(candidatos_filter_dist)

p = multiprocessing.Pool(multiprocessing.cpu_count()-1)

rmsd_1 = p.map(partial(fc.calculate_rmsd_rot_trans_m,
                       array_cliques1=array_df_cliques1,
                       array_cliques2=array_df_cliques2,
                       num_cliques=number_elements_clique), candidatos_filter_dist
               )

p.close()
p.join()

f1 = pd.DataFrame(rmsd_1)
f1 = f1[f1[0] <= restriccion_rmsd]
f1['candidato_clique_1'] = f1[1].str.get(0)
f1['candidato_clique_2'] = f1[1].str.get(1)
candidatos = f1[1].values

time = datetime.datetime.now()

print('numero de candidatos:', len(candidatos))
print('tiempo pasado:', time - timenow)

new_df_cliques1 = fc.add_element_clique(df_cliques1, 'candidato_clique_1', cliques1, f1, number_elements_clique)
new_df_cliques2 = fc.add_element_clique(df_cliques2, 'candidato_clique_2', cliques2, f1, number_elements_clique)
number_elements_clique = number_elements_clique + 1


def iter_rmsd(new_df_cliques1,new_df_cliques2,number_elements_clique):
    for i in new_df_cliques1.columns:
        mask = np.where(new_df_cliques1[i].isin(residuos_unicos_1), True, False)
        new_df_cliques1 = new_df_cliques1[mask].reset_index(drop=True)

    for i in new_df_cliques2.columns:
        mask = np.where(new_df_cliques2[i].isin(residuos_unicos_2), True, False)
        new_df_cliques2 = new_df_cliques2[mask].reset_index(drop=True)

    new_df_cliques1 = fc.paste_SS(ss1, new_df_cliques1, num_cliques=number_elements_clique)
    new_df_cliques2 = fc.paste_SS(ss2, new_df_cliques2, num_cliques=number_elements_clique)
    candidatos_ss = fc.compare_SS(new_df_cliques1, new_df_cliques2, num_cliques=number_elements_clique)
    df_cliques1 = fc.get_coords_clique(df_atoms1, new_df_cliques1, number_elements_clique)
    df_cliques2 = fc.get_coords_clique(df_atoms2, new_df_cliques2, number_elements_clique)
    df_cliques1 = fc.baricenter_clique(df_cliques1, number_elements_clique)
    df_cliques2 = fc.baricenter_clique(df_cliques2, number_elements_clique)
    df_cliques1 = fc.center_vectors(df_cliques1, number_elements_clique)
    df_cliques2 = fc.center_vectors(df_cliques2, number_elements_clique)
    idx_rmsd1, idx_rmsd2 = 3 * number_elements_clique, 4 * number_elements_clique + 3
    array_df_cliques1 = df_cliques1.values[:, range(idx_rmsd1, idx_rmsd2)]  # del 9 al 15
    array_df_cliques2 = df_cliques2.values[:, range(idx_rmsd1, idx_rmsd2)]
    # print(len(candidatos_ss))
    df_cliques1 = fc.get_distancia_promedio(number_elements_clique, df_cliques1)
    df_cliques2 = fc.get_distancia_promedio(number_elements_clique, df_cliques2)
    array_dist_promedio1 = df_cliques1.values[:, -1]  # el ultimo valor de distancia.
    array_dist_promedio2 = df_cliques2.values[:, -1]
    limite_distancia_minima = 0.45
    if number_elements_clique == 4:
        limite_distancia_minima = 0.9
    if number_elements_clique == 5:
        limite_distancia_minima = 1.8
    if number_elements_clique == 6:
        limite_distancia_minima = 3.6
    if number_elements_clique == 7:
        limite_distancia_minima = 4.5
    if number_elements_clique == 8:
        limite_distancia_minima = 8.0

    # filtro por distancia minima
    candidatos_filter_dist = [(i, j) for i, j in candidatos_ss if (
            array_dist_promedio1[i] - array_dist_promedio2[j] >= -limite_distancia_minima) & (
                                      array_dist_promedio1[i] - array_dist_promedio2[j] <= limite_distancia_minima)]

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    restriccion_rmsd = 0.15
    if number_elements_clique == 4:
        restriccion_rmsd = 0.30
    if number_elements_clique == 5:
        restriccion_rmsd = 0.60
    if number_elements_clique == 6:
        restriccion_rmsd = 0.90
    if number_elements_clique == 7:
        restriccion_rmsd = 1.50
    if number_elements_clique == 8:
        restriccion_rmsd = 1.80

    rmsd_1 = p.map(partial(fc.calculate_rmsd_rot_trans_m,
                           array_cliques1=array_df_cliques1,
                           array_cliques2=array_df_cliques2,
                           num_cliques=number_elements_clique), candidatos_filter_dist)

    p.close()
    p.join()

    f1 = pd.DataFrame(rmsd_1, columns=['rmsd','candidatos','matriz_rotacion'])
    f1 = f1[f1.rmsd <= restriccion_rmsd]
    f1['candidato_clique_1'] = f1.candidatos.str.get(0)
    f1['candidato_clique_2'] = f1.candidatos.str.get(1)
    candidatos = f1.candidatos.values
    time = datetime.datetime.now()

    print('numero de candidatos:', len(candidatos))
    print('tiempo pasado:', time - timenow)

    if number_elements_clique < 7:
        new_df_cliques1 = fc.add_element_clique(df_cliques1, 'candidato_clique_1', cliques1, f1, number_elements_clique)
        new_df_cliques2 = fc.add_element_clique(df_cliques2, 'candidato_clique_2', cliques2, f1, number_elements_clique)
        number_elements_clique = number_elements_clique + 1
    return(new_df_cliques1, new_df_cliques2, number_elements_clique,candidatos,f1)


for k in range(4):
    new_df_cliques1, new_df_cliques2, number_elements_clique, candidatos, rmsd = iter_rmsd(
        new_df_cliques1, new_df_cliques2, number_elements_clique)




# Una ves que agrega el 8tavo candidato corre lo siguiente
# def alineacion(new_df_cliques1,new_df_cliques2, candidatos, df_atoms1,df_atoms2,number_elements_clique = 8):
# rmsd = None

# new_df_cliques1 = fc.get_coords_clique(df_atoms1, new_df_cliques1, number_elements_clique)
# new_df_cliques2 = fc.get_coords_clique(df_atoms2, new_df_cliques2, number_elements_clique)
#
# new_df_cliques1 = fc.baricenter_clique(new_df_cliques1, number_elements_clique)
# new_df_cliques2 = fc.baricenter_clique(new_df_cliques2, number_elements_clique)

new_df_cliques1.to_pickle('clique1.pkl')
new_df_cliques2.to_pickle('clique2.pkl')
df_atoms1.to_pickle('clique1_df_atributos.pkl')
df_atoms2.to_pickle('clique2_df_atributos.pkl')

pd.DataFrame(rmsd).reset_index(drop=True).to_pickle('rmsd_picke.pkl')
pd.DataFrame(candidatos).to_csv("candidatos.csv", index=False)

rmsd.reset_index(drop=True, inplace=True)

lista_vectores_gorro = []
for i, bari in enumerate(new_df_cliques1.baricentro_clique.values):
    lista_pre_vectores = []
    for coord in df_atoms1.vector.values:
        lista_pre_vectores.append(coord - bari)

    lista_vectores_gorro.append(lista_pre_vectores)

vectores_gorro_proteina_1 = pd.DataFrame(lista_vectores_gorro)

lista_vectores_gorro = []
for i, bari in enumerate(new_df_cliques2.baricentro_clique.values):
    lista_pre_vectores = []
    for coord in df_atoms2.vector.values:
        lista_pre_vectores.append(coord - bari)

    lista_vectores_gorro.append(lista_pre_vectores)

vectores_gorro_proteina_2 = pd.DataFrame(lista_vectores_gorro)
# se obtiene la matriz de rotacion del menor rmsd
# se aplica a todos los vectores gorro de la proteina 1 que ya se le quito el baricentro del candidato 1
# para cada candidato
candidato = []
protein_to_compare = [i for i in df_atoms2.vector.values]

for idx in range(rmsd.shape[0]):
    # tomas la matriz de rotacion y se la aplicas a los vectores gorro correspondientes
    matriz_rotacion = rmsd.iloc[idx].matriz_rotacion

    vector_gorro = vectores_gorro_proteina_1.iloc[rmsd.iloc[idx].candidato_clique_1].values

    coord_vectores_rotados = [np.matmul(matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]

    baricentro_proteina_2 = new_df_cliques2.iloc[rmsd.iloc[idx].candidato_clique_2].baricentro_clique

    protein_trasladado_rotado = coord_vectores_rotados + baricentro_proteina_2  # nuevas coordendas proteina 1
    # RMSD
    p12 = np.sum((protein_to_compare - protein_trasladado_rotado) ** 2, 1)
    rmsd_i = lambda i: np.sqrt(i) / 3
    candidato.append([np.where(rmsd_i(p12) <= 0.05, 1, 0).mean(), idx])

df_so = pd.DataFrame(candidato, columns=['SO', 'index'])
print(df_so.idxmax())
print(df_so.max())
numero = int(input("escribe el numero de idx"))
    # df_so[df_so.SO == df_so.SO.quantile(0.95)].index[np.random.randint(df_so[df_so.SO == df_so.SO.quantile(0.95)].shape[0])]

idx = numero
matriz_rotacion = rmsd.iloc[idx].matriz_rotacion

vector_gorro = vectores_gorro_proteina_1.iloc[rmsd.iloc[idx].candidato_clique_1].values

coord_vectores_rotados = [np.matmul(matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]

baricentro_proteina_2 = new_df_cliques2.iloc[rmsd.iloc[idx].candidato_clique_2].baricentro_clique

protein_trasladado_rotado = coord_vectores_rotados + baricentro_proteina_2  # nuevas coordendas proteina 1

new_df_atom1 = pd.concat([df_atoms1,pd.DataFrame(protein_trasladado_rotado, columns=['x','y','z'])],1)
new_df_atom1['new_vector'] = [
    [new_df_atom1.iloc[i]['x'], new_df_atom1.iloc[i]['y'], new_df_atom1.iloc[i]['z']] for i in range(new_df_atom1.shape[0])]

for i in pdb11:
    mask = np.where(i.resi == new_df_atom1.residue_number, True, False)
    for j in new_df_atom1[mask].atom_name:
        mask_2 = np.where(new_df_atom1[mask].atom_name == j, True, False)
        i.GetAtom(j).UpDateValue('coord', new_df_atom1[mask][mask_2].new_vector.values[0])

pdb1.pdbdata = pdb11
pdb1.WriteToFile()

time_bueno = datetime.datetime.now()
print('iteraciones completas:', time_bueno - timenow_bueno)
