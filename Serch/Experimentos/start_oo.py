#!/usr/bin/env python
# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
# sys.path.append("../math_tricks/")
# import math_vect_tools as mymath

# herramientas para leer pdbs
import read_pdb_tools as rpt
# calculo de distancia
from scipy.spatial.distance import pdist
# libreria de tablas
import pandas as pd
# funciones de click generadas en pandas
import funciones_CLICK as fc
# cuenta tiempo de ejecucion
import datetime
# multiprocessing
import multiprocessing
from functools import partial

# Inicio de tiempo
timenow_bueno = datetime.datetime.now()

timenow = datetime.datetime.now()

# lectura de archivo
file1 = 'pdbs/2mhu.pdb'  # sys.argv[1]
file2 = 'pdbs/2mrt.pdb'  # sys.argv[2]

# numero de cliques, preguntar en el software para generalizarlo INPUT...
number_elements_clique = 3

# se define la estructura
pdb1 = rpt.PdbStruct(file1)
pdb2 = rpt.PdbStruct(file2)

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)

# Se calculan sus vecinos mas cercanos
pdb1.GetNeighbors()
pdb2.GetNeighbors()

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

# Siempre la proteina 1 es el que se rota y traslada para embonar en la proteina 2
if len(pdb22) < len(pdb11):

    import copy
    pdb1_temp = copy.copy(pdb1)
    pdb2_temp = copy.copy(pdb2)

    pdb11_temp = copy.copy(pdb11)
    pdb22_temp = copy.copy(pdb22)

    pdb1 = pdb2_temp
    pdb2 = pdb1_temp

    pdb11 = pdb22_temp
    pdb22 = pdb11_temp

    del [pdb1_temp]
    del [pdb2_temp]
    del [pdb11_temp]
    del [pdb22_temp]

    print("Intercambio de nombre ya que la proteina 1 es mas grande que la 2")
    print(pdb1.name, len(pdb11))
    print(pdb2.name, len(pdb22))
    print("No te preocupes ya quedo :)")

# Agregando estructura seucndaria por medio de DSSP (necesario Linux)
pdb1.Set_SS()
pdb2.Set_SS()
# creando tabla de estructura secundaria para filtro de SS
ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)


def get_df_distancias(ref):
    """Funcion para obtener el dataframe de distancias de cada residuo
    Dudas en codigo pueden revisar fc.distancia_entre_atomos en ese se basa
    esta funcion, la diferencia es que se crea con el objeto residuo"""
    # se generan listas con coordenadas y numero de atomo
    coord = [res.GetAtom('CA').coord for res in ref]
    lista_residuos = [res.resi for res in ref]

    # calcula distancia y regresa dataframe
    distancias = [ [pdist(np.array([v, av]), metric='euclidean').item() for av in coord] for v in coord]

    # se genera la matriz de adyacencias para la red
    df_da = pd.DataFrame(index=lista_residuos, columns=lista_residuos, data=distancias)
    return(df_da, lista_residuos)


# devuelve tabla e indices de el dataframe de distancias entre atomos de la misma proteina con dth < 10A
df_distancias1, lista_residuos_1 = get_df_distancias(pdb11)
df_distancias2, lista_residuos2 = get_df_distancias(pdb22)

# se generan cliques, te devuleve dataframe con cliques de k(numero_de_cliques) y la lista de cliques maximales
df_cliques1, cliques1 = fc.gen_3_cliques(df_distancias1,  dth=10, k=number_elements_clique)  # file1[:-4] pal gexf
print('**'*50)
df_cliques2, cliques2 = fc.gen_3_cliques(df_distancias2,  dth=10, k=number_elements_clique)  # file2[:-4]
print('**'*50)

# REVISAR DONDE GUARDA EL GEXF Y COLOCARLO EN LA CARPETA Grafos
# se agrega filtro de residuos que pertenecen a cliques de 7 elementos
mc1_7 = cliques1[cliques1.numero_elementos == 7].drop('numero_elementos', 1)
mc2_7 = cliques2[cliques2.numero_elementos == 7].drop('numero_elementos', 1)

residuos_unicos_1 = []
for i in [list(mc1_7[i].unique()) for i in mc1_7.dropna(1).columns]:
    for j in i:
        residuos_unicos_1.append(int(j))

residuos_unicos_2 = []
for i in [list(mc2_7[i].unique()) for i in mc2_7.dropna(1).columns]:
    for j in i:
        residuos_unicos_2.append(int(j))
# set de residuos unicos en 7-clique
residuos_unicos_1 = set(residuos_unicos_1)
residuos_unicos_2 = set(residuos_unicos_2)


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
    res_ngb = []
    for res in list_of_residues:
        for atom in res.atoms:
            atom_number.append(atom.atom_number)
            atom_name.append(atom.name)
            residue_name.append(res.resn)
            residue_number.append(res.resi)
            coord.append(atom.coord)
            res_ngb.append(res.ngb)

    df_atoms = pd.DataFrame(columns=['atom_number', 'atom_name', 'residue_name',
                                   'residue_number', 'vector', 'ngb'])
    df_atoms.atom_number = atom_number
    df_atoms.atom_name = atom_name
    df_atoms.residue_name = residue_name
    df_atoms.residue_number = residue_number
    df_atoms.vector = coord
    df_atoms.ngb = res_ngb

    return(df_atoms)


# CREAR DF_atomos_CA #
df_atoms1 = get_df_ca(pdb11)
df_atoms2 = get_df_ca(pdb22)

# calculo del RMSD y filtros

time = datetime.datetime.now()
print('tiempo generando red:', time - timenow)

print("Se procede a calculo de cliques")

timenow = datetime.datetime.now()


#ESTA MAL COMO CALCULA EL RMSD O COMO ROTA Y TRASLADA RESTRUCTURAR O HACER NUEVAMENTE ESTA FUNCION

def iter_rmsd(new_df_cliques1, new_df_cliques2, number_elements_clique):

    new_df_cliques1.sort_values([0,1,2]).to_pickle('cliques1_'+str(number_elements_clique)+'.pkl')
    new_df_cliques2.sort_values([0,1,2]).to_pickle('cliques2_'+str(number_elements_clique)+'.pkl')

    if number_elements_clique == 7:
        print("Filtrando candidatos y preparando datos para alineamiento")

    # filtro residuos de 7 unicos
    # for col in new_df_cliques1.columns:
    #     mask = np.where(new_df_cliques1[col].isin(residuos_unicos_1), True, False)
    #     new_df_cliques1 = new_df_cliques1[mask].reset_index(drop=True)
    #
    # for col in new_df_cliques2.columns:
    #     mask = np.where(new_df_cliques2[col].isin(residuos_unicos_2), True, False)
    #     new_df_cliques2 = new_df_cliques2[mask].reset_index(drop=True)

    # filtro estructura secundaria
    new_df_cliques1 = fc.paste_SS(ss1, new_df_cliques1, num_cliques=number_elements_clique)
    new_df_cliques2 = fc.paste_SS(ss2, new_df_cliques2, num_cliques=number_elements_clique)
    candidatos_ss = fc.compare_SS(new_df_cliques1, new_df_cliques2, num_cliques=number_elements_clique)

    print(candidatos_ss)

    print(len(candidatos_ss))
    # rotando y trasladando
    df_cliques1 = fc.get_coords_clique(df_atoms1, new_df_cliques1, number_elements_clique)
    df_cliques2 = fc.get_coords_clique(df_atoms2, new_df_cliques2, number_elements_clique)
    df_cliques1 = fc.baricenter_clique(df_cliques1, number_elements_clique)
    df_cliques2 = fc.baricenter_clique(df_cliques2, number_elements_clique)
    df_cliques1 = fc.center_vectors(df_cliques1, number_elements_clique)
    df_cliques2 = fc.center_vectors(df_cliques2, number_elements_clique)
    idx_rmsd1, idx_rmsd2 = 3 * number_elements_clique, 4 * number_elements_clique + 3  # guardas la posicion de los
    # vectores
    array_df_cliques1 = df_cliques1.values[:, range(idx_rmsd1, idx_rmsd2)]  # del 9 al 15
    array_df_cliques2 = df_cliques2.values[:, range(idx_rmsd1, idx_rmsd2)]

    candidatos_filter_dist = candidatos_ss

    # filtro por restriccion de RMSD despues de ajuste 3D
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

    df_candidatos = pd.DataFrame(rmsd_1, columns=['rmsd', 'candidatos', 'matriz_rotacion'])
    # df_candidatos.to_pickle('rmsd_'+str(number_elements_clique)+'.pkl')
    df_candidatos = df_candidatos[df_candidatos.rmsd <= restriccion_rmsd]
    df_candidatos['candidato_clique_1'] = df_candidatos.candidatos.str.get(0)
    df_candidatos['candidato_clique_2'] = df_candidatos.candidatos.str.get(1)
    time = datetime.datetime.now()

    print(df_candidatos)

    print('numero de candidatos:', df_candidatos.shape[0])
    print('tiempo pasado:', time - timenow)
    flag = 0
    if df_candidatos.shape[0] == 1:
        flag = 1
        return new_df_cliques1, new_df_cliques2, number_elements_clique, df_candidatos, flag

    # Se agrega un nuevo elemento a los cliques.
    if number_elements_clique < 7:
        new_df_cliques1 = fc.add_element_clique(df_cliques1, 'candidato_clique_1', cliques1, df_candidatos,
                                                number_elements_clique)

        new_df_cliques2 = fc.add_element_clique(df_cliques2, 'candidato_clique_2', cliques2, df_candidatos,
                                                number_elements_clique)
        number_elements_clique = number_elements_clique + 1


    return (new_df_cliques1, new_df_cliques2, number_elements_clique, df_candidatos,flag)


# primeros cliques a filtrar
new_df_cliques1 = df_cliques1
new_df_cliques2 = df_cliques2

# Si se empieza en 3-cliques solo itera 5 veces para llegar a 7 :v.
for k in range(5):
    new_df_cliques1, new_df_cliques2, number_elements_clique, rmsd, stop = iter_rmsd(
        new_df_cliques1, new_df_cliques2, number_elements_clique)
    print("iteracion", k + 1, "numero de elementos:", number_elements_clique)
    print("===" * 10)
    if stop:
        break


time_all = datetime.datetime.now()
print('tiempo generando cliques candidatos pasando por todos los filtros:', time_all - timenow)
print("Se procede a alineamiento")

timenow = datetime.datetime.now()
# Obtenemos los elementos para generar el Alineamiento final.
# Utilizando los datos de los cliques, el numero de elementos del clique, la lista de candidatos que sobrevivieron
# y el dataframe del indice de candidatos con el RMSD de cada alineamiento
# quitando el clique seleccionado

df_rmsd = rmsd.reset_index(drop=True)

# se obtiene la matriz de rotacion del menor rmsd
# se aplica a todos los vectores gorro de la proteina 1 que ya se le quito el baricentro del candidato 1
# para cada candidato

# protein_to_compare = np.array([vectors for vectors in df_atoms2[df_atoms2.atom_name == 'CA'].vector.values])

atoms_numbers1 = df_atoms1[df_atoms1.atom_name == 'CA'].index.values
atoms_numbers2 = df_atoms2[df_atoms2.atom_name == 'CA'].index.values

df_new_df_clique1 = new_df_cliques1.values
df_new_df_clique2 = new_df_cliques2.values

df_rmsd_1 = df_rmsd.values

vecs1 = np.array([vecs for vecs in df_atoms1.vector.values])
vecs2 = np.array([vecs for vecs in df_atoms2.vector.values])

# baricentro
bari_1 = vecs1.mean(0)
bari_2 = vecs2.mean(0)

# vectores centricos
vecs_center_1 = vecs1 - bari_1
vecs_center_2 = vecs2 - bari_2

number_of_residues_final = df_atoms1[df_atoms1.atom_name == 'CA'].shape[0]

new_df_cliques1.to_pickle('clique1.pkl')
new_df_cliques2.to_pickle('clique2.pkl')
# df_atoms1.to_pickle('clique1_df_atributos.pkl')
# df_atoms2.to_pickle('clique2_df_atributos.pkl')
# # df_candidatos.to_pickle('df_alineados.pkl')
# # pd.DataFrame(parejas_clique).to_pickle('parejas.pkl')
# df_rmsd.to_pickle('df_rmsd.pkl')


def align_residues(idx, so):

    clique_in_protein_1 = df_rmsd_1[idx, 3]  # se extrae el indice de los cliques
    clique_in_protein_2 = df_rmsd_1[idx, 4]

    # se obtienen el numero de residuo de los candidatos
    los_del_clique_1 = df_new_df_clique1[clique_in_protein_1, [0, 1, 2, 3, 4, 5, 6]]

    los_del_clique_2 = df_new_df_clique2[clique_in_protein_2, [0, 1, 2, 3, 4, 5, 6]]

    baricentros_1 = new_df_cliques1.baricentro_clique.values
    # Comprension de lista
    lista_vectores_gorro = [[coord - bari for coord in df_atoms1[
            (df_atoms1.atom_name == 'CA') &
            (df_atoms1.residue_number.isin(
             los_del_clique_1) == False)].vector.values] for bari in baricentros_1]

    vectores_gorro_proteina_1 = pd.DataFrame(lista_vectores_gorro)

    df_vectores_gorro_proteina_1 = vectores_gorro_proteina_1.values
    # OJO se extrae por indice no por numero de atomos!!!
    atoms1 = df_atoms1[(df_atoms1.atom_name == 'CA') & (df_atoms1.residue_number.isin(los_del_clique_1) == False)].index.values
    atoms2 = df_atoms2[(df_atoms2.atom_name == 'CA') & (df_atoms2.residue_number.isin(los_del_clique_2) == False)].index.values

    # se guardan los vectores
    protein_to_compare = np.array([vectors for vectors in df_atoms2.iloc[atoms2].vector.values])

    matriz_rotacion = df_rmsd_1[idx, 2]  # se obtiene la matriz de rotacion de los cliques candidatos

    # se obtienen los vectores gorro de la proteina a rotar y trasladar
    vector_gorro = df_vectores_gorro_proteina_1[clique_in_protein_1]
    # se aplica la rotacion a vectores gorro
    coord_vectores_rotados = [np.matmul(matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]
    # se obtiene el baricentro de la proteina 2
    baricentro_proteina_2 = df_new_df_clique2[clique_in_protein_2, 22]
    # print(baricentro_proteina_2)
    # se suma el baricentro a vectores rotados de la proteina 1
    protein_trasladado_rotado = coord_vectores_rotados + baricentro_proteina_2  # nuevas coordendas proteina 1
    # se quitan residuos que pertenezcan al clique candidato para calcular la distancia y emparejar al mejor.

    protein_trasladado_rotado_sin_clique = protein_trasladado_rotado
    protein_to_compare_sin_clique = protein_to_compare

    residuos_match = []  # aqui se guardan las parejas de residuos

    # se itera por cada residuo ya rotado y trasladado

    for residuo_not_in_clique, res_1 in zip(protein_trasladado_rotado_sin_clique, atoms1):
        for residuo, res_2 in zip(protein_to_compare_sin_clique, atoms2):
            distancia = pdist(np.array([residuo_not_in_clique, residuo]), metric='euclidean').item()
            if distancia < 3.5:
                residuos_match.append([distancia, (res_1, res_2)])

    residuos_match = sorted(residuos_match)

    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2) or (i[0] > 3.5):
            continue
        else:
            c1.append(i[1][0])
            c2.append(i[1][1])
            cand_n.append(i)

    parejas = [i[1] for i in cand_n]

    def R_ij(i, j, parejas=parejas):
        valor = sum([vecs_center_1[pos_res[0], i] * vecs_center_2[pos_res[1], j] for pos_res in parejas])
        return valor

    def matrix_R():
        """cliques a comparar: i,j
        desde aqui se itera sobre cada i y hay que variar los vectores
        coordenada
        Regresa la matriz gigante (matriz simetrica del articulo)"""
        # primer renglon
        R11R22R33 = (R_ij(0, 0) + R_ij(1, 1) + R_ij(2, 2))
        R23_R32 = (R_ij(1, 2) - R_ij(2, 1))
        R31_R13 = (R_ij(2, 0) - R_ij(0, 2))
        R12_R21 = (R_ij(0, 1) - R_ij(1, 0))
        # segundo renglon
        R11_R22_R33 = (R_ij(0, 0) - R_ij(1, 1) - R_ij(2, 2))
        R12R21 = (R_ij(0, 1) + R_ij(1, 0))
        R13R31 = (R_ij(0, 2) + R_ij(2, 0))
        # tercer renglon
        _R11R22_R33 = (-R_ij(0, 0) + R_ij(1, 1) - R_ij(2, 2))
        R23R32 = (R_ij(1, 2) + R_ij(2, 1))
        # cuarto renglon
        _R11_R22R33 = (-R_ij(0, 0) - R_ij(1, 1) + R_ij(2, 2))

        matriz_R = [
            [R11R22R33, R23_R32, R31_R13, R12_R21],
            [R23_R32, R11_R22_R33, R12R21, R13R31],
            [R31_R13, R12R21, _R11R22_R33, R23R32],
            [R12_R21, R13R31, R23R32, _R11_R22R33]
        ]
        return (matriz_R)


    matriz_R = matrix_R()
    matriz_rotacion = fc.rotation_matrix(matriz_R)
    vector_rotado = fc.rotation_vectors(vecs_center_1, matriz_rotacion)
    vector_rotado_trasladado_a_clique2 = vector_rotado + np.array(bari_2)
    protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
    protein_to_compare = vecs2

    residuos_match = []

    for residuo_1, res_1 in zip(protein_trasladado_rotado, atoms_numbers1):
        for residuo_2, res_2 in zip(protein_to_compare, atoms_numbers2):
            distancia = pdist(np.array([residuo_1, residuo_2]), metric='euclidean').item()
            if distancia < 3.5:
                residuos_match.append([distancia, (res_1, res_2)])

    residuos_match = sorted(residuos_match)

    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2) or (i[0] > 3.5):
            continue
        else:
            c1.append(i[1][0])
            c2.append(i[1][1])
            cand_n.append(i)

    df = pd.DataFrame(cand_n, columns=['distancia', 'candigatos'])

    so_temp = len(cand_n) / number_of_residues_final

    return idx, so_temp, df, matriz_rotacion

timenow_bueno = datetime.datetime.now()

# Aqui se aplica el alineamiento para cada candidato de cliques y obtener el SO
id_so = 0
so = 0.0
df_candidatos = pd.DataFrame()
parejas_clique = []
matriz_rotacion = None
print("Numero candidatos:", len(df_rmsd))
for i in np.arange(len(df_rmsd)):
    id_temp, so_temp, df_temp, matriz_rotacion_temp = (align_residues(i, so))
    if so <= so_temp:
        so = so_temp
        id_so = id_temp
        df_candidatos = df_temp
        matriz_rotacion = matriz_rotacion_temp
        print("candidato: %i" % id_so, "SO: %1.3f" % so, "RMSD: %1.3f" % df_candidatos.distancia.mean())
    if so == 1:
        break

print('ya termine de alinear residuo a residuo')
print(id_so, round(so, 5), df_candidatos.distancia.mean(), df_candidatos)

# new_df_cliques1.to_pickle('clique1.pkl')
# new_df_cliques2.to_pickle('clique2.pkl')
# df_atoms1.to_pickle('clique1_df_atributos.pkl')
# df_atoms2.to_pickle('clique2_df_atributos.pkl')
# df_candidatos.to_pickle('df_alineados.pkl')
# pd.DataFrame(parejas_clique).to_pickle('parejas.pkl')
# df_rmsd.to_pickle('df_rmsd.pkl')


# matriz_R = matrix_R()
# matriz_rotacion = rotation_matrix(matriz_R)
vector_rotado = fc.rotation_vectors(vecs_center_1, matriz_rotacion)
vector_rotado_trasladado_a_clique2 = vector_rotado + np.array(bari_2)
protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
protein_to_compare = vecs2

new_df_atom1 = pd.concat([df_atoms1, pd.DataFrame(protein_trasladado_rotado, columns=['x', 'y', 'z'])], 1)
new_df_atom1['new_vector'] = [
    [new_df_atom1.iloc[i]['x'], new_df_atom1.iloc[i]['y'], new_df_atom1.iloc[i]['z']] for i in range(new_df_atom1.shape[0])]

for i in pdb11:
    mask = np.where(i.resi == new_df_atom1.residue_number, True, False)
    for j in new_df_atom1[mask].atom_name:
        mask_2 = np.where(new_df_atom1[mask].atom_name == j, True, False)
        i.GetAtom(j).UpDateValue('coord', new_df_atom1[mask][mask_2].new_vector.values[0])

pdb1.pdbdata = pdb11
# pdb1.WriteToFile(file_out_name='1xxa_'+str(datetime.datetime.now()))

time_bueno = datetime.datetime.now()
print('iteraciones completas:', time_bueno - timenow_bueno)
print("archivo nuevo con el nombre de:", '1xxa_'+str(datetime.datetime.now()))
