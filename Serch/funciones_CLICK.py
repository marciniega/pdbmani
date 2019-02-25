# coding: utf-8

# libreria de analisis de datos y una caracterizacion para su facil lectura.
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 40)
pd.set_option('display.max_colwidth', -1)
# libreria de generacion de rede y cliques
import networkx as nx, community

# libreria de calculo de distancia euclidiana
from scipy.spatial.distance import pdist, euclidean

# libreria de mate
import numpy as np

# libreria de iteraciones
import itertools as it

# Libreria de MA para RMSD
import sys
sys.path.append('math_tricks/')
# import math_vect_tools as mvt


# se calcula la distancia entre cada par de nodos.
def distancia_entre_atomos(df_atoms):
    """df_ca: Dataframe con coordenadas de los atomos alfa, devuelve otro DataFrame
    df_da: Dataframe como una matriz de adyacencias donde el valor es la distancia"""
    df_atoms_ca = df_atoms[df_atoms.atom_name == 'CA']
    distancias = []
    # se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
    for v, i in zip(df_atoms_ca.vector_coordenadas,df_atoms_ca.atom_number):
        distancia_un_atomo = []
        for av, j in zip(df_atoms_ca.vector_coordenadas,df_atoms_ca.atom_number):
            distancia = pdist(np.array([v,av]), metric='euclidean').item()
            distancia_un_atomo.append(distancia)
        distancias.append(distancia_un_atomo)
    # se genera la matriz de adyacencias para la red
    df_distancias = pd.DataFrame(index=df_atoms_ca.atom_number,
                         columns=df_atoms_ca.atom_number,
                         data=distancias)
    return(df_distancias)


def gen_cliques(red, k=7): # k modificar a 7

    cliques_completos = [clq for clq in nx.find_cliques(red) if len(clq) == k] # mayor o igual que cambiarlo
    print('numero de cliques maximos encontrados:', len(cliques_completos))
    # print(cliques_completos_1)

    lista_cliques = []
    for v in (cliques_completos):
        a = list(it.combinations(v, 3))
        for j in a:
            permutation_temp = it.permutations(j)
            lista_cliques.append(set(permutation_temp))

    lista_cliques = np.unique(lista_cliques)
    # [y for x in list_of_lists for y in x]
    return lista_cliques, cliques_completos


def gen_cliques_3(red):

    combinations = list(it.combinations(red, 3))
    lista_cliques = [list(it.permutations(nclique)) for nclique in combinations]
    lista_cliques1 = [j for i in lista_cliques for j in i]

    return lista_cliques1


def create_ss_table(list_residues):
    ss_list = [i.ss for i in list_residues]
    num_resi_list = [i.resi for i in list_residues]
    chain_list = [i.chain for i in list_residues]

    ss = pd.DataFrame()
    ss['structure'] = ss_list
    ss['residue_number'] = num_resi_list
    ss['chain'] = chain_list
    return ss


def compare_SS(df_clique1, df_clique2, num_cliques):
    cols = df_clique1.columns[num_cliques:]
    producto = it.product(df_clique1.index.values, df_clique2.index.values)
    comp1 = df_clique1[cols].values
    comp2 = df_clique2[cols].values
    candidatos_ss = [(i, j) for i, j in producto if 2 not in (list(map(SSM, comp1[i], comp2[j])))]

    return (candidatos_ss)


def SSM(ss1,ss2):
    """Catalogo SSM siguiendo la tabla 1 y con una funcion extra,
    ss1: string (H,B,C)
    ss2: string (H,B,C)
    devuelve el score: int (0,1,2)"""
    def get_score_from_table(ss1, ss2):

        if (ss1 == 'H') and (ss2 == 'B'):
            score_ss = 2
        elif (ss1 == 'B') and (ss2 == 'H'):
            score_ss = 2
        else:
            print('WTF are u doing!')

        return(score_ss)

    score_ss = 123

    if ss1 == ss2:
        score_ss = 0
    elif (ss1 != ss2) & ((ss1 == 'C') | (ss2 == 'C')):
        score_ss = 1
    else:
        score_ss = get_score_from_table(ss1, ss2)
        
    return(score_ss)


# Funciones que se necesitaran siempre
def rotation_matrix(matriz_R):
    """utilizando la funcion giant_matrix, fijando los valores de i,j
    se calcula la matriz de rotacion con los eigenvectores y eigenvalores
    arroja una matriz de rotacion que depende de la matriz gigante
    """
    eignvalues, eigenvectors = np.linalg.eig(matriz_R)
    q = eigenvectors[:, np.argmax(eignvalues)]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    # matriz de rotacion con eigenvectores
    matriz_rotacion = np.array([
        [(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)]
    ], dtype=np.float64)
    return (matriz_rotacion)


def rotation_vectors(vector_gorro, matriz_rotacion):
    """obtencion de vector rotado,
    utilizando la matriz de rotacion
    y los vectores gorro a rotar y trasladar"""

    coord_rotado = [np.matmul(
        matriz_rotacion, coord.reshape(3, 1)).T[0] for coord in vector_gorro]
    return (coord_rotado)


# EXTRACCION DE VARIABLES
# funcion de lectura con biopandas
# def read_biopdb(path):
#     """Extrae las cordenadas de los atomos de C_alfa y los acomoda en un vector
#     devuelve un dataframe con las coordenadas y el numero de residuo"""
#     # import biopandas as biop #NECESARIO INSTALAR PARA GENERAR ESTO
#     # df = biop.read_pdb(path)
#     df = pd.DataFrame() #Eliminar esta linea y dejar la de arriba si quieren utilizar esto.
#     df_all_atoms = df.df['ATOM']
#
#     df_atoms = df_all_atoms[[
#         'atom_number', 'atom_name', 'residue_name', 'residue_number',
#         'x_coord', 'y_coord', 'z_coord'
#     ]]
#     columna_vector = [np.array(i) for i in zip(df_atoms.x_coord.tolist(), df_atoms.y_coord.tolist(),
#                  df_atoms.z_coord.tolist())]
#
#     df_atoms['vector_coordenadas'] = columna_vector
#     return (df_atoms)

# GENERACION DE CLIQUES!!!
# def gen_3_cliques(df_distancias, nombre=False, dth=10, k=3):
#     """Genera n-cliques de dataframe de distancias, tomando en cuenta los enlaces menores o iguales
#     a dth y forma los k-cliques que elijas
#     valores por default:
#     dth=10, k=3"""
#     # red de distancias completa
#     red = nx.from_pandas_adjacency(df_distancias)
# #     print("red antes de filtros:",nx.info(red))
#
#     # filtro de distancias
#     edgesstrong = [(u, v) for (u, v, d) in red.edges(data=True) if d["weight"] <= dth]
#
#     red = nx.Graph(edgesstrong)
# #     print("=="*20)
# #     print("red despues de filtros:",nx.info(red))
#
#     cliques_completos = [clq for clq in nx.find_cliques(red) if len(clq) >=k]
#     print('numero de cliques maximos encontrados:',len(cliques_completos))
# #     print(n_cliques)
#
#     lista_cliques = []
#     for i, v in enumerate(cliques_completos):
#         a = list(it.combinations(v, k))
#         for j in a:
#             if set(j) not in lista_cliques:
#                 # recuerda que para comparar elementos utiliza set, y apilalos como set
#                 lista_cliques.append(set(j))
#
#     df_cliques = pd.DataFrame(lista_cliques)
#     print("numero de %s-cliques posibles:" % k, df_cliques.shape[0])
#
#     df_maximal_clique = pd.DataFrame(cliques_completos, dtype=int)
#     df_maximal_clique['numero_elementos'] = df_maximal_clique.count(1)
#     df_maximal_clique.sort_values('numero_elementos', inplace=True)
#
#     if nombre:
#         print('guardando red en: %s' % nombre)
#         nx.write_gexf(red, nombre+'.gexf')
#     return(df_cliques, df_maximal_clique)

# MACHEO DE CLIQUES
# funcion para obtener las coordenadas del clique
# def paste_SS(ss, df_cliques, num_cliques=3):
#     """ Obtenemos la estructura secundaria y la pega al dataframe generando el df s
#     """
#     #lista para apilar las estructuras
#     if num_cliques >= 3:
#         c1 = [ss[ss.residue_number == df_cliques.iloc[i, 0]].structure.values[0] for i in df_cliques.index]
#         c2 = [ss[ss.residue_number == df_cliques.iloc[i, 1]].structure.values[0] for i in df_cliques.index]
#         c3 = [ss[ss.residue_number == df_cliques.iloc[i, 2]].structure.values[0] for i in df_cliques.index]
#
#         df_cliques['ss_0'] = c1
#         df_cliques['ss_1'] = c2
#         df_cliques['ss_2'] = c3
#
#     if num_cliques >= 4:
#         c4 = [ss[ss.residue_number == df_cliques.iloc[i, 3]].structure.values[0] for i in df_cliques.index]
#         df_cliques['ss_3'] = c4
#
#     if num_cliques >= 5:
#         c5 = [ss[ss.residue_number == df_cliques.iloc[i, 4]].structure.values[0] for i in df_cliques.index]
#         df_cliques['ss_4'] = c5
#
#     if num_cliques >= 6:
#         c6 = [ss[ss.residue_number == df_cliques.iloc[i, 5]].structure.values[0] for i in df_cliques.index]
#         df_cliques['ss_5'] = c6
#
#     if num_cliques >= 7:
#         c7 = [ss[ss.residue_number == df_cliques.iloc[i, 6]].structure.values[0] for i in df_cliques.index]
#         df_cliques['ss_6'] = c7
#
#     if num_cliques >= 8:
#         c8 = [ss[ss.residue_number == df_cliques.iloc[i, 7]].structure.values[0] for i in df_cliques.index]
#         df_cliques['ss_7'] = c8
#
#     # columna con coordenadas del clique
#     return(df_cliques)


# funcion para obtener las coordenadas del clique
# def get_coords_clique(df_atoms_ca, df_cliques, num_cliques):
#     """df_ca:DataFrame con coordenadas de carbonos alfa,
#     df_lc:Dataframe con cliques, si coincide el numero del atomo
#     le pega su coordenada y genera una matriz de vectores que contiene
#     las coordenadas de cada atomo ordenado de izquierda a derecha como
#     aparecen en df_clique"""
#     if num_cliques >= 3:
#         c0 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 0]].vector.values[0]) for i in
#               df_cliques.index]
#         c1 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 1]].vector.values[0]) for i in
#               df_cliques.index]
#         c2 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 2]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_0'] = c0
#         df_cliques['coord_clique_1'] = c1
#         df_cliques['coord_clique_2'] = c2
#
#         if num_cliques == 3:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i]] for i in df_cliques.index]
#
#     if num_cliques >= 4:
#         c3 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 3]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_3'] = c3
#         if num_cliques == 4:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i]] for i in df_cliques.index]
#
#     if num_cliques >= 5:
#         c4 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 4]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_4'] = c4
#         if num_cliques == 5:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i]] for i in df_cliques.index]
#
#     if num_cliques >= 6:
#         c5 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 5]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_5'] = c5
#         if num_cliques == 6:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i], c5[i]] for i in df_cliques.index]
#
#     if num_cliques >= 7:
#         c6 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 6]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_6'] = c6
#         if num_cliques == 7:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i], c5[i], c6[i]] for i in
#                                                 df_cliques.index]
#
#     if num_cliques >= 8:
#         c7 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 7]].vector.values[0]) for i in
#               df_cliques.index]
#
#         df_cliques['coord_clique_7'] = c7
#         if num_cliques == 8:
#             df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i], c5[i], c6[i], c7[i]] for i in
#                                                 df_cliques.index]
#
#     return(df_cliques)


# def baricenter_clique(df_cliques, num_cliques):
#     """se calcula el baricentro de cada clique
#         siguiendo la formula puntos promedios.
#         df_cliques: Dataframe con los cliques y coordenadas
#         num_cliques:numero de cliques
#         x_barra = la suma de los x_i / numero de elementos en clique
#         y_barra = la suma de los y_i / numero de elementos en clique
#         z_barra = la suma de los z_i / numero de elementos en clique"""
#
#     coord_center = []  # se guardan los valores del baricentro
#     # se guarda el indice de la columna
#     indice_num_cliques_1 = 2 * num_cliques
#     indice_num_cliques_2 = 3 * num_cliques
#
#     if num_cliques == 8:
#         indice_num_cliques_1 = num_cliques
#         indice_num_cliques_2 = 2 * num_cliques
#
#     for i in range(df_cliques.shape[0]): # se recorre la tabla por indice
#         data = [df_cliques.iloc[i, j] for j in range(indice_num_cliques_1, indice_num_cliques_2)]
#         # para cada registro se calcula el punto medio de cada vector
#         coord_center.append(np.mean(data, axis=0))  # se apila el punto medio
#
#     df_cliques['baricentro_clique'] = coord_center  # se genera la columna
#     return(df_cliques)
#
#
# def center_vectors(df_cliques, num_cliques):
#     """se calcula los nuevos vectores gorro o centricos de cada clique
#             como:
#             x_gorro = x_i - x_barra
#             y_gorro = y_i - y_barra
#             z_gorro = z_i - z_barra.
#             df_cliques: Dataframe con los cliques y coordenadas
#             num_cliques:numero de cliques"""
#     indice_num_cliques_1 = 2 * num_cliques
#     indice_num_cliques_2 = 3 * num_cliques
#
#     if num_cliques == 8:
#         indice_num_cliques_1 = num_cliques
#         indice_num_cliques_2 = 2 * num_cliques
#
#     vec0 = []   # aqui se guardaran los valores de los vectores que se les resta el baricentro
#     vec1 = []
#     vec2 = []
#     vec3 = []
#     vec4 = []
#     vec5 = []
#     vec6 = []
#     vec7 = []
#     vectores_centricos = [] # aqui guardaremos todos los vectores, en lugar de tenerlos por separado
#     for i, val in enumerate(df_cliques.baricentro_clique.values):
#         # se recorre la tabla por indice y valor del baricentro
#         data = [df_cliques.iloc[i, j] for j in range(indice_num_cliques_1, indice_num_cliques_2)]
#         # se extrae las columnas de las coordenadas de los cliques
#         for n, k in enumerate(data):
#             vectors = (k - val)  # se le resta el baricentro a cada coordenada de clique
#
#             if n == 0:  # como va iterando sobre cada elemento del clique
#                 a_0 = vectors  # se guarda el vector creado de la resta
#                 vec0.append(vectors)  # se apila en su respectiva lista para generar la columna
#             if n == 1:
#                 a_1 = vectors
#                 vec1.append(vectors)
#             if n == 2:
#                 a_2 = vectors
#                 vec2.append(vectors)
#             if n == 3:
#                 a_3 = vectors
#                 vec3.append(vectors)
#             if n == 4:
#                 a_4 = vectors
#                 vec4.append(vectors)
#             if n == 5:
#                 a_5 = vectors
#                 vec5.append(vectors)
#             if n == 6:
#                 a_6 = vectors
#                 vec6.append(vectors)
#             if n == 7:
#                 a_7 = vectors
#                 vec7.append(vectors)
#
#         if num_cliques == 3:
#             # al final de la iteracion se apilan todos los vectores creados en una columna que contiene todos
#             vectores_centricos.append(np.array([a_0, a_1, a_2]))
#
#         if num_cliques == 4:
#             vectores_centricos.append(np.array([a_0, a_1, a_2, a_3]))
#
#         if num_cliques == 5:
#             vectores_centricos.append(np.array([a_0, a_1, a_2, a_3, a_4]))
#
#         if num_cliques == 6:
#             vectores_centricos.append(np.array([a_0, a_1, a_2, a_3, a_4, a_5]))
#
#         if num_cliques == 7:
#             vectores_centricos.append(np.array([a_0, a_1, a_2, a_3, a_4, a_5, a_6]))
#
#         if num_cliques == 8:
#             vectores_centricos.append(np.array([a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7]))
#
#     df_cliques['vec_gorro_0'] = vec0  # se crea la columna correspondiente
#     df_cliques['vec_gorro_1'] = vec1
#     df_cliques['vec_gorro_2'] = vec2
#
#     if num_cliques >= 4:
#         df_cliques['vec_gorro_3'] = vec3
#     if num_cliques >= 5:
#         df_cliques['vec_gorro_4'] = vec4
#     if num_cliques >= 6:
#         df_cliques['vec_gorro_5'] = vec5
#     if num_cliques >= 7:
#         df_cliques['vec_gorro_6'] = vec6
#     if num_cliques >= 8:
#         df_cliques['vec_gorro_7'] = vec7
#
#     df_cliques['vectores_gorro'] = vectores_centricos
#     return df_cliques


# def get_distancia_promedio(num_cliques, df_cliques):
#     """Calculo de distancia promedio sobre cada vector gorro"""
#     a = (0, 0, 0)
#     df_cliques['distancia_promedio'] = [np.mean([euclidean(a, j) for j in i]) for i in df_cliques.vectores_gorro]
#     return df_cliques


# Funcion para obtener la siguiente iteracion de candidatos.
# def add_element_clique(df_cliques, col_candidatos, cliques, candidatos_df, number_elements_clique):
#     cliques_maximales = cliques[
#         cliques.numero_elementos >= number_elements_clique].drop('numero_elementos', 1).values
#     set_candidatos = [df_cliques.iloc[i, range(
#         number_elements_clique)].values for i in candidatos_df[col_candidatos].unique()]
#     #conjunto de candidatos unicos
#     lista_residuos = []  # aqui se guardara la lista de numero de residuo
#     for candidato in set_candidatos:  # este va a cambiar cada iteracion
#         for clique_maximal in cliques_maximales:
#             clique_maximal = [i for i in clique_maximal if str(i) != 'nan']
#             if set(candidato).issubset(clique_maximal):       # si esta en un clique maximal
#                 no_estan_en_clique = set(clique_maximal).difference(set(candidato))
#                 # obten los elementos que no estan en ese clique que si estan en el clique maximal
#                 for nuevo_elemento in no_estan_en_clique:
#                     candidato_nuevo = candidato.copy()
#                     # se genera una copia para no borrar el orginal
#                     candidato_nuevo = np.append(candidato_nuevo, int(nuevo_elemento))
#                     # se apila un elemento de los que no estan
#                     # print(candidato_nuevo)
#                     # print(lista_residuos)
#                     if candidato_nuevo.tolist() not in lista_residuos:
#                         lista_residuos.append(candidato_nuevo.tolist())
#                         # si no esta en la lista pasa
#     # print(lista_residuos)
#     return pd.DataFrame(lista_residuos)

# CALCULO RMSD ROTO!!! NO LO HACE BIEN!!
# def calculate_rmsd_rot_trans_m(cliques, array_cliques1, array_cliques2, num_cliques):
#
#     res1, res2 = cliques
#
#     def R_ij(i, j, a1=0, a2=0):
#         """Recuerda que 0-->1,1-->2,2-->3 en los indices de R
#         a1,a2 corresponden a que atomo quieren que se compare
#         """
#         idx_vec_gorro1, idx_vec_gorro2 = 2, 2 + num_cliques
#         valor = sum(  # VECTORES GORRO AGARRARLOS!!!
#             [array_cliques1[:, k][a1][i] * array_cliques2[:, k][a2][j] for k in range(idx_vec_gorro1, idx_vec_gorro2)])
#         return (valor)
#
#     def matrix_R(i, j):
#         # CAMBIO (2018/10/24) se dejo de implementar el diccionario de R_ij y se cambiaron los indices
#         # para regresar al calculo del carticulo sumar 1 a los numeros de la matriz R
#         """cliques a comparar: i,j
#         desde aqui se itera sobre cada i y hay que variar los vectores
#         coordenada
#         Regresa la matriz gigante (matriz simetrica del articulo)"""
#         # primer renglon
#         R11R22R33 = (R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
#         R23_R32 = (R_ij(1, 2, a1=i, a2=j) - R_ij(2, 1, a1=i, a2=j))
#         R31_R13 = (R_ij(2, 0, a1=i, a2=j) - R_ij(0, 2, a1=i, a2=j))
#         R12_R21 = (R_ij(0, 1, a1=i, a2=j) - R_ij(1, 0, a1=i, a2=j))
#         # segundo renglon
#         R11_R22_R33 = (R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
#         R12R21 = (R_ij(0, 1, a1=i, a2=j) + R_ij(1, 0, a1=i, a2=j))
#         R13R31 = (R_ij(0, 2, a1=i, a2=j) + R_ij(2, 0, a1=i, a2=j))
#         # tercer renglon
#         _R11R22_R33 = (-R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
#         R23R32 = (R_ij(1, 2, a1=i, a2=j) + R_ij(2, 1, a1=0, a2=0))
#         # cuarto renglon
#         _R11_R22R33 = (-R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
#
#         matriz_R = [
#             [R11R22R33, R23_R32, R31_R13, R12_R21],
#             [R23_R32, R11_R22_R33, R12R21, R13R31],
#             [R31_R13, R12R21, _R11R22_R33, R23R32],
#             [R12_R21, R13R31, R23R32, _R11_R22R33]
#         ]
#         return (matriz_R)
#
#     def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
#         """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
#         y el atomo a comparar"""
#
#         dim_coord = clique_trasladado_rotado.shape[1]
#         N = clique_trasladado_rotado.shape[0]
#         result = 0.0
#         for v, w in zip(atom_to_compare, clique_trasladado_rotado):
#             result += sum([(v[i] - w[i]) ** 2.0 for i in range(dim_coord)])
#
#         rmsd_final = np.sqrt(result / N)
#         return rmsd_final
#
#     matriz_R = matrix_R(res1, res2)
#     matriz_rotacion = rotation_matrix(matriz_R)
#     idx_vectores_gorro = num_cliques + 2
#     vector_rotado = rotation_vectors(array_cliques1[:, idx_vectores_gorro][res1], matriz_rotacion) # antes 14 #vectores gorro
#     vector_rotado_trasladado_a_clique2 = vector_rotado + np.array(array_cliques2[:, 1][res2], dtype=np.float64)  # xrot + baricentro a mover #antes 10
#     rmsd_final = rmsd_between_cliques(vector_rotado_trasladado_a_clique2, np.array(
#         array_cliques2[:, 0][res2], dtype=np.float64))  # antes 9 #PROBABLE CAMBIAR
#     # clique rotado y trasladado vs clique coordenadas
#
#     restriccion_rmsd = 0.15
#     if num_cliques == 4:
#         restriccion_rmsd = 0.30
#     if num_cliques == 5:
#         restriccion_rmsd = 0.60
#     if num_cliques == 6:
#         restriccion_rmsd = 0.90
#     if num_cliques == 7:
#         restriccion_rmsd = 1.50
#     if num_cliques == 8:
#         restriccion_rmsd = 1.80
#
#     if rmsd_final <= restriccion_rmsd:
#
#         return(rmsd_final,(res1,res2), matriz_rotacion)
#
#     return(rmsd_final,(res1,res2), matriz_rotacion)



# las funciones de abajo son para graficar
#
# def get_rot_vector(residuos, array_cliques1, array_cliques2, num_cliques):
#
#     res1, res2 = residuos
#
#     def R_ij(i, j, a1=0, a2=0):
#         """Recuerda que 0-->1,1-->2,2-->2 en los indices de R
#         a1,a2 corresponden a que atomo quieren que se compare
#         """
#         idx_vec_gorro1, idx_vec_gorro2 = 2, 2 + num_cliques
#         valor = sum(  # VECTORES GORRO AGARRARLOS!!!
#          [array_cliques1[:, k][a1][i] * array_cliques2[:, k][a2][j] for k in range(idx_vec_gorro1, idx_vec_gorro2)])
#         return (valor)
#
#     def matrix_R(i, j):
#         # CAMBIO (2018/10/24) se dejo de implementar el diccionario de R_ij y se cambiaron los indices
#         # para regresar al calculo del carticulo sumar 1 a los numeros de la matriz R
#         """cliques a comparar: i,j
#         desde aqui se itera sobre cada i y hay que variar los vectores
#         coordenada
#         Regresa la matriz gigante (matriz simetrica del articulo)"""
#         # primer renglon
#         R11R22R33 = (R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
#         R23_R32 = (R_ij(1, 2, a1=i, a2=j) - R_ij(2, 1, a1=i, a2=j))
#         R31_R13 = (R_ij(2, 0, a1=i, a2=j) - R_ij(0, 2, a1=i, a2=j))
#         R12_R21 = (R_ij(0, 1, a1=i, a2=j) - R_ij(1, 0, a1=i, a2=j))
#         # segundo renglon
#         R11_R22_R33 = (R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
#         R12R21 = (R_ij(0, 1, a1=i, a2=j) + R_ij(1, 0, a1=i, a2=j))
#         R13R31 = (R_ij(0, 2, a1=i, a2=j) + R_ij(2, 0, a1=i, a2=j))
#         # tercer renglon
#         _R11R22_R33 = (-R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
#         R23R32 = (R_ij(1, 2, a1=i, a2=j) + R_ij(2, 1, a1=0, a2=0))
#         # cuarto renglon
#         _R11_R22R33 = (-R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
#
#         matriz_R = [
#             [R11R22R33, R23_R32, R31_R13, R12_R21],
#             [R23_R32, R11_R22_R33, R12R21, R13R31],
#             [R31_R13, R12R21, _R11R22_R33, R23R32],
#             [R12_R21, R13R31, R23R32, _R11_R22R33]
#         ]
#         return (matriz_R)
#
#     def rotation_matrix(matriz_R):
#         """utilizando la funcion giant_matrix, fijando los valores de i,j
#         se calcula la matriz de rotacion con los eigenvectores y eigenvalores
#         arroja una matriz de rotacion que depende de la matriz gigante
#         """
#         eignvalues, eigenvectors = np.linalg.eig(matriz_R)
#         q = eigenvectors[:, np.argmax(eignvalues)]
#         q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
#         # matriz de rotacion con eigenvectores
#         matriz_rotacion = np.array([
#             [(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
#             [2 * (q1 * q2 + q0 * q3), (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
#             [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)]
#         ], dtype=np.float64)
#         return (matriz_rotacion)
#
#     def rotation_vectors(vector_gorro, matriz_rotacion):
#         """obtencion de vector rotado,
#         utilizando la matriz de rotacion
#         y los vectores gorro a rotar y trasladar"""
#         coord_rotado_trasladado = [np.matmul(
#             matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]
#
#         return (coord_rotado_trasladado)
#
#     matriz_R = matrix_R(res1, res2)
#     matriz_rotacion = rotation_matrix(matriz_R)
#     idx_vectores_gorro = num_cliques + 2
#     vector_rotado = rotation_vectors(array_cliques1[:, idx_vectores_gorro][res1], matriz_rotacion)
#
#     return vector_rotado
