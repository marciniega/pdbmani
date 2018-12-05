
# coding: utf-8


# libreria de analisis de datos y una caracterizacion para su facil lectura.
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 40)
pd.set_option('display.max_colwidth', -1)
# libreria de generacion de rede y cliques
import networkx as nx,community

# libreria para calcular la distancia minima del filtro
from scipy.spatial import distance

# mas librerias que voy obteniendo
import biopandas.pdb as bp
biop = bp.PandasPdb() #libreria de lectura de pdbs

# libreria de calculo de distancia euclidiana
from scipy.spatial.distance import pdist, squareform

# libreria de mate
import numpy as np

# libreria de iteraciones
import itertools as it

# Libreria de MA para RMSD
import sys
sys.path.append('math_tricks/')
import math_vect_tools as mvt

#libreria para correr dssp desde bash
import subprocess as sp

#libreria para Parsear DSSP FIles
import DSSPData as dd


# Aqui se cambiaria por los archivos a leer pdbs sin modificar
# path1 = '1xxa.pdb'
# path2 = '1tig.pdb'

# funcion de lectura con biopandas
def read_biopdb(path):
    """Extrae las cordenadas de los atomos de C_alfa y los acomoda en un vector
    devuelve un dataframe con las coordenadas y el numero de residuo"""
    df = biop.read_pdb(path)
    df_all_atoms = df.df['ATOM']
    #OJO AQUI ESTA ADECUADO AL PDB   para elegir solo un frame en trj_0 y trj_0_A [:1805]
    df_atoms = df_all_atoms[[
        'atom_number', 'atom_name', 'residue_name', 'residue_number',
        'x_coord', 'y_coord', 'z_coord'
    ]]
    columna_vector = []
    for i in zip(df_atoms.x_coord.tolist(), df_atoms.y_coord.tolist(),
                 df_atoms.z_coord.tolist()):
        columna_vector.append(np.array(i))

    df_atoms['vector_coordenadas'] = columna_vector
    return (df_atoms)


# In[8]:


# se calcula la distancia entre cada par de nodos.
def distancia_entre_atomos(df_atoms):
    """df_ca: Dataframe con coordenadas de los atomos alfa, devuelve otro DataFrame
    df_da: Dataframe como una matriz de adyacencias donde el valor es la distancia"""
    df_atoms_ca = df_atoms[df_atoms.atom_name == 'CA']
    distancias = []
    # se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
    for v,i in zip(df_atoms_ca.vector_coordenadas,df_atoms_ca.atom_number):
        distancia_un_atomo = []
        for av,j in zip(df_atoms_ca.vector_coordenadas,df_atoms_ca.atom_number):
            distancia = pdist(np.array([v,av]), metric='euclidean').item()
            distancia_un_atomo.append(distancia)
        distancias.append(distancia_un_atomo)
    # se genera la matriz de adyacencias para la red
    df_distancias = pd.DataFrame(index=df_atoms_ca.atom_number,
                         columns=df_atoms_ca.atom_number,
                         data=distancias)
    return(df_distancias)


def gen_3_cliques(df_distancias, dth = 10, k=3):
    """Genera n-cliques de dataframe de distancias, tomando en cuenta los enlaces menores o iguales
    a dth y forma los k-cliques que elijas 
    valores por default:
    dth=10, k=3"""
    # red de distancias completa
    red = nx.from_pandas_adjacency(df_distancias)
#     print("red antes de filtros:",nx.info(red))

    # filtro de distancias
    edgesstrong = [(u,v) for (u,v,d) in red.edges(data=True) if d["weight"] <= dth]

    red = nx.Graph(edgesstrong)
#     print("=="*20)
#     print("red despues de filtros:",nx.info(red))

    cliques_completos = [clq for clq in nx.find_cliques(red) if len(clq) >=k]
    print('numero de cliques maximos encontrados:',len(cliques_completos))
#     print(n_cliques)

    lista_cliques = []
    for i,v in enumerate(cliques_completos):
        a = list(it.combinations(v,k))
        for j in a:
            if set(j) not in lista_cliques:
                #recuerda que para comparar elementos utiliza set, y apilalos como set
                lista_cliques.append(set(j))

    df_cliques = pd.DataFrame(lista_cliques)
    print("numero de %s-cliques posibles:" % (k), df_cliques.shape[0])
    return(df_cliques, cliques_completos)


def mini_dssp(path,index):
    # ejecuto dssp desde bash y guardo archivo como output.log
    sp.run(['dssp','-i',path,'-o','output.log'])
    # parseo el dssp file
    dd_ob = dd.DSSPData()
    dssp_file_name = open('output.log')
    dd_ob.parseDSSP( 'output.log' )
    # obtengo la estructura y la guardo, posible no es necesario los residuos
    # solo el numero de atomo que le pego arbitrariamente REVISAR si esta bien
    ss = [i[2] for i in dd_ob.struct ]
    ss = pd.DataFrame([i for i in zip(ss,dd_ob.resnum,dd_ob.getChainType())])
    ss.columns = ['pre_ss','residue_number','chain']
    ss = ss[ss.chain == 'A']
    ss = ss[ss.residue_number != ''].reset_index(drop=True)
    ss['atom_number'] = index
    # catalogo  Yo tomo B y E como betas, G H I como alfa y lo demás como coil
    # B - betas
    # H - alfas
    ss['structure'] = np.where(ss.pre_ss.isin(['B','E']),'B',
                               np.where(ss.pre_ss.isin(['G','H','I']),'H',
                                        'C'))
    # checks
    print(ss.structure.value_counts(normalize = True) * 100)
    print(path)
    return(ss)


# funcion para obtener las coordenadas del clique
def paste_SS(ss, df_cliques, num_cliques=3):
    """ Obtenemos la estructura secundaria y la pega al dataframe generando el df s
    """
    #lista para apilar las estructuras
    if num_cliques >= 3:
        c1 = [ss[ss.residue_number == df_cliques.iloc[i, 0]].structure.values[0] for i in df_cliques.index]
        c2 = [ss[ss.residue_number == df_cliques.iloc[i, 1]].structure.values[0] for i in df_cliques.index]
        c3 = [ss[ss.residue_number == df_cliques.iloc[i, 2]].structure.values[0] for i in df_cliques.index]

        df_cliques['ss_0'] = c1
        df_cliques['ss_1'] = c2
        df_cliques['ss_2'] = c3

    if num_cliques >= 4:
        c4 = [ss[ss.residue_number == df_cliques.iloc[i, 3]].structure.values[0] for i in df_cliques.index]
        df_cliques['ss_3'] = c4

    if num_cliques >= 5:
        c5 = [ss[ss.residue_number == df_cliques.iloc[i, 4]].structure.values[0] for i in df_cliques.index]
        df_cliques['ss_4'] = c5

    if num_cliques >= 6:
        c6 = [ss[ss.residue_number == df_cliques.iloc[i, 5]].structure.values[0] for i in df_cliques.index]
        df_cliques['ss_5'] = c6

    if num_cliques >= 7:
        c7 = [ss[ss.residue_number == df_cliques.iloc[i, 6]].structure.values[0] for i in df_cliques.index]
        df_cliques['ss_6'] = c7

    if num_cliques >= 8:
        c8 = [ss[ss.residue_number == df_cliques.iloc[i, 7]].structure.values[0] for i in df_cliques.index]
        df_cliques['ss_7'] = c8


    #columna con coordenadas del clique
    return(df_cliques)


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
    def get_score_from_table(ss1,ss2):

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


#funcion para obtener las coordenadas del clique
def get_coords_clique(df_atoms_ca, df_cliques, num_cliques):
    """df_ca:DataFrame con coordenadas de carbonos alfa,
    df_lc:Dataframe con cliques, si coincide el numero del atomo
    le pega su coordenada y genera una matriz de vectores que contiene 
    las coordenadas de cada atomo ordenado de izquierda a derecha como 
    aparecen en df_clique"""
    if num_cliques >= 3:
        c0 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 0]].vector.values[0]) for i in
              df_cliques.index]
        c1 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 1]].vector.values[0]) for i in
              df_cliques.index]
        c2 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 2]].vector.values[0]) for i in
              df_cliques.index]

        df_cliques['coord_clique_0'] = c0
        df_cliques['coord_clique_1'] = c1
        df_cliques['coord_clique_2'] = c2

        if num_cliques == 3:
            df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i]] for i in df_cliques.index]

    if num_cliques >= 4:
        c3 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 3]].vector.values[0]) for i in
              df_cliques.index]

        df_cliques['coord_clique_3'] = c3
        if num_cliques == 4:
            df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i]] for i in df_cliques.index]

    if num_cliques >= 5:
        c4 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 4]].vector.values[0]) for i in
              df_cliques.index]

        df_cliques['coord_clique_4'] = c4
        if num_cliques == 5:
            df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i],c4[i]] for i in df_cliques.index]

    if num_cliques >= 6:
        c5 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 5]].vector.values[0]) for i in
              df_cliques.index]

        df_cliques['coord_clique_5'] = c5
        if num_cliques == 6:
            df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i], c5[i]] for i in df_cliques.index]

    if num_cliques >= 7:
        c6 = [np.array(df_atoms_ca[df_atoms_ca.atom_number == df_cliques.iloc[i, 6]].vector.values[0]) for i in
              df_cliques.index]

        df_cliques['coord_clique_6'] = c6
        if num_cliques == 7:
            df_cliques['matriz_coordenadas'] = [[c0[i], c1[i], c2[i], c3[i], c4[i], c5[i], c6[i]] for i in df_cliques.index]

    return(df_cliques)


# funcion de calculo de baricentro
def baricenter_clique(df_cliques, num_cliques):
    """se calcula el baricentro de cada clique 
    siguiendo la formula de arriba.
    df_lc: Dataframe con los cliques y coordenadas
    regresa
    df_lc:Dataframe con el baricentro de ese clique"""

    coord_center = []
    for i in range(df_cliques.shape[0]):
        # se extrae las coordenadas de los atomos
        A = df_cliques.coord_clique_0[i]
        B = df_cliques.coord_clique_1[i]
        C = df_cliques.coord_clique_2[i]

        # X,Y,Z del baricentro, promediando por numero de cliques
        x1 = (A[0] + B[0] + C[0]) / num_cliques
        y1 = (A[1] + B[1] + C[1]) / num_cliques
        z1 = (A[2] + B[2] + C[2]) / num_cliques
        if num_cliques >= 4:
            D = df_cliques.coord_clique_3[i]
            x1 = (A[0] + B[0] + C[0] + D[0]) / num_cliques
            y1 = (A[1] + B[1] + C[1] + D[1]) / num_cliques
            z1 = (A[2] + B[2] + C[2] + D[2]) / num_cliques
        if num_cliques >= 5:
            E = df_cliques.coord_clique_4[i]
            x1 = (A[0] + B[0] + C[0] + D[0] + E[0]) / num_cliques
            y1 = (A[1] + B[1] + C[1] + D[1] + E[1]) / num_cliques
            z1 = (A[2] + B[2] + C[2] + D[2] + E[2]) / num_cliques
        if num_cliques >= 6:
            F = df_cliques.coord_clique_5[i]
            x1 = (A[0] + B[0] + C[0] + D[0] + E[0] + F[0]) / num_cliques
            y1 = (A[1] + B[1] + C[1] + D[1] + E[1] + F[1]) / num_cliques
            z1 = (A[2] + B[2] + C[2] + D[2] + E[2] + F[2]) / num_cliques
        if num_cliques >= 7:
            G = df_cliques.coord_clique_6[i]
            x1 = (A[0] + B[0] + C[0] + D[0] + E[0] + F[0] + G[0]) / num_cliques
            y1 = (A[1] + B[1] + C[1] + D[1] + E[1] + F[1] + G[1]) / num_cliques
            z1 = (A[2] + B[2] + C[2] + D[2] + E[2] + F[2] + G[2]) / num_cliques
        # se calcula el punto promedio

        # se apila para pegarlo en una sola fila correspondiente al clique
        coord_center.append(np.array([x1, y1, z1]))

    #generacion de la columna
    df_cliques['baricentro_clique'] = coord_center
    return(df_cliques)


def center_vectors(df_cliques, num_cliques):
    """Calculo de los vectores gorro que van del baricentro 
    a la coordenada del atomo
    df_lc: Dataframe con baricentro y coordenadas de cada clique
    regresa
    df_lc:Dataframe con vectores gorro en otra columna"""
    vec0 = []
    vec1 = []
    vec2 = []
    vec3 = []
    vec4 = []
    vec5 = []
    vec6 = []
    vectores_centricos = []
    for i, val in enumerate(df_cliques.baricentro_clique):
        # extraccion de coordenadas de cada atomo
        A = df_cliques.coord_clique_0[i]
        B = df_cliques.coord_clique_1[i]
        C = df_cliques.coord_clique_2[i]

        # calculo de vectores DEL CENTRO AL PUNTO COORDENADA
        vec_a = list(A - val)
        vec_b = list(B - val)
        vec_c = list(C - val)
    # SE APILAN PARA QUE ESTEN EN EL MISMO CLIQUE CORRESPONDIENTE A CADA UNO.
        vec0.append(vec_a)
        vec1.append(vec_b)
        vec2.append(vec_c)

        if num_cliques == 3:
            vectores_centricos.append(np.array([vec_a, vec_b, vec_c]))

        if num_cliques >= 4:
            D = df_cliques.coord_clique_3[i]
            vec_d = list(D - val)
            vec3.append(vec_d)
            if num_cliques == 4:
                vectores_centricos.append(np.array([vec_a, vec_b, vec_c, vec_d]))

        if num_cliques >= 5:
            E = df_cliques.coord_clique_4[i]
            vec_e = list(E - val)
            vec4.append(vec_e)
            if num_cliques == 5:
                vectores_centricos.append(np.array([vec_a, vec_b, vec_c, vec_d, vec_e]))

        if num_cliques >= 6:
            F = df_cliques.coord_clique_5[i]
            vec_f = list(F - val)
            vec5.append(vec_f)
            if num_cliques == 6:
                vectores_centricos.append(np.array([vec_a, vec_b, vec_c, vec_d, vec_e, vec_f]))

        if num_cliques >= 7:
            G = df_cliques.coord_clique_6[i]
            vec_g = list(G - val)
            vec6.append(vec_g)
            if num_cliques == 7:
                vectores_centricos.append(np.array([vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g]))

    # se generan la columna de cada vector correspondiente a cada atomo
    df_cliques['vec_gorro_0'] = vec0
    df_cliques['vec_gorro_1'] = vec1
    df_cliques['vec_gorro_2'] = vec2

    if num_cliques >= 4:
        df_cliques['vec_gorro_3'] = vec3
    if num_cliques >= 5:
        df_cliques['vec_gorro_4'] = vec4
    if num_cliques >= 6:
        df_cliques['vec_gorro_5'] = vec5
    if num_cliques >= 7:
        df_cliques['vec_gorro_6'] = vec6

    df_cliques['vectores_gorro'] = vectores_centricos
    return(df_cliques)


# baricentro de vectores gorro, todos estan en el origen.
# def baricenter_vectores_gorro(df_cliques, num_cliques):
#     """se calcula el baricentro de cada clique
#     siguiendo la formula de arriba.
#     df_lc: Dataframe con los cliques y coordenadas
#     regresa
#     df_lc:Dataframe con el baricentro de ese clique"""
#
#     coord_center = []
#     for i in range(df_cliques.shape[0]):
#         # se extrae las coordenadas de los atomos
#         A = df_cliques.vec_gorro_0[i]
#         B = df_cliques.vec_gorro_1[i]
#         C = df_cliques.vec_gorro_2[i]
#
#         # X,Y,Z del baricentro, promediando por numero de cliques
#         x1 = (A[0] + B[0] + C[0]) / num_cliques
#         y1 = (A[1] + B[1] + C[1]) / num_cliques
#         z1 = (A[2] + B[2] + C[2]) / num_cliques
#         if num_cliques >= 4:
#             D = df_cliques.vec_gorro_3[i]
#             x1 = (A[0] + B[0] + C[0] + D[0]) / num_cliques
#             y1 = (A[1] + B[1] + C[1] + D[1]) / num_cliques
#             z1 = (A[2] + B[2] + C[2] + D[2]) / num_cliques
#         if num_cliques >= 5:
#             E = df_cliques.vec_gorro_4[i]
#             x1 = (A[0] + B[0] + C[0] + D[0] + E[0]) / num_cliques
#             y1 = (A[1] + B[1] + C[1] + D[1] + E[1]) / num_cliques
#             z1 = (A[2] + B[2] + C[2] + D[2] + E[2]) / num_cliques
#         if num_cliques >= 6:
#             F = df_cliques.vec_gorro_5[i]
#             x1 = (A[0] + B[0] + C[0] + D[0] + E[0] + F[0]) / num_cliques
#             y1 = (A[1] + B[1] + C[1] + D[1] + E[1] + F[1]) / num_cliques
#             z1 = (A[2] + B[2] + C[2] + D[2] + E[2] + F[2]) / num_cliques
#         if num_cliques >= 7:
#             G = df_cliques.vec_gorro_6[i]
#             x1 = (A[0] + B[0] + C[0] + D[0] + E[0] + F[0] + G[0]) / num_cliques
#             y1 = (A[1] + B[1] + C[1] + D[1] + E[1] + F[1] + G[1]) / num_cliques
#             z1 = (A[2] + B[2] + C[2] + D[2] + E[2] + F[2] + G[2]) / num_cliques
#         # se calcula el punto promedio
#
#         # se apila para pegarlo en una sola fila correspondiente al clique
#         coord_center.append(np.array([x1, y1, z1]))
#
#     #generacion de la columna
#     df_cliques['baricentro_vectores_gorro'] = coord_center
#     return(df_cliques)


def calculate_rmsd_rot_trans_m(residuos, array_cliques1, array_cliques2, num_cliques):
    
    res1, res2 = residuos

    def R_ij(i, j, a1=0, a2=0):
        """Recuerda que 0-->1,1-->2,2-->2 en los indices de R
        a1,a2 corresponden a que atomo quieren que se compare
        """

        # SE QUITA DICCIONARIO PARA MAYOR VELOCIDAD (2018/10/24).
        # se genera un diccionario para asignar los valores como en el articulo
        # y no tener equivocaciones
        # dict_convencion = {1: 0, 2: 1, 3: 2}
        #
        # i = dict_convencion.get(i)
        # j = dict_convencion.get(j)

        # values = []
        # append = values.append
        # for k in [2, 3, 4]:  # 8,9,10 corresponde a la columna de vec_gorro_0,_1,_2 #antes 11,12,13
        #     # REVISAR VEC_GORRO
        #     atom_value1 = array_cliques1[:, k][a1][i]
        #     atom_value2 = array_cliques2[:, k][a2][j]
        #     append(atom_value1 * atom_value2)
        #
        # valor = sum(values)

        idx_vec_gorro1, idx_vec_gorro2 = 2, 2 + num_cliques
        valor = sum(  # VECTORES GORRO AGARRARLOS!!!
            [array_cliques1[:, k][a1][i] * array_cliques2[:, k][a2][j] for k in range(idx_vec_gorro1, idx_vec_gorro2)])
        return (valor)

    def matrix_R(i, j):
        # CAMBIO (2018/10/24) se dejo de implementar el diccionario de R_ij y se cambiaron los indices
        # para regresar al calculo del carticulo sumar 1 a los numeros de la matriz R
        """cliques a comparar: i,j
        desde aqui se itera sobre cada i y hay que variar los vectores
        coordenada
        Regresa la matriz gigante (matriz simetrica del articulo)"""
        # primer renglon
        R11R22R33 = (R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
        R23_R32 = (R_ij(1, 2, a1=i, a2=j) - R_ij(2, 1, a1=i, a2=j))
        R31_R13 = (R_ij(2, 0, a1=i, a2=j) - R_ij(0, 2, a1=i, a2=j))
        R12_R21 = (R_ij(0, 1, a1=i, a2=j) - R_ij(1, 0, a1=i, a2=j))
        # segundo renglon
        R11_R22_R33 = (R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
        R12R21 = (R_ij(0, 1, a1=i, a2=j) + R_ij(1, 0, a1=i, a2=j))
        R13R31 = (R_ij(0, 2, a1=i, a2=j) + R_ij(2, 0, a1=i, a2=j))
        # tercer renglon
        _R11R22_R33 = (-R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
        R23R32 = (R_ij(1, 2, a1=i, a2=j) + R_ij(2, 1, a1=0, a2=0))
        # cuarto renglon
        _R11_R22R33 = (-R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))

        matriz_R = [
            [R11R22R33, R23_R32, R31_R13, R12_R21],
            [R23_R32, R11_R22_R33, R12R21, R13R31],
            [R31_R13, R12R21, _R11R22_R33, R23R32],
            [R12_R21, R13R31, R23R32, _R11_R22R33]
        ]
        return (matriz_R)

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
        # multiplicacion de matrices de cada vector rotado
        # coord_rotado_trasladado = []
        # append = coord_rotado_trasladado.append
        # matmul = np.matmul
        # for i in vector_gorro:
        #     append(matmul(matriz_rotacion, i.reshape(3, 1)).T[0])

        coord_rotado_trasladado = [np.matmul(
            matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]
        return (coord_rotado_trasladado)

    def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
        """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
        y el atomo a comparar, por el momento solo imprime el resultado"""
        # primer RMSD entre atomos
        p12 = np.sum((np.array(
            atom_to_compare, dtype=np.float64) - clique_trasladado_rotado) ** 2, 1)
        rmsd_i = lambda i: np.sqrt(i) / 3
        rmsd_final = rmsd_i(p12).mean()

        return (rmsd_final)

    matriz_R = matrix_R(res1, res2)
    matriz_rotacion = rotation_matrix(matriz_R)
    idx_vectores_gorro = num_cliques + 2
    vector_rotado = rotation_vectors(array_cliques1[:, idx_vectores_gorro][res1], matriz_rotacion) # antes 14 #vectores gorro
    vector_rotado_trasladado_a_clique2 = vector_rotado + np.array(array_cliques2[:, 1][res2], dtype=np.float64)  # xrot + baricentro a mover #antes 10
    rmsd_final = rmsd_between_cliques(vector_rotado_trasladado_a_clique2, np.array(
        array_cliques2[:, 0][res2], dtype=np.float64)) # antes 9 #PROBABLE CAMBIAR
    # clique rotado y trasladado vs clique coordenadas

    restriccion_rmsd = 0.15
    if num_cliques == 4:
        restriccion_rmsd = 0.30
    if num_cliques == 5:
        restriccion_rmsd = 0.60
    if num_cliques == 7:
        restriccion_rmsd = 1.50
    if num_cliques == 8:
        restriccion_rmsd = 1.80
    if rmsd_final <= restriccion_rmsd:    
        return(rmsd_final,(res1,res2))
    
    return(rmsd_final,(res1,res2))


## reorganizacion de funciones

def get_rot_vector(residuos, array_cliques1, array_cliques2, num_cliques):

    res1, res2 = residuos

    def R_ij(i, j, a1=0, a2=0):
        """Recuerda que 0-->1,1-->2,2-->2 en los indices de R
        a1,a2 corresponden a que atomo quieren que se compare
        """
        idx_vec_gorro1, idx_vec_gorro2 = 2, 2 + num_cliques
        valor = sum(  # VECTORES GORRO AGARRARLOS!!!
            [array_cliques1[:, k][a1][i] * array_cliques2[:, k][a2][j] for k in range(idx_vec_gorro1, idx_vec_gorro2)])
        return (valor)

    def matrix_R(i, j):
        # CAMBIO (2018/10/24) se dejo de implementar el diccionario de R_ij y se cambiaron los indices
        # para regresar al calculo del carticulo sumar 1 a los numeros de la matriz R
        """cliques a comparar: i,j
        desde aqui se itera sobre cada i y hay que variar los vectores
        coordenada
        Regresa la matriz gigante (matriz simetrica del articulo)"""
        # primer renglon
        R11R22R33 = (R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))
        R23_R32 = (R_ij(1, 2, a1=i, a2=j) - R_ij(2, 1, a1=i, a2=j))
        R31_R13 = (R_ij(2, 0, a1=i, a2=j) - R_ij(0, 2, a1=i, a2=j))
        R12_R21 = (R_ij(0, 1, a1=i, a2=j) - R_ij(1, 0, a1=i, a2=j))
        # segundo renglon
        R11_R22_R33 = (R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
        R12R21 = (R_ij(0, 1, a1=i, a2=j) + R_ij(1, 0, a1=i, a2=j))
        R13R31 = (R_ij(0, 2, a1=i, a2=j) + R_ij(2, 0, a1=i, a2=j))
        # tercer renglon
        _R11R22_R33 = (-R_ij(0, 0, a1=i, a2=j) + R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j))
        R23R32 = (R_ij(1, 2, a1=i, a2=j) + R_ij(2, 1, a1=0, a2=0))
        # cuarto renglon
        _R11_R22R33 = (-R_ij(0, 0, a1=i, a2=j) - R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j))

        matriz_R = [
            [R11R22R33, R23_R32, R31_R13, R12_R21],
            [R23_R32, R11_R22_R33, R12R21, R13R31],
            [R31_R13, R12R21, _R11R22_R33, R23R32],
            [R12_R21, R13R31, R23R32, _R11_R22R33]
        ]
        return (matriz_R)

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
        coord_rotado_trasladado = [np.matmul(
            matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]

        return (coord_rotado_trasladado)

    matriz_R = matrix_R(res1, res2)
    matriz_rotacion = rotation_matrix(matriz_R)
    idx_vectores_gorro = num_cliques + 2
    vector_rotado = rotation_vectors(array_cliques1[:, idx_vectores_gorro][res1], matriz_rotacion)

    return(vector_rotado)

def get_distancia_promedio(num_cliques,df_cliques):
    a = (0, 0, 0)
    dist = distance.euclidean
    if num_cliques == 3:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2])]) for i in df_cliques.vectores_gorro]

    elif num_cliques == 4:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2]), dist(a, i[3])]) for i in df_cliques.vectores_gorro]

    elif num_cliques == 5:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2]), dist(a, i[3]), dist(a, i[4])]) for i in
            df_cliques.vectores_gorro]

    elif num_cliques == 6:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2]), dist(a, i[3]), dist(a, i[4]), dist(a, i[5])]) for i in
            df_cliques.vectores_gorro]

    elif num_cliques == 7:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2]), dist(a, i[3]), dist(a, i[4]), dist(a, i[5]), dist(a, i[6])])
            for i in df_cliques.vectores_gorro]

    elif num_cliques == 8:
        df_cliques['distancia_promedio'] = [np.mean(
            [dist(a, i[0]), dist(a, i[1]), dist(a, i[2]), dist(a, i[3]), dist(a, i[4]), dist(a, i[5]), dist(a, i[6]),
             dist(a, i[7])]) for i in df_cliques.vectores_gorro]

    else:
        print('No soportamos ese numero de cliques')
        exit()

    return(df_cliques)
