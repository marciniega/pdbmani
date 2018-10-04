
# coding: utf-8

# # Algoritmo Click
# ## Generacion de cliques
# + Se generan con biopandas para obtener los atomos de  CαCα  y sus coordenadas.
# + Se calcula la distancia y se genera un grafo completo con la distancia entre cada par de atomos.
# + Se restringen los enlaces por una distancia dada y se generan los cliques que tengas un numero k de elementos para pertencer al clique.
# + Una ves generados los cliques de cada proteina se extraen sus coordenadas para poderlas comparar

# In[1]:


#libreria de analisis de datos y una caracterizacion para su facil lectura.
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 40)
pd.set_option('display.max_colwidth', -1)
#libreria de generacion de rede y cliques
import networkx as nx,community

#libreria de visualizacion de datos y un formato dado
import matplotlib.pyplot as plt
plt.style.use('ggplot')
font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams[u'figure.figsize'] = (16,8)

#mas librerias que voy obteniendo
import biopandas.pdb as bp
biop = bp.PandasPdb() #libreria de lectura de pdbs

#libreria de calculo de distancia euclidiana
from scipy.spatial.distance import pdist, squareform

#libreria de mate
import numpy as np

#libreria de iteraciones
import itertools as it

#Libreria de MA para RMSD
import sys
sys.path.append('math_tricks/')
import math_vect_tools as mvt

# #Libreria de graficacion interactiva
# import plotly.plotly as py
# import plotly.graph_objs as go
#
# #Para optimizar el codigo para hacer mas rapido los calculos
# get_ipython().run_line_magic('load_ext', 'Cython')


# In[2]:


# Aqui se cambiaria por los archivos a leer pdbs sin modificar
path1 ='C://Users/serch/pdbmani/Serch/1xxa.pdb'
path2 ='C://Users/serch/pdbmani/Serch/1tig.pdb'

#funcion de lectura con biopandas
def read_biopdb(path):
    """Extrae las cordenadas de los atomos de C_alfa y los acomoda en un vector
    devuelve un dataframe con las coordenadas y el numero de residuo"""
    df = biop.read_pdb(path)
    df_atom = df.df['ATOM']
    #OJO AQUI ESTA ADECUADO AL PDB   para elegir solo un frame en trj_0 y trj_0_A [:1805]
    df_ca = df_atom[df_atom.atom_name == 'CA'][[
    'atom_number','atom_name','residue_name','residue_number',
    'x_coord','y_coord','z_coord']]
    columna_vector = []
    for i in zip(df_ca.x_coord.tolist(),df_ca.y_coord.tolist(),df_ca.z_coord.tolist()):
        columna_vector.append(np.array(i))

    df_ca['vector'] = columna_vector
    return(df_ca)


# In[3]:


#lectura de pdbs
df_ca1 = read_biopdb(path1)
df_ca2 = read_biopdb(path2)


# In[4]:


#se calcula la distancia entre cada par de nodos.
def distancia_entre_atomos(df_ca):
    """df_ca: Dataframe con coordenadas de los atomos alfa, devuelve otro DataFrame
    df_da: Dataframe como una matriz de adyacencias donde el valor es la distancia"""
    distancias = []
    #se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
    for v,i in zip(df_ca.vector,df_ca.atom_number):
        distancia_un_atomo = []
        for av,j in zip(df_ca.vector,df_ca.atom_number):
            distancia = pdist([v,av],metric='euclidean').item()
            distancia_un_atomo.append(distancia)
        distancias.append(distancia_un_atomo)
    #se genera la matriz de adyacencias para la red
    df_da = pd.DataFrame(index=df_ca.atom_number,columns=df_ca.atom_number,data=distancias)
    return(df_da)


# In[5]:


#generacion de matriz de adyacencias
df_da1 = distancia_entre_atomos(df_ca1)
df_da2 = distancia_entre_atomos(df_ca2)
#podriamos solo mantener la matriz diagonal y dejarla como un array de arrays


# In[6]:


def gen_3_cliques(df_da, dth = 10, k=3):
    """Genera n-cliques de dataframe de distancias, tomando en cuenta los enlaces menores o iguales
    a dth y forma los k-cliques que elijas 
    valores por default:
    dth=10, k=3"""
    #red de distancias completa
    red = nx.from_pandas_adjacency(df_da)
    print("red antes de filtros:",nx.info(red))

    #filtro de distancias
    edgesstrong = [(u,v) for (u,v,d) in red.edges(data=True) if d["weight"] <= dth]

    red = nx.Graph(edgesstrong)
    print("=="*20)
    print("red despues de filtros:",nx.info(red))

    n_cliques = [clq for clq in nx.find_cliques(red) if len(clq) >=k]
    print('numero de cliques maximos encontrados:',len(n_cliques))

    lista_cliques = []
    for i,v in enumerate(n_cliques):
        a = list(it.combinations(v,k))
        for j in a:
            if set(j) not in lista_cliques:
                #recuerda que para comparar elementos utiliza set, y apilalos como set
                lista_cliques.append(set(j))

    df_lc = pd.DataFrame(lista_cliques)            
    print("numero de %s-cliques posibles:" % (k), df_lc.shape[0])
    return(df_lc)


# In[7]:


df_lc1 = gen_3_cliques(df_da1,dth = 10, k=3)
print('--'*59)
df_lc2 = gen_3_cliques(df_da2,dth = 10, k=3)


# In[8]:


#funcion para obtener las coordenadas del clique
def get_coord_clique(df_ca,df_lc):
    """df_ca:DataFrame con coordenadas de carbonos alfa,
    df_lc:Dataframe con cliques, si coincide el numero del atomo
    le pega su coordenada y genera una matriz de vectores que contiene 
    las coordenadas de cada atomo ordenado de izquierda a derecha como 
    aparecen en df_lc"""
    lista_matriz_coordendas = [] #lista para apilar las coordenadas
    x = []
    y = []
    z = []

    for i in df_lc.index:
        #si coincide el numero de atomo con el numero de atomo del clique le coloca el vector de coordenadas
        x_temp = np.array(df_ca[df_ca.atom_number==df_lc.iloc[i,0]].vector.values[0])
        y_temp = np.array(df_ca[df_ca.atom_number==df_lc.iloc[i,1]].vector.values[0])
        z_temp = np.array(df_ca[df_ca.atom_number==df_lc.iloc[i,2]].vector.values[0])
        mat_dist = [x_temp,y_temp,z_temp]

        x.append(x_temp)
        y.append(y_temp)
        z.append(z_temp)
        lista_matriz_coordendas.append(mat_dist)

    df_lc['coord_clique_0'] = x
    df_lc['coord_clique_1'] = y
    df_lc['coord_clique_2'] = z
    df_lc['matriz_coordenadas'] = lista_matriz_coordendas #columna con coordenadas del clique
    return(df_lc)


# In[9]:


#pegado de coordendas
df_lc1 = get_coord_clique(df_ca1,df_lc1)
df_lc2 = get_coord_clique(df_ca2,df_lc2)


# ## Comparacion de cliques
# ### Pasos para comparar
# Para obtener el __RMSD__ es necesario primero rotar y trasladar un atomo con respecto al atomo a comparar (de la otra proteina) y calcular el __RMSD__.
# 
# Siguiendo al metodologia en *Using quaternions to calculate RMSD*.
# Se generan las funciones de traslado y rotacion.
# 
# __Segunda parte esto va mas abajo__
# 
# Para obtener C, $\alpha$, $\beta$ con:
#    + $\Phi$
#    + $\Psi$
# 1. Matriz de comparacion de Estructura Secundaria (SSM)
# 2. Solvente Accesible (SAM)

# ### Traslacion
# Se calcula el baricentro de cada clique en ambas moleculas y se generan nuevos vectores que van del baricentro al atomo llamados $\hat{x}$.
# 
# El baricentro se calcula como $\bar{x} =$($\frac{(x_1 + x_2 + x_3)}{3}$,$\frac{(y_1 + y_2 + y_3)}{3}$,$\frac{(z_1 + z_2 + z_3)}{3}$)
# 
# $\hat{x} = x_k - \bar{x}$

# In[10]:


# funcion de calculo de baricentro
def baricenter_clique(df_lc):
    """se calcula el baricentro de cada clique 
    siguiendo la formula de arriba.
    df_lc: Dataframe con los cliques y coordenadas
    regresa
    df_lc:Dataframe con el baricentro de ese clique"""
    coord_center = []
    for i in range(df_lc.shape[0]):
        #se extrae las coordenadas de los atomos
        A = df_lc.coord_clique_0[i]
        B = df_lc.coord_clique_1[i]
        C = df_lc.coord_clique_2[i]
        #se calcula el punto promedio
        x1 = round((A[0]+B[0]+C[0])/3,5)
        y1 = round((A[1]+B[1]+C[1])/3,5)
        z1 = round((A[2]+B[2]+C[2])/3,5)
        #se apila para pegarlo en una sola fila correspondiente al clique
        coord_center.append(np.array([x1,y1,z1]))

    #generacion de la columna
    df_lc['baricentro_clique'] = coord_center
    return(df_lc)


# In[11]:


#calculo de baricentro
df_lc1 = baricenter_clique(df_lc1)
df_lc2 = baricenter_clique(df_lc2)


# In[12]:


def center_vectors(df_lc):
    """Calculo de los vectores gorro que van del baricentro 
    a la coordenada del atomo
    df_lc: Dataframe con baricentro y coordenadas de cada clique
    regresa
    df_lc:Dataframe con vectores gorro en otra columna"""
    vec1 = []
    vec2 = []
    vec3 = []
    vectores_centricos = []
    for i,val in enumerate(df_lc.baricentro_clique):
    #     extraccion de coordenadas de cada atomo
        A = df_lc.coord_clique_0[i]
        B = df_lc.coord_clique_1[i]
        C = df_lc.coord_clique_2[i]
        #calculo de vectores DEL CENTRO AL PUNTO COORDENADA
        vec_a = np.round(list(A - val),6)
        vec_b = np.round(list(B - val),6)
        vec_c = np.round(list(C - val),6)
    #SE APILAN PARA QUE ESTEN EN EL MISMO CLIQUE CORRESPONDIENTE A CADA UNO.
        vec1.append(vec_a)
        vec2.append(vec_b)
        vec3.append(vec_c)
        vectores_centricos.append(np.array([vec_a,vec_b,vec_c]))
    #se generan la columna de cada vector correspondiente a cada atomo
    df_lc['vec_gorro_0'] = vec1
    df_lc['vec_gorro_1'] = vec2
    df_lc['vec_gorro_2'] = vec3
    df_lc['vectores_gorro'] = vectores_centricos
    return(df_lc)


# In[13]:


#generacion de vectores gorro
df_lc1 = center_vectors(df_lc1)
df_lc2 = center_vectors(df_lc2)


# In[14]:


df_lc1.head(1)


# ### Rotacion
# Para generar la rotacion tenemos que generar la *matriz gigante* que depende de los elemento de la matriz de correlacion $R_{ij}$
# 
# Donde $R_{ij} = \sum\limits_{k=1}^N{x_{ik}y_{jk}}, i,j = 1,2,3$
# 
# Posteriormente se calculan los eigenvalores y eigenvectores de esta matriz gigante
# Para obtener los quaterniones y generar la matriz de rotacion y con ella calcular el vector rotado
# 
# Por ultimo, se suma al vector rotado y trasladado se suma el baricentro del clique a comparar y se calcula el RMSD

# In[15]:


# POR OPTIMIZAR CODIGO !!!
prueba1 = df_lc1.values
prueba2 = df_lc2.values

# In[16]:

# funcion para obtener los valores de la prerotacion, de los valores de la matriz de correlaciones
# en check por que se utiliza vector gorro en lugar de posiciones iniciales
# el articulo no dice...
def R_ij(i, j, a1=0, a2=0):
    """Recuerda que 0-->1,1-->2,2-->2 en los indices de R
    a1,a2 corresponden a que atomo quieren que se compare
    """
    # se genera un diccionario para asignar los valores como en el articulo
    # y no tener equivocaciones
    dict_convencion = {1: 0, 2: 1, 3: 2}

    i = dict_convencion.get(i)
    j = dict_convencion.get(j)

    values = []
    for k in [8, 9, 10]:  # 8,9,10 corresponde a la columna de vec_gorro_0,_1,_2
        atom_value1 = prueba1[:, k][a1][i]
        atom_value2 = prueba2[:, k][a2][j]
        value = atom_value1 * atom_value2
        values.append(value)
        valor = sum(values)

    return (valor)


# In[17]:


def giant_matrix(i,j):
    """cliques a comparar: i,j
    desde aqui se itera sobre cada i y hay que variar los vectores 
    coordenada 
    Regresa la matriz gigante (matriz simetrica del articulo)"""
    #primer renglon
    R11R22R33 = (R_ij(1,1,a1=i,a2=j) + R_ij(2,2,a1=i,a2=j) + R_ij(3,3,a1=i,a2=j))
    R23_R32 = (R_ij(2,3,a1=i,a2=j) - R_ij(3,2,a1=i,a2=j))
    R31_R13 = (R_ij(3,1,a1=i,a2=j) - R_ij(1,3,a1=i,a2=j))
    R12_R21 = (R_ij(1,2,a1=i,a2=j) - R_ij(2,1,a1=i,a2=j))
    #segundo renglon
    R11_R22_R33 = (R_ij(1,1,a1=i,a2=j) - R_ij(2,2,a1=i,a2=j) - R_ij(3,3,a1=i,a2=j))
    R12R21 = (R_ij(1,2,a1=i,a2=j) + R_ij(2,1,a1=i,a2=j))
    R13R31 = (R_ij(1,3,a1=i,a2=j) + R_ij(3,1,a1=i,a2=j))
    #tercer renglon
    _R11R22_R33 = (-R_ij(1,1,a1=i,a2=j) + R_ij(2,2,a1=i,a2=j) - R_ij(3,3,a1=i,a2=j))
    R23R32 = (R_ij(2,3,a1=i,a2=j) + R_ij(3,2,a1=0,a2=0))
    #cuarto renglon
    _R11_R22R33 = (-R_ij(1,1,a1=i,a2=j) - R_ij(2,2,a1=i,a2=j) + R_ij(3,3,a1=i,a2=j))

    matriz_gigante = np.round([
        [R11R22R33, R23_R32 , R31_R13, R12_R21],
        [R23_R32, R11_R22_R33, R12R21, R13R31],
        [R31_R13, R12R21, _R11R22_R33, R23R32],
        [R12_R21, R13R31, R23R32, _R11_R22R33]
    ],4)
    return(matriz_gigante)


# In[18]:


def rotation_matrix(matriz_gigante):
    """utilizando la funcion giant_matrix, fijando los valores de i,j
    se calcula la matriz de rotacion con los eigenvectores y eigenvalores
    arroja una matriz de rotacion que depende de la matriz gigante
    """
    eignvalues,eigenvectors = np.linalg.eig(matriz_gigante,)
    q = eigenvectors[:,np.argmax(eignvalues)]
    q0,q1,q2,q3 = q[0],q[1],q[2],q[3]
    #matriz de rotacion con eigenvectores
    mat_rot = np.array([
                [(q0**2+q1**2-q2**2-q3**2), 2*(q1*q2-q0*q3),2*(q1*q3+q0*q2)],
                [2*(q1*q2+q0*q3), (q0**2-q1**2+q2**2-q3**2),2*(q2*q3-q0*q1)],
                [2*(q1*q3-q0*q2),2*(q2*q3+q0*q1), (q0**2-q1**2-q2**2+q3**2)]
    ])
    return(mat_rot)


# In[19]:


def rotation_vectors(vector_gorro,mat_rot):
    """obtencion de vector rotado,
    utilizando la matriz de rotacion 
    y los vectores gorro a rotar y trasladar"""
    #multiplicacion de matrices de cada vector rotado
    vec1 = vector_gorro
    coord_rot_tras = []
    for i in vec1:
        coord_rot_tras.append(np.matmul(mat_rot,i.reshape(3,1)).T[0])

    x_rot = coord_rot_tras
    return(x_rot)


# In[20]:


def rmsd_between_cliques(atom_trans_rot,atom_to_compare):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar, por el momento solo imprime el resultado"""
    # primer RMSD entre atomos
    a = atom_trans_rot
    b = np.array(atom_to_compare)

    p12 = np.sum((b-a)**2,1)
    rmsd_i = lambda i: np.sqrt(i)/3
    rmsd_final = rmsd_i(p12).mean()
    
    # if rmsd_final <= 0.15:
    #     print('RMSD_final:', rmsd_final,a,b)


# In[21]:


def calculate_rmsd_rot_trans(atom1, atom2):
    matriz_gigante = giant_matrix(atom1,atom2)
    mat_rot = rotation_matrix(matriz_gigante)
    x_rot = rotation_vectors(df_lc1.vectores_gorro[atom1],mat_rot)
    coord_rot_clique_2 = x_rot + np.array(df_lc2.baricentro_clique[atom2])
    rmsd_between_cliques(coord_rot_clique_2,np.array(df_lc2.matriz_coordenadas[atom2]))


# In[30]:


#!/usr/bin/env python3

# import multiprocessing
# import time
# import random
#
# def hello(n):
#     if __name__ == "__main__":
#         time.sleep(random.randint(1,3))
#         print("[{0}] Hello!".format(n))
#
# processes = [ ]
# for i in range(10):
#     t = multiprocessing.Process(target=hello, args=(i,))
#     processes.append(t)
#     t.start()
#
# for one_process in processes:
#     one_process.join()
#
# print("Done!")


# In[27]:


# get_ipython().run_cell_magic('timeit', '', 'import multiprocessing\nprocesses = [ ]\nfor i in range(df_lc2.shape[0]): \n    t = multiprocessing.Process(target=calculate_rmsd_rot_trans, args=(1,i))\n    processes.append(t)\n    t.start()\n    \nfor one_process in processes:\n    one_process.join()')
# import multiprocessing as mp
# processes = [ ]
# for i in range(df_lc2.shape[0]):
#     t = multiprocessing.Process(target=calculate_rmsd_rot_trans, args=(1,i))
#     processes.append(t)
#     t.start()
#
# for one_process in processes:
#     one_process.join()

# from datetime import datetime
# print(datetime.now())
# pool = mp.Pool(processes=mp.cpu_count() - 1)
# results = pool.map()
#     # calculate_rmsd_rot_trans,    args=(1,i)) for i in range(df_lc2.shape[0])
# pool.close()
# pool.join()
# print(datetime.now(), results)

# In[23]:7

# get_ipython().run_cell_magic('timeit', '', 'f')
# for i in range(df_lc2.shape[0]):
#     calculate_rmsd_rot_trans(1,i)


# In[ ]:


# %%time
# from datetime import datetime
# start = datetime.now()
# for j in range(df_lc1.shape[0]):
#     print('--'*20,j,'--'*20)
#     for i in range(df_lc2.shape[0]):
#         calculate_rmsd_rot_trans(j,i)
#
# end = datetime.now() - start
# print(end)

