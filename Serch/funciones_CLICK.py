
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

#libreria para correr dssp desde bash
import subprocess as sp

#libreria para Parsear DSSP FIles
import DSSPData as dd

#Libreria de graficacion interactiva
# import plotly.plotly as py
# import plotly.graph_objs as go


# In[2]:


# Aqui se cambiaria por los archivos a leer pdbs sin modificar
# path1 = '1xxa.pdb'
# path2 = '1tig.pdb'

#funcion de lectura con biopandas
def read_biopdb(path):
    """Extrae las cordenadas de los atomos de C_alfa y los acomoda en un vector
    devuelve un dataframe con las coordenadas y el numero de residuo"""
    df = biop.read_pdb(path)
    df_atom = df.df['ATOM']
    #OJO AQUI ESTA ADECUADO AL PDB   para elegir solo un frame en trj_0 y trj_0_A [:1805]
    df_ca = df_atom[[
        'atom_number', 'atom_name', 'residue_name', 'residue_number',
        'x_coord', 'y_coord', 'z_coord'
    ]]
    columna_vector = []
    for i in zip(df_ca.x_coord.tolist(), df_ca.y_coord.tolist(),
                 df_ca.z_coord.tolist()):
        columna_vector.append(np.array(i))

    df_ca['vector'] = columna_vector
    return (df_ca)


# In[8]:


#se calcula la distancia entre cada par de nodos.
def distancia_entre_atomos(df_ca):
    """df_ca: Dataframe con coordenadas de los atomos alfa, devuelve otro DataFrame
    df_da: Dataframe como una matriz de adyacencias donde el valor es la distancia"""
    df_ca = df_ca[df_ca.atom_name == 'CA']
    distancias = []
    #se calcula la distancia euclidiana entre cada atomo de carbon alfalfa
    for v,i in zip(df_ca.vector,df_ca.atom_number):
        distancia_un_atomo = []
        for av,j in zip(df_ca.vector,df_ca.atom_number):
            distancia = pdist([v,av],metric='euclidean').item()
            distancia_un_atomo.append(distancia)
        distancias.append(distancia_un_atomo)
    #se genera la matriz de adyacencias para la red
    df_da = pd.DataFrame(index=df_ca.atom_number,columns=df_ca.atom_number,
                         data=distancias)
    return(df_da)


# In[9]:


#generacion de matriz de adyacencias
# df_da1 = distancia_entre_atomos(df_ca1)
# df_da2 = distancia_entre_atomos(df_ca2)
#podriamos solo mantener la matriz diagonal y dejarla como un array de arrays

# In[37]:


def gen_3_cliques(df_da, dth = 10, k=3):
    """Genera n-cliques de dataframe de distancias, tomando en cuenta los enlaces menores o iguales
    a dth y forma los k-cliques que elijas 
    valores por default:
    dth=10, k=3"""
    #red de distancias completa
    red = nx.from_pandas_adjacency(df_da)
#     print("red antes de filtros:",nx.info(red))

    #filtro de distancias
    edgesstrong = [(u,v) for (u,v,d) in red.edges(data=True) if d["weight"] <= dth]

    red = nx.Graph(edgesstrong)
#     print("=="*20)
#     print("red despues de filtros:",nx.info(red))

    n_cliques = [clq for clq in nx.find_cliques(red) if len(clq) >=k]
    print('numero de cliques maximos encontrados:',len(n_cliques))
#     print(n_cliques)

    lista_cliques = []
    for i,v in enumerate(n_cliques):
        a = list(it.combinations(v,k))
        for j in a:
            if set(j) not in lista_cliques:
                #recuerda que para comparar elementos utiliza set, y apilalos como set
                lista_cliques.append(set(j))

    df_lc = pd.DataFrame(lista_cliques)            
    print("numero de %s-cliques posibles:" % (k), df_lc.shape[0])
    return(df_lc,n_cliques)


# In[38]:


# df_lc1, = gen_3_cliques(df_da1,dth = 10, k=3)
# print('--'*59)
# df_lc2 = gen_3_cliques(df_da2,dth = 10, k=3)



# ## Calculo del SSM
# Para calcular la estructura secundaria, es necesario obtener los angulos dihedrales Phi y Psi, y posteriormente empalmarlo con el diagrama de Ramachandran y observar en que clasificacion cae, donde para fines practicos solo se utilizaran 3 estructuras:
#     + alfa helices
#     + beta laminas
#     + coil cualquier otra estructura no definida
#     
# Para obtener C, $\alpha$, $\beta$ con:
#    + $\Phi$
#    + $\Psi$
# 1. Matriz de comparacion de Estructura Secundaria (SSM)
# 2. Solvente Accesible (SAM)

# ### DSSP
# desde bash y se extrae la estructura secundaria.
# 
# __Que siempre si se utiliza este...__
# 
# Para comparar resultados se utiliza este !!!
# 
# Diccionario DSSP
# <img src='ss.jpeg'/>

# In[13]:


def mini_dssp(path,index):
    #ejecuto dssp desde bash y guardo archivo como output.log
    sp.run(['dssp','-i',path,'-o','output.log'])
    #parseo el dssp file
    dd_ob = dd.DSSPData()
    dssp_file_name = open('output.log')
    dd_ob.parseDSSP( 'output.log' )
    #obtengo la estructura y la guardo, posible no es necesario los residuos
    #solo el numero de atomo que le pego arbitrariamente REVISAR si esta bien
    ss = [i[2] for i in dd_ob.struct ]
    ss = pd.DataFrame([i for i in zip(ss,dd_ob.resnum,dd_ob.getChainType())])
    ss.columns = ['pre_ss','residue_number','chain']
    ss = ss[ss.chain == 'A']
    ss = ss[ss.residue_number != ''].reset_index(drop=True)
    ss['atom_number'] = index
    #catalogo  Yo tomo B y E como betas, G H I como alfa y lo demás como coil 
    #B - betas 
    #H - alfas
    ss['structure'] = np.where(ss.pre_ss.isin(['B','E']),'B',
                               np.where(ss.pre_ss.isin(['G','H','I']),'H',
                                        'C'))
    #checks
    print(ss.structure.value_counts(normalize = True) * 100)
    print(path)
    return(ss)


# In[14]:


# ss1 = mini_dssp(path1,df_ca1)
# print('//'*40)
# ss2 = mini_dssp(path2,df_ca2)


# In[15]:


#funcion para obtener las coordenadas del clique
def get_SS(ss,df_lc):
    """
    """
    #lista para apilar las estructuras
    c1 = []
    c2 = []
    c3 = []

    for i in df_lc.index:
        #si coincide el numero de atomo con el numero de atomo del clique le coloca el vector de coordenadas
        c1_temp = ss[ss.atom_number==df_lc.iloc[i,0]].structure.values[0]
        c2_temp = ss[ss.atom_number==df_lc.iloc[i,1]].structure.values[0]
        c3_temp = ss[ss.atom_number==df_lc.iloc[i,2]].structure.values[0]

        c1.append(c1_temp)
        c2.append(c2_temp)
        c3.append(c3_temp)

    df_lc['ss_0'] = c1
    df_lc['ss_1'] = c2
    df_lc['ss_2'] = c3
    
    #columna con coordenadas del clique
    return(df_lc)


# In[16]:


# df_lc1 = get_SS(ss1,df_lc1)
# df_lc2 = get_SS(ss2,df_lc2)


# Hora de comparar los SS

# In[17]:


def SSM(ss1,ss2):
    """Catalogo SSM siguiendo la tabla 1 y con una funcion extra,
    ss1: string (H,B,C)
    ss2: string (H,B,C)
    devuelve el score: int (0,1,2)"""
    def get_score_from_table(ss1,ss2):

        if (ss1 == 'H') and (ss2 == 'B'):
            score = 2
        elif (ss1 == 'B') and (ss2 == 'H'):
            score = 2
        else:
            print('WTF are u doing!')

        return(score)
    
    score = 123
    if ss1 ==  ss2:
        score = 0
    elif ((ss1 != ss2) & ((ss1 == 'C') | (ss2 == 'C'))):
        score = 1
    else:
        score = get_score_from_table(ss1,ss2)
        
    return(score)   


# In[20]:


# producto = it.product(df_lc1.index.values,df_lc2.index.values)
# cols = ['ss_0','ss_1','ss_2']
# for i in producto:
#     if i[0] < 1:
#         print(
#             list(#FUNCION PARA OBTENER LOS VALORES DE SSM
#                  map(SSM,df_lc1.iloc[i[0]][cols],df_lc2.iloc[i[1]][cols]))
#         )
#     else:
#         break


# In[21]:


# # df_lc1['score_ss'] =
# a = df_lc1.iloc[0][cols]
# b = df_lc2.iloc[-1][cols]
# list(map(SSM,a,b))


# # In[ ]:


# AGUAS CON EL OUTPUT NO LO DEJES ASI!!! HAY QUE QUITARLO
# for i in df_lc1['ss_0']:
#     for j in df_lc2['ss_0']:
# #         print(SSM(i,j))


# In[ ]:


# for i in df_lc1.index:
#     print(list(it.permutations(df_lc1[[0,1,2]].values[i],3)))
#     break


# ### DSSP A MANO 
# sin puentes de hidrogeno solo por Ramachandran Plot y filtros feeling
# 
# __QUE SIEMPRE NO SE UTILIZA ESTE__

# In[20]:


# #se genera un dataset con solo los atomos de interes para obtener la estructura
# df_dh1 = df_ca1[df_ca1.atom_name.isin(['N','CA','C',])].reset_index()
# df_dh2 = df_ca2[df_ca2.atom_name.isin(['N','CA','C',])].reset_index()


# In[21]:


# def calculate_phi_psi(df_dh):
#     #calculo de Phi observar el orden que es C--N--CA--C
#     index_atom_number = []
#     # index_phi = []
#     angulos_phi = []
#     append = angulos_phi.append
#     valores = df_dh.vector.values
#     dihedral = mvt.dihedral
#     nombres = df_dh.atom_name
#     for i in range(df_dh.shape[0]-3):
#         if i == 0: # COMO NO TIENE CON QUIEN COMPARAR SE AGREGA QUE EL PRIMERO COMIENCE A 360 GRADOS
#             append(360.0)

#         elif (nombres[i] == 'C') and (nombres[i+1] == 'N') and (nombres[i+2] == 'CA') and (nombres[i+3] == 'C'):
#     #         index_phi.append(df_dh1.residue_number.values[i])
#             index_atom_number.append(df_dh.atom_number.values[i]-1)
#             append(dihedral(valores[i],valores[i+1],valores[i+2],valores[i+3]))

#     # index_phi.append(df_dh1.residue_number.values[-1])
#     index_atom_number.append(df_dh.atom_number.values[-1]-1)
    
#     #Calculo de psi con el orden N--CA--C--N
#     angulos_psi = []
#     append = angulos_psi.append
#     valores = df_dh.vector.values
#     dihedral = mvt.dihedral
#     nombres = df_dh.atom_name
#     for i in range(df_dh.shape[0]-3):
#             if (nombres[i] == 'N') and (nombres[i+1] == 'CA') and (nombres[i+2] == 'C') and (nombres[i+3] == 'N'):
#                 append(dihedral(valores[i],valores[i+1],valores[i+2],valores[i+3]))
                
#     angulos = pd.DataFrame([angulos_phi,angulos_psi]).T
#     angulos.columns = ['phi','psi']
#     angulos.phi = np.where(angulos.phi > 180, angulos.phi - 360, angulos.phi)
#     angulos.psi = np.where(angulos.psi > 180, angulos.psi - 360, angulos.psi)
#     angulos.replace( 0, 360, inplace = True)
#     angulos['atom_number'] = index_atom_number
#     angulos.fillna(360.0, inplace=True)
#     return(angulos)


# In[22]:


# angulos1 = calculate_phi_psi(df_dh1)
# angulos2 = calculate_phi_psi(df_dh2)


# In[23]:


# #Calculo de psi con el orden N--CA--C--N
# angulos_psi = []
# append = angulos_psi.append
# valores = df_dh1.vector.values
# dihedral = mvt.dihedral
# nombres = df_dh1.atom_name
# for i in range(df_dh1.shape[0]-3):
#         if (nombres[i] == 'N') and (nombres[i+1] == 'CA') and (nombres[i+2] == 'C') and (nombres[i+3] == 'N'):
#             append(dihedral(valores[i],valores[i+1],valores[i+2],valores[i+3]))


# In[24]:


# SE REVISAN LOS ANGULOS QUE SEAN ADECUADOS
# SE CHECO UTILIZANDO DSSP ONLINE Y SI DAN LOS ANGULOS
#AHORA FALTA GENERAR O UN CATALOGO O EMPALMAR EL RAMACHANDRAN PLOT PARA
#OBTENER LA ESTRUCTURA SECUNDARIA
# angulos = pd.DataFrame([angulos_phi,angulos_psi]).T
# angulos.columns = ['phi','psi']
# angulos.phi = np.where(angulos.phi > 180, angulos.phi - 360, angulos.phi)
# angulos.psi = np.where(angulos.psi > 180, angulos.psi - 360, angulos.psi)
# angulos.replace( 0, 360, inplace = True)
# angulos['atom_number'] = index_atom_number
# angulos.fillna(360.0, inplace=True)


# In[25]:


# siguiendo el catalogo de: https://www.researchgate.net/publication/220777003_Protein_Secondary_Structure_Prediction_Based_on_Ramachandran_Maps
#crearemos el catalogo para la SS
# para alfa-helice tienen que caer los puntos dentro del circulo de radio 7 con centro en (-63,-45)
# def inside_circle(x,y):
#     R = 7
#     x_center = -63.0
#     y_center = -45.0
#     distancia = np.sqrt((x - (x_center))**2 + (y - (y_center))**2)
#     return(distancia <= R)

# boolean_list_alfa_circle =  np.where(inside_circle(angulos.phi,angulos.psi),1,0)


# In[26]:


# def inside_alfa_helix(x,y):
#     esta_en_x = -75.0 <= x <= -45.0
#     esta_en_y = -60.0 <= y <= -30.0
#     boolean = esta_en_x and esta_en_y
#     return(boolean)

# boolean_list_alfa = []
# for i in angulos.index:
#     resultado = inside_alfa_helix(angulos.phi[i],angulos.psi[i])
#     boolean_list_alfa.append(resultado)
    
# boolean_list_alfa = np.array(boolean_list_alfa) * 1


# In[27]:


# def inside_beta_sheets(x,y):
#     esta_en_x = -180.0 <= x <= -105.0
#     esta_en_y = 120.0 <= y <= 180.0
#     boolean = esta_en_x and esta_en_y
#     return(boolean)

# boolean_list_beta = []
# for i in angulos.index:
#     resultado = inside_beta_sheets(angulos.phi[i],angulos.psi[i])
#     boolean_list_beta.append(resultado)
    
# boolean_list_beta = np.array(boolean_list_beta) * 1


# In[28]:


# def secundary_structure(angulos,df_ca):
#     #alfa helices
#     def inside_circle(x,y):
#         R = 7
#         x_center = -63.0
#         y_center = -45.0
#         distancia = np.sqrt((x - (x_center))**2 + (y - (y_center))**2)
#         return(distancia <= R)

#     boolean_list_alfa_circle =  np.where(inside_circle(angulos.phi,angulos.psi),1,0)
#     #mas alfa helices
#     def inside_alfa_helix(x,y):
#         esta_en_x = -75.0 <= x <= -45.0
#         esta_en_y = -60.0 <= y <= -30.0
#         boolean = esta_en_x and esta_en_y
#         return(boolean)

#     boolean_list_alfa = []
#     for i in angulos.index:
#         resultado = inside_alfa_helix(angulos.phi[i],angulos.psi[i])
#         boolean_list_alfa.append(resultado)
    
#     boolean_list_alfa = np.array(boolean_list_alfa) * 1
#     #beta sheets
#     def inside_beta_sheets(x,y):
#         esta_en_x = -180.0 <= x <= -105.0
#         esta_en_y = 120.0 <= y <= 180.0
#         boolean = esta_en_x and esta_en_y
#         return(boolean)

#     boolean_list_beta = []
#     for i in angulos.index:
#         resultado = inside_beta_sheets(angulos.phi[i],angulos.psi[i])
#         boolean_list_beta.append(resultado)
    
#     boolean_list_beta = np.array(boolean_list_beta) * 1
#     #generacion de dataframe structure
#     structure = pd.DataFrame([boolean_list_alfa_circle,
#                               boolean_list_alfa,
#                               boolean_list_beta]).T
#     structure.columns = ['alfa_circulo','alfa_helice','beta_sheets']
#     structure['SS'] = np.where(structure.alfa_circulo == 1,'alfa_circulo',
#                               np.where(structure.alfa_helice == 1,'alfa_helice',
#                                  np.where(structure.beta_sheets == 1,'beta_sheets','COIL')))
#     angulos['SS'] = np.where(structure.SS.isin(['alfa_circulo','alfa_helice']),'H',
#                              np.where(structure.SS == 'beta_sheets','E','C'))
#     # # H -- ALFA HELIX
#     # # E -- BETA SHEETS
#     # # C -- COIL
# #     print(angulos.SS.value_counts(normalize = True) * 100)
#     df_ca = df_ca.merge(angulos[['atom_number','SS']], 
#                         how='left',on='atom_number')
#     return(df_ca)


# In[29]:


# #calculo de SS y pegado
# df_ca1 = secundary_structure(angulos1,df_ca1)
# df_ca2 = secundary_structure(angulos2,df_ca2)


# In[30]:


# #check de SS
# df_ca1.SS.value_counts(normalize=True)*100


# In[31]:


# %%timeit
# secundary_structure(angulos1,df_ca1)


# ### Checks de Estructura secundaria
# Para generar los checks correr el codigo sin funciones

# In[32]:


# structure.SS.value_counts(normalize = True) * 100


# In[33]:


# angulos['SS'] = np.where(structure.SS.isin(['alfa_circulo','alfa_helice']),'H',
#                          np.where(structure.SS == 'beta_sheets','E','C'))
# # # H -- ALFA HELIX
# # # E -- BETA SHEETS
# # # C -- COIL
# angulos.SS.value_counts(normalize = True) * 100


# In[34]:


# angulos['color_SS'] = np.where(angulos.SS == 'H','r',
#                                np.where(angulos.SS == 'E','navy','g'))

# angulos[['atom_number','SS']]


# In[35]:


# #Metodologia wiki
# #alfa helice (−90°, −15°) to (−35°, −70°) ROJO
# #betta (–135°, 135°) to (–180°, 180°) NAVY


# In[36]:


# #RAMACHANDRAN PLOT SOLO ANGULOS
# angulos.plot.scatter('phi','psi', title='Ramachandran Plot', 
#                      c = angulos.color_SS.values.tolist(),
#                      marker = 'x',
#                      alpha=0.6, figsize=(10,10), s=80
#                     )
# limite1,limite2 = -190,190
# plt.xlim(limite1,limite2)
# plt.ylim(limite1,limite2);


# In[37]:


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


# In[38]:


# #pegado de coordendas
# df_lc1 = get_coord_clique(df_ca1,df_lc1)
# df_lc2 = get_coord_clique(df_ca2,df_lc2)


# ## Comparacion de cliques
# Para obtener el __RMSD__ es necesario primero rotar y trasladar un atomo con respecto al atomo a comparar (de la otra proteina) y calcular el __RMSD__.
# 
# Siguiendo al metodologia en *Using quaternions to calculate RMSD*.
# Se generan las funciones de traslado y rotacion.

# ### Traslacion
# Se calcula el baricentro de cada clique en ambas moleculas y se generan nuevos vectores que van del baricentro al atomo llamados $\hat{x}$.
# 
# El baricentro se calcula como $\bar{x} =$($\frac{(x_1 + x_2 + x_3)}{3}$,$\frac{(y_1 + y_2 + y_3)}{3}$,$\frac{(z_1 + z_2 + z_3)}{3}$)
# 
# $\hat{x} = x_k - \bar{x}$

# In[40]:


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


# In[41]:


# #calculo de baricentro
# df_lc1 = baricenter_clique(df_lc1)
# df_lc2 = baricenter_clique(df_lc2)


# In[43]:


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


# In[44]:


# #generacion de vectores gorro
# df_lc1 = center_vectors(df_lc1)
# df_lc2 = center_vectors(df_lc2)


# ### Rotacion
# Para generar la rotacion tenemos que generar la *matriz gigante* que depende de los elemento de la matriz de correlacion $R_{ij}$
# 
# Donde $R_{ij} = \sum\limits_{k=1}^N{x_{ik}y_{jk}}, i,j = 1,2,3$
# 
# Posteriormente se calculan los eigenvalores y eigenvectores de esta matriz gigante
# Para obtener los quaterniones y generar la matriz de rotacion y con ella calcular el vector rotado
# 
# Por ultimo, se suma al vector rotado y trasladado se suma el baricentro del clique a comparar y se calcula el RMSD

# In[46]:


# No conviene utilizar pandas ya que tarda al acceder a los datos, buscar la manera
# #usual para acceder a los datos y seguir avanzando
# prueba1 = df_lc1.values
# prueba2 = df_lc2.values


# # In[52]:


# for i,val in enumerate(df_lc1.columns):
#     print(i,val)


# In[53]:


#funcion para obtener los valores de la prerotacion, de los valores de la matriz de correlaciones
# en check por que se utiliza vector gorro en lugar de posiciones iniciales 
# el articulo no dice...




# In[54]:


# R_ij(1,1)


# In[56]:





# In[58]:


# giant_matrix(1,1)


# In[59]:




# In[60]:


# rotation_matrix(giant_matrix(1,1))


# # In[61]:



# ## Calculo del RMSD con previo filtro de SSM y SAM
# Aqui iria el codigo de SSM y SAM para filtrado y despues se calcula el rmsd de cada clique, por lo que primero hay que filtrar

# In[93]:




# In[94]:


# matriz_gigante = giant_matrix(1,1)
# mat_rot = rotation_matrix(matriz_gigante)
# x_rot = rotation_vectors(prueba1[:,14][1],mat_rot)
# coord_rot_clique_2 = x_rot + np.array(prueba2[:,10][1])
# print(rmsd_between_cliques(coord_rot_clique_2,np.array(prueba2[:,9][1])))


# In[95]:


# prueba2[:,9][1]


# In[96]:


def calculate_rmsd_rot_trans(atom1,atom2,prueba1,prueba2):

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
        append = values.append
        for k in [11, 12, 13]:  # 8,9,10 corresponde a la columna de vec_gorro_0,_1,_2
            # REVISAR VEC_GORRO
            atom_value1 = prueba1[:, k][a1][i]
            atom_value2 = prueba2[:, k][a2][j]
            value = atom_value1 * atom_value2
            append(value)

        valor = sum(values)
        return (valor)

    def giant_matrix(i, j):
        """cliques a comparar: i,j
        desde aqui se itera sobre cada i y hay que variar los vectores
        coordenada
        Regresa la matriz gigante (matriz simetrica del articulo)"""
        # primer renglon
        R11R22R33 = (R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j) + R_ij(3, 3, a1=i, a2=j))
        R23_R32 = (R_ij(2, 3, a1=i, a2=j) - R_ij(3, 2, a1=i, a2=j))
        R31_R13 = (R_ij(3, 1, a1=i, a2=j) - R_ij(1, 3, a1=i, a2=j))
        R12_R21 = (R_ij(1, 2, a1=i, a2=j) - R_ij(2, 1, a1=i, a2=j))
        # segundo renglon
        R11_R22_R33 = (R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j) - R_ij(3, 3, a1=i, a2=j))
        R12R21 = (R_ij(1, 2, a1=i, a2=j) + R_ij(2, 1, a1=i, a2=j))
        R13R31 = (R_ij(1, 3, a1=i, a2=j) + R_ij(3, 1, a1=i, a2=j))
        # tercer renglon
        _R11R22_R33 = (-R_ij(1, 1, a1=i, a2=j) + R_ij(2, 2, a1=i, a2=j) - R_ij(3, 3, a1=i, a2=j))
        R23R32 = (R_ij(2, 3, a1=i, a2=j) + R_ij(3, 2, a1=0, a2=0))
        # cuarto renglon
        _R11_R22R33 = (-R_ij(1, 1, a1=i, a2=j) - R_ij(2, 2, a1=i, a2=j) + R_ij(3, 3, a1=i, a2=j))

        matriz_gigante = [
            [R11R22R33, R23_R32, R31_R13, R12_R21],
            [R23_R32, R11_R22_R33, R12R21, R13R31],
            [R31_R13, R12R21, _R11R22_R33, R23R32],
            [R12_R21, R13R31, R23R32, _R11_R22R33]
        ]
        return (matriz_gigante)

    def rotation_matrix(matriz_gigante):
        """utilizando la funcion giant_matrix, fijando los valores de i,j
        se calcula la matriz de rotacion con los eigenvectores y eigenvalores
        arroja una matriz de rotacion que depende de la matriz gigante
        """
        eignvalues, eigenvectors = np.linalg.eig(matriz_gigante)
        q = eigenvectors[:, np.argmax(eignvalues)]
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        # matriz de rotacion con eigenvectores
        mat_rot = np.array([
            [(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)]
        ])
        return (mat_rot)

    def rotation_vectors(vector_gorro, mat_rot):
        """obtencion de vector rotado,
        utilizando la matriz de rotacion
        y los vectores gorro a rotar y trasladar"""
        # multiplicacion de matrices de cada vector rotado
        coord_rot_tras = []
        append = coord_rot_tras.append
        matmul = np.matmul
        for i in vector_gorro:
            append(matmul(mat_rot, i.reshape(3, 1)).T[0])

        return (coord_rot_tras)

    def rmsd_between_cliques(atom_trans_rot, atom_to_compare):
        """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
        y el atomo a comparar, por el momento solo imprime el resultado"""
        # primer RMSD entre atomos
        p12 = np.sum((np.array(atom_to_compare) - atom_trans_rot) ** 2, 1)
        rmsd_i = lambda i: np.sqrt(i) / 3
        rmsd_final = rmsd_i(p12).mean()

        # if rmsd_final <= 0.15:  ##AQUI LOS DETECTA QUIENES CUMPLEN CON EL FILTRO...
        #     print('RMSD_final:', rmsd_final)

        return(rmsd_final)

    matriz_gigante = giant_matrix(atom1,atom2)
    mat_rot = rotation_matrix(matriz_gigante)
    x_rot = rotation_vectors(prueba1[:,14][atom1],mat_rot)
    coord_rot_clique_2 = x_rot + np.array(prueba2[:,10][atom2]) #xrot + baricentro a mover
    rmsd_final = rmsd_between_cliques(coord_rot_clique_2,np.array(prueba2[:,9][atom2]))
    # clique rotado y trasladado vs clique coordenadas
    return(rmsd_final)


# In[98]:


# for i in range(df_lc2.shape[0]): 
#     print(calculate_rmsd_rot_trans(10,i))


# In[ ]:


# %%time
# for j in range(df_lc1.shape[0]):
#     for i in range(df_lc2.shape[0]):
#         calculate_rmsd_rot_trans(j,i)

# %%time
#para quitarme el for anidado puedo hacerlo con este producto por lo que reduce ligeramente pero aun no se puede medir
# #dado que son muchas operaciones 13000 * 2000
# producto = it.product(df_lc1.index.values,df_lc2.index.values)
# for i,j in producto[:100]:
#     calculate_rmsd_rot_trans(j,i)

