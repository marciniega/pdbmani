import numpy as np
import random
from collections import defaultdict

# This class represents a undirected graph using adjacency list representation
class Graph:
    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = defaultdict(list) # default dictionary to store graph
        self.paths = [] # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self,v,w):
        self.graph[v].append(w) #Add w to v_s list
        self.graph[w].append(v) #Add v to w_s list

    def BuildNet(self,list_of_pairs):
        for i in list_of_pairs:
            self.addEdge(i[0]-1, i[1]-1)

    def CreateAdjMat(self):
        length = self.V
        adj_mat = np.zeros([length,length])
	for i in range(length):
            for j in range(length):
                if not i==j and i < j and j in self.graph[i] :
                   adj_mat[i][j] = 1
                   adj_mat[j][i] = 1
	self.AdjMat = adj_mat

    def MarkAro(self,aromatic):
        self.aro_atoms = [ i for i in aromatic ]

    def PowAdjMat(self,power):
        t = self.AdjMat
        for i in range(power-1):
            t = np.dot(t,self.AdjMat)
	return t

    def getAllPathsUtil(self, u, d, visited, path,c):
        '''A recursive function to find all paths from 'u' to 'd'.
           visited[] keeps track of vertices in current path.
           path[] stores actual vertices and path_index is current
           index in path[]'''

        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)

        # If current vertex is same as destination and
        # path is of the expected length and the vertices in the
        # path are unique, then added to the paths attribute
        if u == d and len(path) == c+1 and not sorted(path) in self.paths :
            self.paths.append(sorted(path))
        else:
            #Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i]==False:
                   self.getAllPathsUtil(i, d, visited, path, c)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False

    def getAllPaths(self, s , d , c):
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
        # Create an array to store paths
        path = []
        # Call the recursive helper function to print all paths
        self.getAllPathsUtil(s , d , visited , path , c )

    def findCyclesOfSize(self,target):
        self.paths = []
        target -= 1 # the search is actully over edges
        self.CreateAdjMat()
        #print self.AdjMat
        powmat = self.PowAdjMat(target)
        #print powmat
        lr = range(self.V) #MODIFICAR ESTA PARTE PARA QUE NO ITERE SOBRE EL NUMERO DE NODOS TOTALES SI NO SOBRE LOS NODOS DE ATOMOS AROMATICOS
        random.shuffle(lr)
        for i in lr:
            # Get vertices that can be reached from "i" in "target" steps
            # and that are neighbors of "i"
            z = [ x for x in range(self.V) if powmat[i][x] > 0 and x in self.graph[i] ]
            # Search for paths between "i" and those vertices in "j".
            for j in z:
              self.getAllPaths(i,j,target)  #DETENER LA BUSQUEDA CUANDO HAYA ENCONTRADO LOS CICLOS DESEADOS

    def GetAroCycles(self,number):
        self.findCyclesOfSize(number)
        paths = (np.asarray(self.paths) + 1)
        return [ i for i in paths if set(i).issubset(set(self.aro_atoms))]

