import numpy as np
import random
from collections import defaultdict

# Python Program to detect cycle in an undirected graph
#This class represents a undirected graph using adjacency list representation
class Graph:
    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = defaultdict(list) # default dictionary to store graph
        self.wgts = defaultdict(list) # default dictionary to store graph
        self.paths = [] # default dictionary to store graph
        self.cycles = [] # default dictionary to store graph

    def SetGraphByEdges(self,list_of_edges):
        if len(list_of_edges[0]) == 3:
           have_wgts = True
        elif len(list_of_edges[0]) == 2:
           have_wgts = False
        else:
           print("The length of the elements in the list should be: ")
           print(" 2 for unweight edges or 3 for weighted edges ")
           print("Elements with length %s were found. "%len(list_of_edges[0]))
           sys.exit()

        for pair in list_of_edges:
            n1 = int(pair[0])-1
            n2 = int(pair[1])-1
            if have_wgts:
               w = float(pair[2])
               self.addEdge(n1,n2,w)
            else:
               self.addEdge(n1,n2)

    def addEdge(self,u,v,w=1.):
        """function to add an edge to graph"""
        key_entry = str(u)+"_"+str(v)
        self.wgts[key_entry] = w
        self.graph[u].append(v) #Add w to v_s list
        self.graph[v].append(u) #Add v to w_s list

    def LoadDictWeigths(self,dict_wts_edges):
        for pair in list_of_edges:
            self.addEdge(int(pair[0])-1, int(pair[1])-1)


    def CreateAdjMat(self):
        length = self.V
        adj_mat = np.zeros([length,length])
        for i in range(length):
            for j in range(length):
                if not i==j and i < j and j in self.graph[i] :
                   adj_mat[i][j] = 1
                   adj_mat[j][i] = 1
        self.AdjMat = adj_mat

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


    def findCyclesOfSize(self,target):
        target -= 1 # the search is actully over edges
        self.CreateAdjMat()
        #print self.AdjMat
        powmat = self.PowAdjMat(target)
        #print powmat
        lr = list(range(self.V)) #MODIFICAR ESTA PARTE PARA QUE NO ITERE SOBRE EL NUMERO DE NODOS TOTALES SI NO SOBRE LOS NODOS DE ATOMOS AROMATICOS
        random.shuffle(lr)
        for i in lr:
            # Get vertices that can be reached from "i" in "target" steps
            # and that are neighbors of "i"
            z = [ x for x in range(self.V) if powmat[i][x] > 0 and x in self.graph[i] ]
            # Search for paths between "i" and those vertices in "j".
            for j in z:
              self.getAllPaths(i,j,target)  #DETENER LA BUSQUEDA CUANDO HAYA ENCONTRADO LOS CICLOS DESEADOS
        for p in self.paths:
            if p not in self.cycles:
               self.cycles.append(p)

    def getAllPaths(self, s , d , c, dist = 0):
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
        # Create an array to store paths
        path = []
        # Call the recursive helper function to print all paths
        if dist == 0:
           self.getAllPathsUtil(s , d , visited , path , c )
        else:
           of = open('paths/from_%s_to_%s.txt'%(s,d),'w')
           setattr(self,"paths",[])
           self.getAllPathsCutoff(s , d , visited , path , c , dist,of )


    def getAllPathsCutoff(self, u, d, visited, path, steps, dist, of):
        '''A recursive function to find all paths from 'u' to 'd' with
           weighted distance less than c.
           visited[] keeps track of vertices in current path.
           path[] stores actual vertices and path_index is current
           index in path[]'''

        if not visited[u]:
           visited[u]= True
           path.append(u)
           ed_dist = 0.
           if len(path)>1:
              cur_edge = str(path[-2])+"_"+str(u)
              ed_dist = self.getWgt(cur_edge)
           dist -= ed_dist
           steps -= 1
           if steps == -1:
              #print("reached deep: ",path)
              pass
           if dist < 0.:
              #print("reached deep: ",path)
              pass
           #elif u == d and dist >= 0. and not path in self.paths:
           elif u == d and dist >= 0.:
                #print("reached destiny: ", path , dist)
                g = '_'.join([str(x) for x in path])
                of.write("%-10.10f %s\n"%(dist,g))
                #self.paths.append(list(path))
           else:
               for i in self.graph[u]:
                   if visited[i]==False:
                      self.getAllPathsCutoff(i, d, visited, path, steps,dist, of)
           path.pop()
           visited[u]= False

    def getWgt(self,edge):
        ed_dis = self.wgts[edge]
        if not isinstance(ed_dis,float):
          alt = edge.split('_')
          alt.reverse()
          alt_edge = '_'.join(alt)
          ed_dis = self.wgts[alt_edge]
        return ed_dis

    def checkPathDistance(self,path):
        pairs = [ '_'.join([str(i),str(j)]) for i,j in zip(path[:-1],path[1:]) ]
        return np.sum( [ self.getWgt(i) for i in pairs ] )


"""
Ligando completo con benzofurano
g = Graph(21)
g.addEdge(4, 9)
g.addEdge(4, 3)
g.addEdge(11, 10)
g.addEdge(2, 3)
g.addEdge(2, 1)
g.addEdge(9, 8)
g.addEdge(3, 6)
g.addEdge(10, 1)
g.addEdge(10, 12)
g.addEdge(1, 5)
g.addEdge(8, 19)
g.addEdge(8, 7)
g.addEdge(18, 12)
g.addEdge(18, 16)
g.addEdge(6, 5)
g.addEdge(6, 7)
g.addEdge(12, 13)
g.addEdge(19, 20)
g.addEdge(16, 15)
g.addEdge(13, 14)
g.addEdge(15, 14)
g.addEdge(15, 17)
g.addEdge(0, 17)
g.findCyclesOfSize(6)
print g.paths
"""

"""
#Aromaticos benzofurano
g = Graph(15)
g.addEdge(1, 5)
g.addEdge(12, 13)
g.addEdge(13, 14)
g.addEdge(15, 14)
g.addEdge(16, 15)
g.addEdge(18, 12)
g.addEdge(18, 16)
g.addEdge(2, 3)
g.addEdge(2, 1)
g.addEdge(3, 6)
g.addEdge(4, 9)
g.addEdge(4, 3)
g.addEdge(6, 5)
g.addEdge(6, 7)
g.addEdge(8, 7)
g.addEdge(9, 8)

g.findCyclesOfSize(6)
print g.paths
"""

"""
#Benzofurano-benceno 9 nodos
g = Graph(10)
g.addEdge(4, 9)
g.addEdge(4, 3)
g.addEdge(2, 3)
g.addEdge(2, 1)
g.addEdge(9, 8)
g.addEdge(3, 6)
g.addEdge(1, 5)
g.addEdge(8, 7)
g.addEdge(6, 5)
g.addEdge(6, 7)

g.findCyclesOfSize(6)
print g.paths
"""


"""
#Tresbecenos 13 nodos
g = Graph(13)
g.addEdge(1, 0)
g.addEdge(1, 2)
g.addEdge(2, 3)
g.addEdge(3, 12)
g.addEdge(12, 11)
g.addEdge(11, 10)
g.addEdge(10, 9)
g.addEdge(9, 8)
g.addEdge(8, 7)
g.addEdge(6, 7)
g.addEdge(5, 6)
g.addEdge(5, 0)
g.addEdge(3, 4)
g.addEdge(4, 9)
g.addEdge(4, 5)

g.findCyclesOfSize(6)
print g.paths
"""
