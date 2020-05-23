#!/usr/bin/env python
import sys
import numpy as np
from numpy import pi
import graph as mygraph
from math_vect_tools import normalize_vec,dihedral



dict_sybyl = {'C.3' : 4 , 'C.2'  : 3    , 'C.1': 2 , 'C.ar': 3,
              'N.3' : 4 , 'N.2'  : 3    , 'N.1': 2 , 'N.ar': 3,
              'N.am': 4 , 'N.pl3': 4    , 'N.4': 4 , 'O.3' : 4,
              'O.2' : 3 , 'O.co2': 3.25 , 'S.3': 4 , 'S.2' : 4,
              'S.O' : 4 , 'S.O2' : 4    , 'P.3': 4 , 'Se'  : 4,
              'H'   : 1 , 'B'    : 3    , 'F'  : 4 , 'Cl'  : 4,
              'Br'  : 4 , 'I'    : 4 }

dict_VE = {'B'  : 3, 'C' : 4, 'N'  : 5, 'P'  : 5 , 'O' : 6, 'S' : 6,
           'Se' : 6, 'F' : 7, 'Cl' : 7, 'Br' : 7 , 'I' : 7, 'H' : 1}

dict_expected = {'N.3' : 3 , 'N.2'  : 2 , 'N.1' : 1 , 'N.ar': 2 ,
                 'N.am': 3 , 'N.pl3': 3 , 'N.4' : 4 , 'C.3' : 4 ,
                 'C.2' : 3 , 'C.1'  : 2 , 'C.ar': 3 , 'O.3' : 2 ,
                 'O.2' : 1 , 'O.co2': 1 , 'S.3' : 2 , 'S.2' : 2 ,
                 'S.O' : 3 , 'S.O2' : 4 }

dict_max_bonds = {'B'  : 3, 'C' : 4, 'N'  : 4, 'P'  : 5 , 'O' : 2, 'S' : 6,
                  'Se' : 1, 'F' : 1, 'Cl' : 1, 'Br' : 1 , 'I' : 1, 'H' : 1}

pi_bonds = [ 'am','ar','cat','2']
pi_electrons = [ 'C.2','C.ar','N.2','N.ar','O.2','S.2']

def pdb_line(coor,at_nu="1",at_nm="C",rs_nu=1,rs_nm="XXX",ch="X"):
    line = "ATOM"
    line += "%7s"%at_nu
    line += "%5s"%at_nm
    line += "%4s"%rs_nm
    line += "%2s"%ch
    line += "%4s"%rs_nu
    line += "    "
    line += "%8.3f"%coor[0]
    line += "%8.3f"%coor[1]
    line += "%8.3f"%coor[2]
    line += "%6.2f"%0.0
    line += "%6.2f"%0.0
    line += " "
    line += "%3s"%at_nm
    return line

def check_cat_atom(atmobj):
    """This function only makes sen"""
    valence_complete = False
    if atmobj.atype == 'C.cat' :
       n_bonds = 0
       for i in atmobj.bonds.values():
           try:
              val = int(i)
           except ValueError:
              val = 2
           n_bonds += val
       if n_bonds == 4:
          valence_complete = True
       else:
          print("The C.cat atom is probably right assigned. Thus, the structure is probably wrong.")
          print("The carbocations are not supported.")
          sys.exit()
    return valence_complete

def check_no_single_aro(val,attp):
    if val == 1 and not "O.co2":
       msg = "\nSomething is weird.\n"
       msg += "The molecule on has an odd name of aromatic bonds.\n"
       msg += "Better check that. We leave with error.\n"
       raise ValueError(msg)
    else:
       pass

def find_num_aro_bonds(atmobj):
    n_aro_bonds = np.sum(np.array([ 1 for n in atmobj.bonds.values() if n == 'ar' ]))
    try:
        check_no_single_aro(n_aro_bonds,atmobj.atype)
        return n_aro_bonds
    except ValueError as e:
        print(e)
        sys.exit()

def list_duplicates(seq):
    """This returns the elements that appear more than once in a list.
       The elements in the list should be int or str."""
    seen = set()
    seen_add = seen.add
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
    return list( seen_twice )           

class Atom():
      def __init__(self,ndx,name,atype,coord,charge):
          self.index = ndx
          self.name = name
          self.atype = atype
          self.coord = coord
          self.charge = charge
          self.element = atype.split('.')[0]
          try:
            self.hybrid = dict_sybyl[atype]
          except KeyError:
            if atype == 'C.cat':
               print("Atom of type C.cat detected. If it has")
               print("4 bonds the analysis will continue as with type C.2")
            else:
               print("Atom with unusual sybyl atom type : %s"%atype)
               exit()
          self.VE = dict_VE[self.element]
          self.neighbors = []
          self.bonds = {}

      def setNumBonds(self):
          n_aro_b = find_num_aro_bonds(self)
          n_bonds = 0
          for b in self.bonds.values():
              if not b in ['ar','am']:
                 val = int(b)
              else:
                 if b == 'ar' and self.element in ['C']:
                    val = 1+(1/n_aro_b)
                 elif b == 'am' :
                    val = 1
                 elif b == 'ar' and self.atype in ['N.ar'] and len(self.neighbors) == 2:
                    val = 1+(1/n_aro_b)
                 elif b == 'ar' and self.atype in ['O.co2']:
                    val = 2
                 elif b == 'ar' and self.element in ['O','S','N']:
                    val = 1
                 else:
                    val = None
              n_bonds += val
          if n_bonds > dict_max_bonds[self.element]:
             print("\nFake correction on number of bonds was done!\n")
             print("\n Atom index: %s Atom name: %s  Atom type: %s\n"%(self.index,self.name,self.atype))
             n_bonds = dict_max_bonds[self.element]
          self.nbonds = n_bonds

      def setNumLonePairs(self):
          easy = ['F','Cl','Br','I','B','Se','P'] # always with right number of neighbors
          hard = ['N','C','S','O']
          if self.element in easy:
             self.nlonepairs = 2*(self.hybrid - len(self.neighbors))
          elif self.element in hard:
             if self.atype == 'N.pl3' and len([1 for i in self.bonds.values() if i == '2']) > 0 :
                self.nlonepairs = 2*(self.hybrid - 1 - dict_expected[self.atype])
             #elif self.atype == 'N.ar' and len([1 for i in self.bonds.values() if i == 'ar']) == 3 : # indolizine
             #   self.nlonepairs = 2*(self.hybrid - 1 - dict_expected[self.atype])
             #elif self.atype == 'S.2' and len([1 for i in self.bonds.values() if i == 'ar']) > 0 :
             #   self.nlonepairs = 2*(self.hybrid + 1 - dict_expected[self.atype])
             else:
                self.nlonepairs = 2*(self.hybrid - dict_expected[self.atype])
          else: # only H
             self.nlonepairs = 0

class Molmol2():
      def __init__(self, name ):
          self.name = name
          self.atoms = []
          self.bonds = {}
          self.aro_cycles = []

      def __iter__(self):
          return iter(self.atoms)

      def next(self): # Python 3: def __next__(self)
          if self.current == self.end:
             self.resetAtomIter()
             raise StopIteration
          else:
             self.current += 1
             return self.atoms[self.current - 1]

      def resetAtomIter( self , start = 0):
          self.current = start

      def printMolSummary(self):
          """ Print Mol information
              No.Atoms"""
          #print ('Notes:\n For HBD only the number of heavy atoms is printed.\n')
          print ('HevAtoms %2s  HBD %2s  HBA %2s  Aro %2s  Cations %2s  Anions %2s  Phobic %2s'\
                %(len([ 1 for i in self if not i.element == 'H' ]),\
                  len(set([ self.getAtom(p[0]).index for p in self.donors])),len(self.acceptors),\
                  len(self.aro_cycles) ,len(self.cations),len(self.anions),len(self.phobic)))

      def readMolFile(self,infile):
          found_bounds = False
          flag = False
          count = 0
          for line in open(infile,'r').readlines():
              line = line.split('\n')[0].split()
              if '@<TRIPOS>MOLECULE' in line:
                  count += 1
                  continue
              if '@<TRIPOS>BOND' in line:
                  found_bounds = True
                  continue
              if '@<TRIPOS>SUBSTRUCTURE' in line or count > 1:
                  break
              if '@<TRIPOS>ATOM' in line and not found_bounds:
                 flag = True
                 continue
              if flag and not found_bounds:
                 ndx = int(line[0])-1
                 coor = np.array([ float(i) for i in line[2:5]])
                 name = line[1]
                 atype = line[5]
                 charge = float(line[-1])
                 self.addAtom(ndx,name,atype,coor,charge)
              if found_bounds and count == 1:
                 atm1 = int(line[1])-1
                 atm2 = int(line[2])-1
                 btyp = line[3]
                 if (self.getAtom(atm1).atype == 'O.co2' or self.getAtom(atm2).atype == 'O.co2') and not btyp == 'ar':
                    btyp = 'ar'
                 bond = str(atm1)+"_"+str(atm2)
                 self.bonds[bond] = btyp
                 self.atoms[atm1].neighbors.append(atm2)
                 self.atoms[atm2].neighbors.append(atm1)
                 self.atoms[atm1].bonds[bond] = btyp
                 self.atoms[atm2].bonds[bond] = btyp
          self.end = len(self.atoms)
          self.current = 0
          for a in self.atoms:
              if check_cat_atom(a):
                 a.atype = 'C.2'
                 a.hybrid = dict_sybyl[a.atype]
              a.setNumLonePairs()
              a.setNumBonds()
          self.genMolGraph()
          self.setCycles()
          self.getAroCycles()
          self.setAroData()
          self.findMultiAromatic()
          self.getHbs()
          self.getCharged()
          self.getPhobic()
          self.findClusterHydrophobic()

      def addAtom(self,nx,nm,atp,crd,chrg):
          new_atom = Atom(nx,nm,atp,crd,chrg)
          self.atoms.append(new_atom)

      def getListOfPairs(self):
          return [ (int(pair.split('_')[0])+1,int(pair.split('_')[1])+1)  for pair in self.bonds ]

      def getAtom(self,ndx):
          return self.atoms[ndx]

      def genMolGraph(self):
          adj_list_all_atms = self.getListOfPairs()
          g = mygraph.Graph(self.end)
          g.SetGraphByEdges(adj_list_all_atms)
          self.molgraph = g

      def setCycles(self,lengths=[3,4,5,6,7,8]):
          g = self.molgraph
          for i in lengths:
              g.findCyclesOfSize(i)
          self.molgraph = g

      def getAroCycles(self):
          def check_planarity(molobj,cyc):
              """Check if the ring is planar. If it is, then
                 added to the aro_cycles attribute of the object"""
              c = cyc
              len_cyc = len(cyc)
              planar = False
              np.random.shuffle(cyc)
              plane_1 = cyc[:3]
              plane_2 = cyc[3:]
              v1 = molobj.getAtom(plane_1[0]).coord
              v2 = molobj.getAtom(plane_1[1]).coord
              v3 = molobj.getAtom(plane_1[2]).coord
              dih_deg = np.array([dihedral(v1,v2,v3,molobj.getAtom(i).coord) for i in plane_2 ])
              dih_avg = np.mean(np.array([np.abs(np.cos(i*pi/180)) for i in dih_deg ]))
              if dih_avg > np.cos(11*pi/180): # Ring planarity less than 11 degrees
                 planar = True
              return planar

          def identify_potential(molobj,cyc,poss):
              """Identify atoms that are ring members but no with
                 the right hybridization state"""
              potential = []
              if len(cyc) - len(poss) < 3:
                 missing = [ i for i in c if not i in poss ]
                 for i in missing:
                     potential_pi = False
                     atm = molobj.getAtom(i)
                     expected_number = atm.hybrid
                     neig_in_ring = [ x for x in atm.neighbors if molobj.getAtom(x).index in poss ]
                     neig_not_ring = set([ molobj.getAtom(x).atype for x in atm.neighbors if not molobj.getAtom(x).index in c ])
                     if len(neig_not_ring) == 1 and 'H' in neig_not_ring:
                        potential_pi = True
                     if len(neig_not_ring) == 0:
                        potential_pi = True
                     if expected_number-len(atm.neighbors)<= 2 and len(neig_in_ring)==2 and potential_pi :
                        potential.append(atm)
              return potential

          cycles = [ np.asarray(c) for c in self.molgraph.cycles if len(c) in [5,6] ]
          for c in cycles:
              len_cyc = len(c)
              possibles = np.array([ a.index for a in self if a.atype in pi_electrons and a.index in c ])
              if not len(possibles) - len_cyc == 0:
                 potential = identify_potential(self,c,possibles)
              else:
                 potential = []

              if len_cyc - len(possibles) - len(potential) == 0 and check_planarity(self,c):
                 self.aro_cycles.append(np.sort(c))
              else:
                 pass

      def setAroData(self):
          aro_data = []
          ascending_order=self.aro_cycles
          ao = sorted(ascending_order,key=lambda c: c[0])
          setattr(self,'aro_cycles',ao)
          for c in self.aro_cycles:
              com = np.mean(np.array([ self.getAtom(a).coord for a in c ]),axis=0)
              v1 = normalize_vec(self.getAtom(c[0]).coord - com)
              ngb = [ a for a in c[1:] if a in self.getAtom(c[0]).neighbors ][0]
              v2 = normalize_vec(self.getAtom(ngb).coord - com)
              nm = normalize_vec(np.cross(v1,v2))
              aro_data.append((com,nm))
          self.aro_data = aro_data

      def getHbs(self):
          hyd_bds = []
          hydros = [ a for a in self if a.atype == "H" and self.getAtom(a.neighbors[0]).element in [ 'S','O','N','Se'] ]
          for a in hydros:
              h = a.coord
              donor_index = a.neighbors[0]
              donor = self.getAtom(donor_index)
              if not donor.atype in ['N.3','N.4','S.O2']:
                 d = donor.coord
                 dh = normalize_vec(h-d)
                 hyd_bds.append((donor_index,dh))
          self.donors = hyd_bds
          accepts = []
          for a in self:
              if a.atype in ['O.3','O.2','O.co2','S.3','S.2','Se']:
                 accepts.append(a.index)
                 continue
              if a.atype in ['N.2'] and a.nbonds < 3:
                 accepts.append(a.index)
                 continue
              if a.atype in ['N.ar','N.2'] and find_num_aro_bonds(a) < 3 and len(a.neighbors) == 2 :
                 accepts.append(a.index)
                 continue
          self.acceptors = accepts

      def getCharged(self):
          cations = [ a.index for a in self if a.VE-a.nlonepairs-a.nbonds > 0 and a.element != "C"]
          pre_anions = [ a.index for a in self if a.VE-a.nlonepairs-a.nbonds < 0 ]
          anions = []
          for a in [self.getAtom(i) for i in pre_anions]:
              if not a.atype == "O.co2":
                 anions.append(a.index)
              else:
                 bond = [ i for i in a.bonds if a.bonds[i] in ['ar','2'] ][0]
                 car = [ int(i) for i in bond.split('_') if not int(i) == a.index ][0]
                 if not car in anions:
                    anions.append(car)
          self.cations = cations
          self.anions = anions

      def getPhobic(self):
          alos  = [ a.index for a in self if a.element in ['I','B','Br','Cl','F']]
          aliph = []
          for a in self:
              element = a.element
              if element == 'C':
                 polar_nghb = [i for i in a.neighbors if not self.getAtom(i).element in ['C','H','I','B','Br','Cl','F']]
                 if len(polar_nghb) == 0 and len( [ a.index for c in self.aro_cycles if a.index in c ]) == 0  :
                    aliph.append(a.index)
          self.phobic = aliph+alos

      def writePharPhore(self,name_out):
          outfile = open(name_out,'w')
          sarn = len(self.donors)+len(self.acceptors)+1 # starting aromatic residue number 
          for g in self.multi_aro:
              outfile.write("REMARK connected aromatic cycles : %s\n"% ','.join([ str(c+sarn) for c in g]))
          sprn = sarn+len(self.aro_data)+len(self.cations)+len(self.anions)+1 # starting phobic residue number 
          for g in self.multi_pho:
              outfile.write("REMARK connected phobic atoms : %s\n"% ','.join([ str(c+sprn) for c in g]))
          a_count = 1
          r_count = 1
          for a,v in self.donors:
              d = self.getAtom(a)
              outfile.write("%s\n"%pdb_line(d.coord,a_count,"N",r_count,rs_nm="DON"))
              dh = d.coord+v
              a_count += 1
              outfile.write("%s\n"%pdb_line(dh,a_count,"H",r_count,rs_nm="DON"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1
          for a in self.acceptors:
              d = self.getAtom(a)
              outfile.write("%s\n"%pdb_line(d.coord,a_count,"O",r_count,rs_nm="ACC"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1
          for a,v in self.aro_data:
              outfile.write("%s\n"%pdb_line(a,a_count,"P",r_count,rs_nm="ARO"))
              nv = a+v
              a_count += 1
              outfile.write("%s\n"%pdb_line(nv,a_count,"H",r_count,rs_nm="ARO"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1
          for a in self.cations:
              d = self.getAtom(a)
              outfile.write("%s\n"%pdb_line(d.coord,a_count,"Na",r_count,rs_nm="POS"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1
          for a in self.anions:
              d = self.getAtom(a)
              outfile.write("%s\n"%pdb_line(d.coord,a_count,"Cl",r_count,rs_nm="NEG"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1
          for a in self.phobic:
              d = self.getAtom(a)
              outfile.write("%s\n"%pdb_line(d.coord,a_count,"C",r_count,rs_nm="PHO"))
              outfile.write("TER\n")
              r_count += 1
              a_count += 1

      def findMultiAromatic(self):
          """Gets atoms on each aro cycle, groups the adjacent ones and sets them into groups if any. Prints on terminal and 
          stores on text file with the same name as the .mol2 file
          """
          def find_adjacent(rings):
              """ This function take the list cycles stored in self.aro_cycles
                  and returns a list of pair of indexes (refering to the position 
                  the input list) that are adjacent.""" 
              #Goes through all but the last element in self.aro_cycles to avoid exceeding len(self.aro_cycles)
              pairs = []
              ch = -1
              for cycle_hunt in rings[:-1]:    
                  ch += 1
                  cp = 0
                  for cycle_prey in rings[1:]:
                      cp += 1
                      if cp>ch and any( atm in cycle_prey for atm in cycle_hunt ): # If one atom is shared, then 
                         pairs.append([ch,cp])                                     # two rings are adjacent. 
              return pairs 

          def group_adjacent( pairs ):
              """This function returns a list of cycles that are conected. 
                 The list are indexes of self.aro_cycles """
              groups = [pairs[0]]
              if len(pairs) >  1:    
                 for p in pairs[1:]:
                     flag = False     
                     cg = -1
                     for g in groups:
                         cg += 1
                         if any(c in g for c in p): # if cycle index c is already in the group g, then 
                            flag = True             # exit the loop and add c to g. Update the group list.
                            break                   # if c is not found in any g, then the pair containing c
                     if flag:                       # starts its own group.
                        g.append(p[0])               
                        g.append(p[1])              
                        g = list(set(g))            
                        groups[cg]= g              
                     else:
                        groups.append(p)

                 cycles_in_pairs = list(set([ x for p in pairs for x in p ]))
                 cycles_in_groups = [ x for g in groups for x in g ]
              
                 while not len(cycles_in_pairs) == len(cycles_in_groups): # each cycle index should belong to only one group. 
                    repeated = list_duplicates(cycles_in_groups)          # If a cycle index appears in more than one group,
                    for rp in repeated:                                   # then those groups should merge.   
                        membership = [ ndx for ndx in range(len(groups)) if rp in groups[ndx] ]
                        new_groups = [ groups[ndx] for ndx in range(len(groups)) if not ndx in membership ]
                        ng = list(set([ c for ndx in membership for c in groups[ndx] ]))
                        new_groups.append(ng)
                        groups = new_groups
                    cycles_in_groups = [ x for g in groups for x in g ]
              return groups

          if len(self.aro_cycles) > 1: # if the molecule has aro cycles
             list_of_adj = find_adjacent(self.aro_cycles)
             if len(list_of_adj):  # if there are adjacent cycles
                grouped = group_adjacent(list_of_adj)
             else:   
                grouped = []
          else:
             grouped = []
          self.multi_aro = grouped

      def findClusterHydrophobic(self):
          """Merge into one group all the hydrophobic atoms that are connected.
          """
          def group_phobic(phobics):
              atm = self.getAtom(phobics[0])
              groups[0] = [atm.ndx]+atm.neighbors
              clust_ndx = 1
              for a in phobics[1:]:
                  atm = self.getAtom(a)
                  net = [atm.ndx]+atm.neighbors
                  flag = False               # exit the loop and add c to g. Update the group list.
                  for k in groups:
                      g = groups[k]
                      if any(c in g for c in net): # if atom index c is already in the group g, then 
                         flag = True               # exit the loop and add c to g. Update the group list.
                         break                     # if c is not found in any g, then the network containing c
                  if flag:                         # starts its own group.
                     for c in net:
                         g.append(c)
                     g = list(set(g))            
                     groups[k]= g              
                  else:
                     groups[clust_ndx] = net
                     clust_ndx += 1

              atms_in_groups = [ x for clust_ndx in groups for x in groups[clust_ndx] ]
              while not len(phobic) == len(atms_in_groups):            # each atom index should belong to only one group. 
                 repeated = list_duplicates(atms_in_groups)            # If a atom index appears in more than one group,
                 for rp in repeated:                                   # then those groups should merge.   
                     membership = [ clust_ndx for clust_ndx in groups if rp in groups[clust_ndx] ]
                     new_groups = {}
                     to_merge = []
                     for clust_ndx in groups:
                         if  not clust_ndx in membership:
                             new_groups[clust_ndx] = groups[clust_ndx]
                         else:
                             to_merge += groups[clust_ndx]  
                     new_groups[membership[0]] = to_merge
                     groups = new_groups
                 atms_in_groups = [ x for clust_ndx in groups for x in groups[clust_ndx] ]
              return groups

          if len(self.phobic) > 1:
             list_of_cluster = group_phobic(self.phobic)
          else:
             list_of_cluster = []
          self.multi_pho = list_of_cluster 
