import numpy as np
import numpy.linalg as np_linalg
from math_vect_tools import *

excluded_hetatms= ['SOL','HOH']
excluded_res= ['ACE','NME']

class Atom(object):
      #TODO check if coord has 3 dimensions.
      def __init__(self, name, coord, rfact, atom_number, occup, element,rfact_std=None):
          self.name = name
          self.coord = np.array(coord)
          self.rfact = float(rfact)
          self.atom_number = int(atom_number)
          self.occup = occup
          self.element = element
          self.rfact_std = rfact_std

      def print_info(self):
          coord_print = '%7.2f %7.2f %7.2f'%(self.coord[0],self.coord[1],self.coord[2])
          print('%4s %s %3s %s'%(self.resi,self.resn,self.name,coord_print))

      def UpDateValue(self,property_to_change,new_value):
          """ Re-name a given attribute."""
          setattr(self, property_to_change, new_value)

      def writePdbAtom(self,resn,resc,resi,atomn,res_type="ATOM"):
          """ Write a line in pdb format.
          Example of line:
          0         1         2         3         4         5         6         7
          01234567890123456789012345678901234567890123456789012345678901234567890123456789
          ATOM   1855  C   GLU D 250     -16.312 -74.893  -0.456  1.00133.59           C
          """
          line = "%-6s"%res_type
          line += "%5s"%atomn 
          line += "  %-3s"%self.name #the whitespaces is important
          line += "%4s"%resn
          line += "%2s"%resc
          try:
              res_line = int(resi)
              line += "%4s"%resi
              line += "    "
          except ValueError:
              line += "%5s"%resi
              line += "   "
          line += "%8.3f"%self.coord[0]
          line += "%8.3f"%self.coord[1]
          line += "%8.3f"%self.coord[2]
          line += "%6.2f"%self.occup
          line += "%6.2f"%self.rfact
          line += "           "
          line += "%-3s"%self.element
          return "%s\n"%line

class Residue(object):
      """Store residue info
              Remember that the Atom Class is accessed through Residue.
              Atoms are defined as attributes of the Residue."""
      def __init__(self, resi, resn, chain, resx ,atomnames=None,atoms=None):
          self.resi = resi
          self.resn = resn
          self.chain = chain
          self.resx = resx # index on the pdbdata array
          if atomnames is None:
             self.atomnames = []
          if atoms is None:
             self.atoms = []
          self.chain_start = False
          self.chain_end = False

      def __iter__(self):
          return iter(self.atoms)

      def next(self): # Python 3: def __next__(self)
          if self.current > self.end:
             raise StopIteration
          else:
             self.current += 1
             return self.atomnames[self.current - 1]

      def ResetAtomIter( self , start = 0):
          self.current = start

      def PrintResSummary(self):
          """ Print residue information
                Resi   Resn   Chain   No.Atoms"""
          print ('Resi %4s Resn  %4s Chain %2s No.AToms  %2s'\
                %(self.resi,self.resn,self.chain,self.atomwithin))

      def AddAtom(self, name, coords, rfact, atom_number, occup, element,rfact_std=None):
          """ Add an atom information to current residue."""
          if rfact_std is None:
             self.atoms.append(Atom(name,coords,rfact, atom_number, occup , element))
             #setattr(self, name, Atom(name,coords,rfact, atom_number, occup , element))
          else:
             self.atoms.append(Atom(name,coords,rfact, atom_number, occup , element,rfact_std))
             #setattr(self, name, Atom(name,coords,rfact, atom_number, occup , element,rfact_std))
          self.atomnames.append(name)
          self.atomwithin = len(self.atomnames)
          self.current = 0
          self.end = self.atomwithin
      
      def delAtom(self,atomtodel):
          atpos= self.atomnames.index(atomtodel)
          nm = self.atomnames
          nm.pop(atpos)
          self.atomnames = nm
          self.atomwithin = len(self.atomnames)
          self.end = self.atomwithin
          aa = self.atoms
          aa.pop(atpos)
          self.atoms = aa

      def renameAtom(self,atomtorename,newname):
          atpos= self.atomnames.index(atomtorename)
          nm = self.atomnames
          nm[atpos] = newname 
          self.atomnames = nm
          setattr(self.atoms[atpos],'name',newname)

      def GetMainChainCoord(self):
          """ Get coordinates of the mainchain atoms (N,CA,C) as numpy array."""
          return np.array([self.N.coord,self.CA.coord,self.C.coord])

      def SetDihe(self,phi,psi):
          """ Assign phi and psi dihedral values to current residue."""
          setattr(self,'phi', float(phi))
          setattr(self,'psi', float(psi))

      def UpDateValue(self,property_to_change,value):
          """ Re-assign values associated with a given attribute.
              Remember that the Atom Class is accessed through Residue.
              Atoms are defined as attributes of the Residue."""
          for atom_in_res in self.atomnames:
              current_atom = getattr(self,atom_in_res)
              if property_to_change == 'coord':
                 setattr(current_atom,property_to_change, value)
              else:
                 setattr(current_atom,property_to_change, float(value))

      def GetAtom(self,atom_name):
          return [self.atoms[i] for i in range(self.atomwithin) if self.atomnames[i] == atom_name ][0]

      def UpDateName(self,property_to_change,new_name):
          """ Re-name a given attribute."""
          setattr(self, property_to_change, new_name)

      def Add_h2n(self,c_prev):
          n = self.GetAtom('N').coord
          c = normalize_vec(c_prev - n)
          a = normalize_vec(self.GetAtom('CA').coord - n)
          t = np.cross(c,a)
          angle = np.cos(118.2*np.pi/180.)
          equ = np.array([[a[0],a[1],a[2]],\
                          [c[0],c[1],c[2]],\
                          [t[0],t[1],t[2]]])
          sol = np.array([[angle],[angle],[0.0]])
          h = np_linalg.solve(equ,sol)
          pos = n+h.transpose()[0]
          self.AddAtom('H',pos, '0.0', 0 , '0.0','H')

      def getGeometricCenter(self,section='all',consider=None):
          if section == 'all':
             exclude = []
          if section == 'side_chain':
             exclude = ['N','CA','C','O']
          if consider == None:
             return np.mean(np.array([ i.coord for i in self.atoms if not i.name in exclude ]),axis=0)
          else :
             return np.mean(np.array([ i.coord for i in self.atoms if i.name in consider ]),axis=0)

      def getHDs(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          dict_one_don = {'SER':('HG','OG')  ,'TYR':('HH','OH')   , 'THR':('HG1','OG1'),
                          'CYS':('HG','SG')  ,'GLU':('HE2','OE2') , 'ASP':('HD2','OD2'),
                          'TRP':('HE1','NE1'),'ASN':('HD21','ND2'),'GLN':('HE21','NE2'),
                          'ARG':('HE','NE')}
          dict_two_don = {'ASN':('HD22','ND2'),'GLN':('HE22','NE2')}

          def set_h_mvec(h_name,d_name):
              h = self.GetAtom(h_name).coord
              d = self.GetAtom(d_name).coord
              dh = normalize_vec(h-d)
              return (h,dh)

          hdons = []
          n_hdons = []

          if self.resn in [ 'PRO' ]:
             pass
          else:
              if "H" in self.atomnames: # N-term case does not have it
                  hdons.append(set_h_mvec("H","N"))
                  n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"H"))

          if self.resn in dict_one_don :
             if not self.resn in ['ASP','GLU','CYS']:
                hdons.append(set_h_mvec(dict_one_don[self.resn][0],dict_one_don[self.resn][1]))
                n_hdons.append("%s_%s_%s"%(self.resn,self.resi,dict_one_don[self.resn][0]))
             else:
                flag = False
                if self.resn == 'ASP' and 'HD2' in self.atomnames: 
                    flag = True
                if self.resn == 'GLU' and 'HD2' in self.atomnames: 
                    flag = True
                if self.resn == 'CYS' and 'HG' in self.atomnames: 
                    flag = True
                if flag:    
                   hdons.append(set_h_mvec(dict_one_don[self.resn][0],dict_one_don[self.resn][1]))
                   n_hdons.append("%s_%s_%s"%(self.resn,self.resi,dict_one_don[self.resn][0]))

          if self.resn in dict_two_don:
             hdons.append(set_h_mvec(dict_two_don[self.resn][0],dict_two_don[self.resn][1]))
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,dict_two_don[self.resn][0]))

          if self.resn[:2] == 'HI':
              if 'HD1' in self.atomnames: 
                  hdons.append(set_h_mvec("HD1","ND1"))
                  n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HD1"))
              if 'HE2' in self.atomnames:
                  hdons.append(set_h_mvec("HE2","NE2"))
                  n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HE2"))

          if self.resn == 'ARG':
             hdons.append(set_h_mvec("HH21","NH2"))
             hdons.append(set_h_mvec("HH22","NH2"))
             hdons.append(set_h_mvec("HH11","NH1"))
             hdons.append(set_h_mvec("HH12","NH1"))
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HH21"))
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HH22"))
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HH11"))
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"HH12"))
          if debug:
             return n_hdons
          else:
             #return hdons
             return [(hdons[i],n_hdons[i]) for i in range(len(n_hdons))]

      def getHDsNoH(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          dict_one_don = {'SER':('HG','OG')  ,'TYR':('HH','OH')   , 'THR':('HG1','OG1'),
                          'CYS':('HG','SG')  ,'GLU':('HE2','OE2') , 'ASP':('HD2','OD2'),
                          'TRP':('HE1','NE1'),'ASN':('HD21','ND2'),'GLN':('HE21','NE2'),
                          'ARG':('HE','NE')}
          dict_two_don = {'ASN':('HD22','ND2'),'GLN':('HE22','NE2')}

          def set_h_mvec(h_name,d_name):
              h = self.GetAtom(h_name).coord
              d = self.GetAtom(d_name).coord
              dh = normalize_vec(h-d)
              return (h,dh)

          hdons = []
          n_hdons = []

          if self.resn in [ 'PRO' ]:
             pass
          else:
              hdons.append(self.GetAtom("N").coord)
              n_hdons.append("%s_%s_%s"%(self.resn,self.resi,"N"))

          if self.resn in dict_one_don:
             heavyatomname = dict_one_don[self.resn][1]
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))

          if self.resn in dict_two_don:
             heavyatomname = dict_two_don[self.resn][1]
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))

          if self.resn[:2] == 'HI':
             heavyatomname = 'ND1'
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))
             heavyatomname = 'NE2'
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))

          if self.resn == 'ARG':
             heavyatomname = 'NH1'
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))
             heavyatomname = 'NH2'
             hdons.append(self.GetAtom(heavyatomname).coord)
             n_hdons.append("%s_%s_%s"%(self.resn,self.resi,heavyatomname))
          if debug:
             print(n_hdons)
          else:
             self.donors = [(hdons[i],np.array([0.,0.,0.])) for i in range(len(n_hdons))]

      def getHAcceptors(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          dict_one_acc = {'SER':'OG','TYR':'OH','THR':'OG1',
                          'CYS':'SG','GLU':'OE1','ASP':'OD1',
                          'ASN':'OD1','GLN':'OE1','HIS':'ND1'}

          dict_two_acc = {'GLU':'OE2','ASP':'OD2','HIS':'NE2'}

          haccs =[]
          n_haccs =[]

          haccs.append(self.GetAtom("O").coord)
          n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"O"))

          if self.resn in dict_one_acc :
             haccs.append(self.GetAtom(dict_one_acc[self.resn]).coord)
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,dict_one_acc[self.resn]))
          if self.resn in dict_two_acc :
             haccs.append(self.GetAtom(dict_two_acc[self.resn]).coord)
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,dict_two_acc[self.resn]))
          if debug:
             print(n_haccs)
          else:
             self.acceptors = [haccs[i] for i in range(len(n_haccs))]

      def getHAcceptorsPDB(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          dict_one_acc = {'SER':('OG','CB'),'TYR':('OH','CZ'),'THR':('OG1','CB'),
                          'CYS':('SG','CB'),'GLU':('OE1','CD'),'ASP':('OD1','CG'),
                          'ASN':('OD1','CG'),'GLN':('OE1','CD'),}
          dict_two_acc = {'GLU':('OE2','CD'),'ASP':('OD2','CG')}
          
          def h2n(n_coord,neigh_1,neigh_2,tar_angle):
              n1 = normalize_vec(neigh_1 - n_coord)
              n2 = normalize_vec(neigh_2 - n_coord)
              t = np.cross(n1,n2)
              angle = np.cos(tar_angle*np.pi/180.)
              equ = np.array([[n2[0],n2[1],n2[2]],\
                              [n1[0],n1[1],n1[2]],\
                              [ t[0], t[1], t[2]]])
              sol = np.array([[angle],[angle],[0.0]])
              h = np_linalg.solve(equ,sol)
              pos = n_coord+h.transpose()[0]
              return pos

          def set_h_mvec(h_name,d_name):
              h = self.GetAtom(h_name).coord
              d = self.GetAtom(d_name).coord
              dh = normalize_vec(h-d)
              return (h,dh)

          def lp2carbonyl(c,ca,o):
              n1 = normalize_vec(ca - c)
              n2 = normalize_vec(o - c)
              t = normalize_vec(np.cross(n1,n2))
              equ = np.array([[n2[0],n2[1],n2[2]],\
                              [n1[0],n1[1],n1[2]],\
                              [ t[0], t[1], t[2]]])
              dev = (angle(n1,n2)*180./np.pi)-120.0
              ang = np.cos(60.*np.pi/180.)
              ang1 = np.cos((60.+dev)*np.pi/180.)
              ang2 = np.cos((180+dev)*np.pi/180.)

              sol1 = np.array([[ang1],[ang],[0.0]])
              h1 = np_linalg.solve(equ,sol1)
              pos1 = o+h1.transpose()[0]
              #print( 'ATOM      4  O   MET A  10    '+''.join( ['%8.3f'%i for i in pos])+'  0.00 23.46           O   ')
              #r=angle(n1,h.transpose()[0])*180./np.pi
              sol2 = np.array([[ang],[ang2],[0.0]])
              h2 = np_linalg.solve(equ,sol2)
              pos2 = o+h2.transpose()[0]
              #print( 'ATOM      5  O   MET A  11    '+''.join( ['%8.3f'%i for i in pos])+'  0.00 23.46           O   ')
              #s=angle(n2,h.transpose()[0])*180/np.pi
              #print("%.2f  %.2f"%(r,s))
              return [(pos1,normalize_vec(pos1-o)),(pos2,normalize_vec(pos2-o))] 

          haccs =[]
          n_haccs =[]

          if "O" in self.atomnames: # N-term case does not have it
             cblp = lp2carbonyl(self.GetAtom('C').coord,self.GetAtom('CA').coord,self.GetAtom('O').coord)    
             haccs.append(cblp[0])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"O"))
             haccs.append(cblp[1])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"O"))

          if self.resn in dict_one_acc:
             haccs.append(set_h_mvec(dict_one_acc[self.resn][0],dict_one_acc[self.resn][1]))
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,dict_one_acc[self.resn][0]))
          if self.resn in dict_two_acc:
             haccs.append(set_h_mvec(dict_two_acc[self.resn][0],dict_two_acc[self.resn][1]))
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,dict_two_acc[self.resn][0]))

          if self.resn in ['ASP','ASN']:
             cblp = lp2carbonyl(self.GetAtom('CG').coord,self.GetAtom('CB').coord,self.GetAtom('OD1').coord)    
             haccs.append(cblp[0])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OD1"))
             haccs.append(cblp[1])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OD1"))
             if self.resn in ['ASP']:
                cblp = lp2carbonyl(self.GetAtom('CG').coord,self.GetAtom('CB').coord,self.GetAtom('OD2').coord)    
                haccs.append(cblp[0])
                n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OD2"))
                haccs.append(cblp[1])
                n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OD2"))

          if self.resn in ['GLU','GLN']:
             cblp = lp2carbonyl(self.GetAtom('CD').coord,self.GetAtom('CG').coord,self.GetAtom('OE1').coord)    
             haccs.append(cblp[0])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OE1"))
             haccs.append(cblp[1])
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OE1"))
             if self.resn in ['GLU']:
                cblp = lp2carbonyl(self.GetAtom('CD').coord,self.GetAtom('CG').coord,self.GetAtom('OE2').coord)    
                haccs.append(cblp[0])
                n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OE2"))
                haccs.append(cblp[1])
                n_haccs.append("%s_%s_%s"%(self.resn,self.resi,"OE2"))


          if self.resn in ['HIS','HID','HIE','HIP']:
             nd = h2n(self.GetAtom('ND1').coord,self.GetAtom('CG').coord,self.GetAtom('CE1').coord,120.0)     
             ne = h2n(self.GetAtom('NE2').coord,self.GetAtom('CE1').coord,self.GetAtom('CD2').coord,120.0)
             vn_nd =normalize_vec(nd-self.GetAtom('ND1').coord)
             vn_ne =normalize_vec(ne-self.GetAtom('NE2').coord)
             haccs.append((self.GetAtom('ND1').coord,vn_nd))
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,'ND1'))
             haccs.append((self.GetAtom('NE2').coord,vn_ne))
             n_haccs.append("%s_%s_%s"%(self.resn,self.resi,'NE2'))

          if debug:
             return n_haccs
          else:
             return [(haccs[i],n_haccs[i]) for i in range(len(n_haccs))]

      def getCations(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          atoms_charged = []
          n_atoms_charged = []
          if self.resn == 'ARG':
             atoms_charged.append(np.array([ i.coord for i in self if i.name in ['CZ','NH1','NH2']]).mean(axis=0))
             n_atoms_charged.append('%s_%s_%s'%(self.resn,self.resi,'CZ'))
          if self.resn == 'LYS':
             atoms_charged.append(self.GetAtom('NZ').coord)
             n_atoms_charged.append('%s_%s_%s'%(self.resn,self.resi,'NZ'))
          if self.resn in [ 'HIS', 'HIP'] and len([1 for i in self if i.name in ['HD1','HE2']]) == 2 :
             atoms_charged.append(np.array([ i.coord for i in self if i.name in ['ND1','NE2']]).mean(axis=0))
             n_atoms_charged.append('%s_%s_%s'%(self.resn,self.resi,'ND1'))
          if self.chain_start:
             atom_charge.append(self.GetAtom('N').coord)
             n_atoms_charged.append('%s_%s_%s'%(self.resn,self.resi,'N'))
          if debug:
             return n_atoms_charged 
          else:
             #return atoms_charged
             return [(atoms_charged[i],n_atoms_charged[i]) for i in range(len(n_atoms_charged))]

      def getCationsNoH(self,debug=False):
          """ Histide is not included """
          """ Name convection as found in gromacs amber99sb forcefield """
          dic_cations = { 'ARG': ['CZ','NH1','NH2'],
                          'LYS': ['NZ'],
                          #'HIS' : ['ND1','NE2'],
                          'HIP' : ['ND1','NE2'],
                          'end_1': ['N'],
                        }
          if not self.resn in dic_cations.keys() and not self.chain_start:
             self.cations = None
             return 
          if self.resn in dic_cations.keys():
             atoms_charged_info = dic_cations.get(self.resn,None)
             posi_info = getGeometricCenter(consider=atoms_charged_info)
             name_info = '%s_%s_%s'%(self.resn,self.resi,atoms_charged_info[0])
             if debug:
                print(name_info)
             else:
                self.cations = [posi_info]
          if self.chain_start:
             end_case = 'end_1'
             atoms_charged_info = dic_cations.get(end_case,None)
             posi_info = self.getGeometricCenter(consider=atoms_charged_info)
             name_info = '%s_%s_%s'%(self.resn,self.resi,atoms_charged_info[0])
             if debug:
                print(name_info)
             else:
                 try:
                    self.cations.append(posi_info)
                 except AttributeError:
                    self.cations = [posi_info]

      def getAnions(self,debug=False):
          """ Name convection as found in gromacs amber99sb forcefield """
          dic_anions = { 'GLU': ['CD','OE1','CE2'],
                         'ASP': ['CG','OD1','OD2'],
                         'end_1': ['C','OC1','OC2'],
                         'end_2': ['C','O','OXT'],
                        }
                  
          if not self.resn in dic_anions.keys() and not self.chain_end:
             self.anions = None
             return 
          if self.resn in dic_anions.keys():
             atoms_charged_info = dic_anions.get(self.resn,None)
             posi_info = getGeometricCenter(consider=atoms_charged_info)
             name_info = '%s_%s_%s'%(self.resn,self.resi,atoms_charged_info[0])
             if debug:
                print(name_info)
             else:
                self.anions = [posi_info]
          if self.chain_end:
             try:
                self.GetAtom('OC1')
                end_case = 'end_1'
             except IndexError:
                end_case = 'end_2'
             atoms_charged_info = dic_anions.get(end_case,None)
             posi_info = self.getGeometricCenter(consider=atoms_charged_info)
             name_info = '%s_%s_%s'%(self.resn,self.resi,atoms_charged_info[0])
             if debug:
                print(name_info)
             else:
                 try:
                    self.anions.append(posi_info)
                 except AttributeError:
                    self.anions = [posi_info]

      def getAromaticData(self,debug=False):
          """
          Returns the GeometricCenter and NormalVector of the benzene of Aromatic Residues.
          """
          dic_aro_atms = { 'TRP6': ['CZ2','CH2','CZ3','CE3','CD2','CE2'],
                           'TRP5': ['NE1','CD1','CG','CD2','CE2'],
                           'TYR' : ['CG','CD1','CD2','CE1','CE2','CZ'],
                           'PHE' : ['CG','CD1','CD2','CE1','CE2','CZ'],
                           'HI'  : ['CG','CD2','NE2','CE1','ND1']
                         }
          if not self.resn in ['TRP','TYR','PHE','HIS','HIE','HID','HIP']:
             self.aro_data = None
             return
            
          n_aro_data = []
          aro_data = []

          if self.resn == 'TRP':
             aro_atms = [ dic_aro_atms['TRP6'] , dic_aro_atms['TRP5'] ]
             n_aro_data.append("%s_%s"%('TRP6',self.resi) )
             n_aro_data.append("%s_%s"%('TRP5',self.resi) )
          if self.resn.find('HI')==0:
             aro_atms = [ dic_aro_atms['HI'] ]
             n_aro_data.append("%s_%s"%(self.resn,self.resi) )
          if self.resn in ['TYR','PHE'] :
             aro_atms = [ dic_aro_atms[self.resn] ]
             n_aro_data.append("%s_%s"%(self.resn,self.resi))
             
          for aa in aro_atms:
              ring_coord = []
              for atom in self.atoms:
                  if atom.name in aa :
                     ring_coord.append(atom.coord)
              ring_coord = np.array(ring_coord)
              if len(ring_coord) != len(aa):
                 raise IncompleteResidue("Aromatic Residue incomplete!!!!")
              center = np.mean(ring_coord,axis=0)
              cross = np.cross(ring_coord[0]-center,ring_coord[1]-center)
              data = (center,normalize_vec(cross))
              aro_data.append(data)
          if debug:
             print(n_aro_data)
          else:
             self.aro_data = [aro_data[i] for i in range(len(n_aro_data))]

      def getPhobic(self,debug=False):
          """ Function to identify the pharmacophoric features
                of a residue."""
          dict_of_res = {'THR':['CG2'],
                         'HIS':['CB'],
                         'ASN':['CB'],
                         'ASP':['CB'],
                         'ALA':['CB'],
                         'TYR':['CB'],
                         'PHE':['CB'],
                         'TRP':['CB'],
                         'ARG':['CB','CG'],
                         'GLU':['CB','CG'],
                         'GLN':['CB','CG'],
                         'PRO':['CB','CG'],
                         'LYS':['CB','CG','CD'],
                         'VAL':['CB','CG1','CG2'],
                         'ILE':['CB','CG1','CG2','CD1'],
                         'LEU':['CB','CG','CD1','CD2'],
                         'MET':['CB','CG','SD','CE']
                         }
          if debug:
             print(dict_of_res.get(self.resn,None))
          else:
             self.phobic = dict_of_res.get(self.resn,None)

      def GetPharPhoreLines(self,a_count=1,r_count=1):
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

          list_of_phobics = ['ILE','VAL',
                             'LEU','PRO',
                             'MET','GLN',
                             'GLU','ARG',
                             'LYS']
          out_lines = []
          for v,a in self.donors:
              line = pdb_line(v,a_count,"N",r_count,rs_nm="DON")
              #print("%s"%line)
              out_lines.append(line)
              dh = v+a
              if all( i != 0 for i in a ):
                a_count += 1
                line = pdb_line(dh,a_count,"H",r_count,rs_nm="DON")
                #print("%s"%line)
                out_lines.append(line)
              r_count += 1
              a_count += 1
          if self.acceptors != None:
             for v in self.acceptors:
                 line = pdb_line(v,a_count,"O",r_count,rs_nm="ACC")
                 #print("%s"%line)
                 out_lines.append(line)
                 r_count += 1
                 a_count += 1
          if self.aro_data != None:
             for a,v in self.aro_data:
                 line = pdb_line(a,a_count,"P",r_count,rs_nm="ARO")
                 #print("%s"%line)
                 out_lines.append(line)
                 nv = a+v
                 a_count += 1
                 line = pdb_line(nv,a_count,"H",r_count,rs_nm="ARO")
                 #print("%s"%line)
                 out_lines.append(line)
                 nv = a-v
                 a_count += 1
                 line = pdb_line(nv,a_count,"H",r_count,rs_nm="ARO")
                 #print("%s"%line)
                 out_lines.append(line)
                 r_count += 1
                 a_count += 1
          if self.cations != None:
             for v in self.cations:
                 line = pdb_line(v,a_count,"Na",r_count,rs_nm="POS")
                 #print("%s"%line)
                 out_lines.append(line)
                 r_count += 1
                 a_count += 1
          if self.anions != None:
             for v in self.anions:
                 line = pdb_line(v,a_count,"Cl",r_count,rs_nm="NEG")
                 #print("%s"%line)
                 out_lines.append(line)
                 r_count += 1
                 a_count += 1
          if self.phobic != None:
             for c,a in enumerate(self.phobic):
                 d = self.GetAtom(a)
                 line = pdb_line(d.coord,a_count,"C",r_count,rs_nm="PHO")
                 #print("%s"%line)
                 out_lines.append(line)
                 a_count += 1
                 if self.resn in list_of_phobics:
                     continue
                 r_count += 1
             if self.resn in list_of_phobics:
                 r_count += 1
          return out_lines,a_count,r_count

      def writePdbRes(self,resi='',atmi=''):
          if resi == '':
             resi = self.resi 
          atn_count = -1
          line = ''
          for atom in self:
              atn_count += 1
              if atmi == '':
                 atmn = atom.atom_number
              else:
                 atmn = atmi + atn_count
              line += atom.writePdbAtom(self.resn,self.chain,resi,atmn)
          return line


class PdbStruct(object):
      """\n This class is defined to store a single pdb file.\n
      """
      def __init__(self,name,pdbdata=None,timefrm=None,ligandname=None):
          self.name = name
          if pdbdata is None:
             self.pdbdata = []
          self.timefrm = timefrm
          self.ligandname = ligandname
          self.chains_starts_resx = []

      def __iter__(self):
          return iter(self.pdbdata)

      def next(self): # Python 3: def __next__(self)
          if self.current > self.end:
             raise StopIteration
          else:
             self.current += 1
             return self.pdbdata[self.current - 1]

      def ResetResIter( self , start = 0):
          self.current = start

      def AddPdbData(self,pdb_name,target_atoms=[]):
          """ Reads a pdb file and stores its information """
          if type(pdb_name) is str:
             data_pdb = open('%s'%pdb_name,'r').readlines()
          else: # it is already read
             data_pdb = pdb_name
          if len(target_atoms)>0:
             check_target_atoms = True
          else:
             check_target_atoms = False
          data = self.pdbdata
          tmp_resi = None
          res_count = -1
          atn_count = 0
          chains_in_data = {}
          flag=False
          for line in data_pdb:
              if check_target_atoms and not line[12:17].replace(" ","") in target_atoms:
                 continue
              if line[:4] == 'ATOM' or ( line[:6] == 'HETATM' and not line[17:20] in excluded_hetatms):
                 atn_count += 1
                 line = line.split('\n')[0]
                 coord = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                 r_fact = float(line[60:66])
                 chain = "".join(line[20:22].split())
                 occup = float("".join(line[56:60].split()))
                 resi = line[22:27].replace(" ","")
                 resn = line[17:20]
                 aton = line[12:17].replace(" ","")
                 chain = line[21]
                 element = line[76:].replace(" ","")

                 if not resi == tmp_resi:
                    res_count += 1
                    data.append(Residue(resi,resn,chain,res_count))
                    tmp_resi = resi
                    residue = data[res_count]
                    if not chain in chains_in_data.keys():
                       chains_in_data[chain] = 1
                       self.chains_starts_resx.append(res_count)
                       residue.chain_start = True
                    else:
                       chains_in_data[chain] += 1
                       if not res_count == 0:
                          data[res_count-1].chain_end = False
                          residue.chain_end = True
                 if aton[0].isdigit():
                    aton = aton[1:]+aton[0]
                 residue.AddAtom(aton,coord,r_fact,atn_count,occup,element)

          self.seqlength = len(data)
          self.current = 0
          self.end = self.seqlength
          self.chains = chains_in_data

      def printPdbInfo(self):
          """ Print information regarding the number of residues and frame"""
          print("Number of residues and frame: %s    %s"%(self.seqlength ,self.timefrm))
          print("Number of chains:             %s "%len(self.chains.keys()))

      def GetResSeq(self):
          """ Retrive the sequence by residue name"""
          return [ i.resn for i in self.pdbdata ]

      def GetRes(self, resi , chain):
          """ Retrive the residue object. As input the residue identifier number should be given."""
          return [ res for res in self.pdbdata if res.resi == str(resi) and res.chain == chain ][0]

      def GetRex(self, idx):
          """ Retrive the residue object. As input the residue index should be given."""
          return self.pdbdata[idx]

      def GetLig(self):
          """ Retrive the residue object. As input the residue number should be given."""
          return [ res for res in self.pdbdata if res.resn == self.ligandname ][0]

      def getChain(self, target_chain):
          """ Retrive data from a specific chain. As input the residue number should be given."""
          if not target_chain in self.chains:
              print("The chain %s was not found in the structure."%target_chain)
              sys.exit()
          return [ res for res in self.pdbdata if res.chain == target_chain ]

      def GetSeqRfact(self,atoms_to_consider=None):
          """ Return an array of the B-factors, each residue has an assingment.
              The assigned value corresponds to the average of B-factors of the
              considered atoms. The option atoms_to_consider take an array of atom name
              to consider in the assigment. Default is consider all atoms in residue"""
          data = [ ]
          for res in self.pdbdata:
              res_rfact = 0
              if not atoms_to_consider == None:
                 atom_names = atoms_to_consider
              else:
                  atoms_names = res.atomnames
              for atm in atom_names:
                  try:
                      atom_ob = res.GetAtom(atm)
                      res_rfact += atom_ob.rfact
                  except IndexError:
                      res_rfact += 0.0
              data.append(res_rfact/float(len(atom_names)))
          return data

      def GetAtomPos(self,atoms_to_consider='CA', setofinterest=None):
          """ Return an array with the coordinates of the requested main chain atoms.
              Default is consider the c-alpha atom and all the residues"""
          # checking atom name
          if atoms_to_consider in ['N','CA','C','O']:
             pass
          else:
             raise NoValidAtomNameError

          # checking which residues
          try:
             assert not isinstance(setofinterest, basestring) # checking no string
          except:
             raise SystemExit("Input should be a list (the residues of interest)")

          if setofinterest == None:
             indexes = range(len(self.pdbdata))
          else:
             indexes = [ int(i) for i in setofinterest ]

          data = []
          atm = atoms_to_consider
          for res in self:
            if res.resx in indexes:
              if hasattr(res, atm):
                 atom_ob = getattr(res,atm)
                 atom_pos = np.array(atom_ob.coord)
              else:
                 raise NoAtomInResidueError("The residue %s%s in structure %s does not have atom %s"%(res.resi,res.resn,self.name,atm))
              data.append(atom_pos)
          return np.array(data)

      def GetDiheMain(self):
          data = []
          for index in [ int(i.resx) for i in self.pdbdata ][1:-1]:
              try:
                 res = self.GetRex(index)
                 data.append(np.array([res.phi,res.psi]))
              except:
                 data.append(np.array([0.0,0.0]))
          return data

      def SetDiheMain(self):
          """ Assign the phi and psi angles residues in the molecule"""
          self.GetRex(0).SetDihe(0.0,0.0)
          self.GetRex(self.seqlength-1).SetDihe(0.0,0.0)
          for index in np.arange(1,self.seqlength-1): # make correction for chains
              try:
                 res_pre = self.GetRex(index-1)
                 res = self.GetRex(index)
                 res_nex = self.GetRex(index+1)
              except:
                 continue
              phi = dihedral(res_pre.GetAtom('C').coord,res.GetAtom('N').coord,res.GetAtom('CA').coord,res.GetAtom('C').coord)
              psi = dihedral(res.GetAtom('N').coord,res.GetAtom('CA').coord,res.GetAtom('C').coord,res_nex.GetAtom('N').coord)
              if phi > 180:
                 phi = phi-360
              if psi > 180:
                 psi = psi-360
              self.GetRex(index).SetDihe(phi,psi)

      def SetRfactor(self , new_data):
          """ Asign external values to a pdb. Specific to put the new value in the B-factor value of the CA.
              DOTO: make it more general, to each atom??? """
          if not len(self.pdbdata) == len(new_data):
             raise NoSameLengthError(\
                        "The current structure has %s residues and data that you want to assign has %s !!!"%(len(self.pdbdata), len(new_data)))
          c = 0
          for res in self:
              res.UpDateValue('rfact',new_data[c])
              c += 1

      def RenameResidues(self, list_of_new_names):
          """ This just change the name, thus atom types remain."""
          if len(self.pdbdata) == len(list_of_new_names):
             pass
          else:
             raise SystemExit("The give list does not have the same size as the sequence")
          c = 0
          for res in self.pdbdata:
              res.UpDateName('resn',list_of_new_names[c])
              c += 1

      def UpdateCoord(self,newcoord):
           """ Update coordinates of protein structure """
           if not newcoord.shape[0] == self.pdbdata[-1].atoms[-1].atom_number:
              print(" The suggested new coordinates are not in the same shape as the current data.")
              print(" new : %s  current : %s "%(newcoord.shape,self.pdbdata[-1].atoms[-1].atom_number))
              raise SystemExit("Nothing was done!!!")
           i = 0
           for res in self:
                for atom in res.atoms:
                    setattr(atom,"coord",newcoord[i])
                    i += 1

      def WriteToFile(self,file_out_name=None,flag_trj=False):
          """ Write a structre back to a pdb file.
          Example of line:
          0         1         2         3         4         5         6         7
          01234567890123456789012345678901234567890123456789012345678901234567890123456789
          ATOM   1855  C   GLU D 250     -16.312 -74.893  -0.456  1.00133.59           C
          """
          if flag_trj:
             out_data = file_out_name
             out_data.write("MODEL\n")
          if file_out_name is None and not flag_trj:
             file_out_name = self.name
             out_data = open('%s.pdb'%file_out_name,'w')
          if file_out_name is not None and not flag_trj:
             out_data = open('%s.pdb'%file_out_name,'w')
          out_data.write("REMARK %s writen by me. \n"%self.name)
          for res in self:
              for atom in res:
                  line = "ATOM"
                  line += "%7s"%atom.atom_number
                  line += "%5s"%atom.name
                  line += "%4s"%res.resn
                  line += "%2s"%res.chain
                  line += "%4s"%res.resi
                  line += "    "
                  line += "%8.3f"%atom.coord[0]
                  line += "%8.3f"%atom.coord[1]
                  line += "%8.3f"%atom.coord[2]
                  line += "%6.2f"%atom.occup
                  line += "%6.2f"%atom.rfact
                  line += "           "
                  line += "%-3s"%atom.element
                  out_data.write("%s\n"%line)
          if flag_trj:
             out_data.write("ENDMDL\n")
             return out_data
          else:
             out_data.write("END\n")

      def getCoorArray(self,exclude=True):
          if exclude:
             return np.array([a.coord for res in self for a in res if not res.resn in excluded_res])
          else:
             return np.array([a.coord for res in self for a in res])

class Trajectory(object):
      """Handles trajectory files. My trajectory file format. """
      def __init__(self, name, frames=None ):
          self.name = name
          if frames is None:
             self.frames = []
             self.length = 0
             self.current = 0

      def __iter__(self):
          return iter(self.frames)

      def next(self): # Python 3: def __next__(self)
          if self.current > self.end:
             raise StopIteration

          else:
             self.current += 1
             return self.frames[self.current - 1]

      def ResetIter( self , start = 0):
          self.current = start

      def WriteTraj(self , out_name , str_frame = 1 ):
          outfile = open('%s.pdb'%out_name,'w')
          for cnt in range(self.length):
              frm = self.frames[cnt]
              outfile = frm.WriteToFile(outfile,True)
          outfile.write("END\n")

      def AddFrame(self,new_frame):
          frames = self.frames
          frames.append(new_frame)
          self.length = len(frames) # Ready
          self.end = self.length         # for
          self.current = self.end-1      # iterations
          self.frames = frames

      def ReadTraj(self,file_to_read,every=1,target_atoms=[]):
          fr = 0
          exfr = 0
          sav_fr = True
          flag = False
          #traj_file = open(file_to_read,'r').readlines()
          with open(file_to_read) as traj_file:
               for line in traj_file:
                   if line[:5] == "MODEL":
                      flag = True
                      if exfr == 0:
                         sav_fr = True
                         frame = []

                   elif line[:6] == "ENDMDL":
                      exfr += 1
                      sav_fr = False
                      if exfr == every:
                         fr += 1
                         temp = PdbStruct('frame_%s'%fr,timefrm=fr)
                         temp.AddPdbData(frame,target_atoms)
                         #temp.PrintPDBSumary()
                         self.frames.append(temp)
                         exfr = 0
                   else:
                      if sav_fr and flag:
                         frame.append(line)
               self.length = len(self.frames) # Ready
               self.current = 0               # for
               self.end = self.length         # iterations

          #for line in traj_file:
          #    if line[:5] == "MODEL":
          #       frame = []
          #    elif line[:6] == "ENDMDL":
          #       temp = PdbStruct('frame_%s'%fr)
          #       temp.AddPdbData(frame)
          #       self.frames.append(temp)
          #       fr += 1
          #    else:
          #       frame.append(line)

      def PrintTrajInfo(self):
          print ('This trajectory file : %s'%self.name)
          print ('has %s frames'%self.length)

      def GetFrame(self,frame):
          return self.frames[frame]


      #def IterFrames(self,intial=None,final=None):
      #    if initial == None:
      #       initial = 0
      #    if final == None:
      #       final = self.length
      #    imap(PrintPDBSumary())

      def GetAverageStruct(self,set_frames=None):
          if set_frames is None:
             set_frames = range(len(self.frames))
          elif not type(set_frames) is list:
             raise ListCheckError("The set_frame should be given as a list of frames to average")

          temp_pdb = PdbStruct('average')
          data = temp_pdb.pdbdata
          res_count = 0
          atn_count = 0
          store_dist_data = {}
          for j in set_frames:
              b_fact_data = self.frames[j].GetSeqRfact(['N','CA','C'])
              store_dist_data[j] = (np.average(b_fact_data),np.std(b_fact_data))

          for temp_ob in self.frames[0]:
              resi = temp_ob.resi
              resn = temp_ob.resn
              chain = temp_ob.chain
              data.append(Residue(resi,resn,chain,res_count))
              residue = data[res_count]
              valid_atoms = ['N','CA','C']
              if residue.resn == 'ACE':
                 valid_atoms = ['C']
              if residue.resn == 'NME':
                 valid_atoms = ['N']
              for atn in valid_atoms:
                  atn_count += 1
                  temp_coor = []
                  temp_rfact = []
                  for i in set_frames:
                      fr = self.frames[i]
                      res = fr.GetRex(temp_ob.resx)
                      atom = res.GetAtom(atn)
                      temp_coor.append(getattr(atom,'coord'))
                      #temp_rfact.append((getattr(atom,'rfact') - store_dist_data[i][0])/store_dist_data[i][1])
                      temp_rfact.append((getattr(atom,'rfact')))
                  ave_coor = np.average(np.array(temp_coor),axis=0)
                  std_coor = np.std(np.array(temp_coor),axis=0)
                  std_coor = np.sqrt(np.sum([ i*i for i in std_coor ]))
                  element = atn[0]
                  bf_ave = np.average(temp_rfact)
                  bf_std = np.std(temp_rfact)
                  residue.AddAtom(atn,ave_coor,bf_ave,atn_count,std_coor,element,bf_std)
              res_count += 1
          self.average = temp_pdb

class Resi_plot(object):
      "store residue info"
      def __init__(self, resi, resn, diff):
           self.resi = int(resi)
           self.resn = resn
           self.diff = diff

class NoSameLengthError(Exception):pass
class DihedralGeometryError(Exception): pass
class AngleGeometryError(Exception): pass
class NoValidAtomNameError(Exception): pass
class ListCheckError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
class NoAtomInResidueError(Exception):
      def __init__(self, msg):
          self.msg = msg
      def __str__(self):
          return self.msg
class IncompleteResidue(Exception):
      def __init__(self, msg):
          self.msg = msg
      def __str__(self):
          return self.msg

