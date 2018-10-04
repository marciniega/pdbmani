import numpy as np
import numpy.linalg as np_linalg
# metodo necesario para importar la libreria desde el repositorio #
import sys
sys.path.append('math_tricks/')
from math_vect_tools import *
from operations import *
#                                     #                            #

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
          print ('%4s %s %3s %s'%(self.resi,self.resn,self.name,coord_print))

      def UpDateValue(self,property_to_change,new_value):
          """ Re-name a given attribute."""
          setattr(self, property_to_change, new_value)

class Residue(object):
      """Store residue info
              Remember that the Atom Class is accessed through Residue.
              Atoms are defined as attributes of the Residue."""
      def __init__(self, resi, resn, chain ,atomnames=None,atoms=None):
          self.resi = int(resi)
          self.resn = resn
          self.chain = chain
          if atomnames is None:
             self.atomnames = []
          if atoms is None:
             self.atoms = []

      def __iter__(self):
          return self

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

#int test_add_h2n ( RowVector3f* bb_atoms ) {
#     RowVector3f N = bb_atoms[1];
#     RowVector3f C = (bb_atoms[0]-N).normalized();
#     RowVector3f A = (bb_atoms[2]-N).normalized();
#     RowVector3f H = bb_atoms[3]-N;
#     RowVector3f T = A.cross(C);
#     Matrix3f equ;
#     Vector3f sol;
#     float angle = cos(118.2*PI/180);
#     equ << A[0],A[1],A[2],
#            C[0],C[1],C[2],
#            T[0],T[1],T[2];
#     sol << angle,angle,0.0;
#     Vector3f x = equ.colPivHouseholderQr().solve(sol);
#     RowVector3f res;
#     res = x.transpose();
#     std::cout << "The norm is:\n" << res.norm() << std::endl;
#     std::cout << "The \% error in the solution is:\n" << 100*(H-res).norm() << std::endl;
#     std::cout << "The H position is:\n" << N+res << std::endl;
#
#     return 0;
# }

class PdbStruct(object):
      """\n This class is defined to store a single pdb file.\n
      """
      def __init__(self,name,pdbdata=None,timefrm=None):
          self.name = name
          if pdbdata is None:
             self.pdbdata = []
          self.timefrm = timefrm

      def __iter__(self):
          return self

      def next(self): # Python 3: def __next__(self)
          if self.current > self.end:
             raise StopIteration
          else:
             self.current += 1
             return self.pdbdata[self.current - 1]

      def ResetResIter( self , start = 0):
          self.current = start

      def AddPdbData(self,pdb_name):
          """ Reads a pdb file and stores its information """
          if type(pdb_name) is str:
             data_pdb = open('%s'%pdb_name,'r').readlines()
          else: # it is already read
             data_pdb = pdb_name
          data = self.pdbdata
          tmp_resi = None
          res_count = -1
          atn_count = 0
          chains_in_data = {}
          for line in data_pdb:
              if line[:4] == 'ATOM':
                 atn_count += 1
                 line = line.split('\n')[0]
                 coord = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                 r_fact = float(line[60:66])
                 chain = "".join(line[20:22].split())
                 occup = float("".join(line[57:61].split()))
                 if line[21]==' ':
                    flag_no_chain = True
                 else:
                    flag_no_chain = False
                 line = line.split()
                 aton = line[2]
                 resn = line[3]
                 if flag_no_chain:
                    resi = line[4]
                 else:
                    resi = line[5]
                 element = line[-1]
                 if not resi == tmp_resi:
                    res_count += 1
                    data.append(Residue(resi,resn,chain))
                    tmp_resi = resi
                    residue = data[res_count]
                    ###
                    if not chain in chains_in_data.keys():
                       chains_in_data[chain] = res_count
                    ###
                 residue.AddAtom(aton,coord,r_fact,atn_count,occup,element)
          self.seqlength = len(data)
          self.current = 0
          self.end = self.seqlength
          self.chains = chains_in_data

      def PrintPdbInfo(self):
          """ Print information regarding the number of residues and frame"""
          print ("Number of residues and frame: %s    %s"%(self.seqlength ,self.timefrm))
          print ("Number of chains:             %s "%len(self.chains.keys()))

      def GetSeqInd(self):
          """ Retrive the sequence by residue number"""
          return [ int(i.resi) for i in self.pdbdata ]

      def GetResSeq(self):
          """ Retrive the sequence by residue name"""
          return [ i.resn for i in self.pdbdata ]

      def GetRes(self, idx):
          """ Retrive the residue object. As input the residue number should be given."""
          return [ res for res in self.pdbdata if int(res.resi) == idx ][0]

      def GetSeqRfact(self,atoms_to_consider=None):
          """ Return an array of the B-factors, each residue has an assingment.
              The assigned value corresponds to the average of B-factors of the
              considered atoms. The option atoms_to_consider take an array of atom name
              to consider in the assigment. Default is consider all atoms in residue"""

          #def check_list_of_atoms(res,atoms_list):
          #    atnames=''
          #    if atoms_list is None: # consider all the atoms
          #       atnames = res.atomnames
          #    elif type(atoms_list) is list: # check if a list is given
          #       atnames = atoms_list
          #    else:
          #      raise ListCheckError("The atoms_to_consider should be given as a list")
          #    return atnames

          data = []
          for res in self.pdbdata:
              res_rfact = 0
              atom_names = check_list_of_atoms(res,atoms_to_consider)
              for atm in atom_names:
                  if hasattr(res, atm):
                     atom_ob = getattr(res,atm)
                     res_rfact += atom_ob.rfact
                  else:
                      raise NoAtomInResidueError("The residue %s%s in structure %s does not have atom %s"%(res.resi,res.resn,self.name,atm))
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
             indexes = self.GetSeqInd()
          else:
             indexes = [ int(i) for i in setofinterest ]

          data = []
          atm = atoms_to_consider
          for idx in indexes:
              res = self.GetRes(idx)
              if hasattr(res, atm):
                 atom_ob = getattr(res,atm)
                 atom_pos = np.array(atom_ob.coord)
              else:
                 raise NoAtomInResidueError("The residue %s%s in structure %s does not have atom %s"%(res.resi,res.resn,self.name,atm))
              data.append(atom_pos)
          return np.array(data)

      def GetDiheMain(self):
          data = []
          for index in [ int(i.resi) for i in self.pdbdata ][1:-1]:
              try:
                 res = self.GetRes(index)
                 data.append(np.array([res.phi,res.psi]))
              except:
                 data.append(np.array([0.0,0.0]))
          return data

      def SetDiheMain(self):
          """ Assign the phi and psi angles residues in the molecule"""
          for index in [ int(i.resi) for i in self.pdbdata ][1:-1]:
              try:
                 res_pre = self.GetRes(index-1)
                 res = self.GetRes(index)
                 res_nex = self.GetRes(index+1)
              except:
                 continue
              phi = dihedral(getattr(res_pre,'C').coord,getattr(res,'N').coord,getattr(res,'CA').coord,getattr(res,'C').coord)
              psi = dihedral(getattr(res,'N').coord,getattr(res,'CA').coord,getattr(res,'C').coord,getattr(res_nex,'N').coord)
              self.GetRes(index).SetDihe(phi-180,psi-180)

      def SetRfactor(self , new_data):
          """ Asign external values to a pdb. Specific to put the new value in the B-factor value of the CA.
              DOTO: make it more general, to each atom??? """
          sequence = self.GetSeqInd()
          if not len(sequence) == len(new_data):
             raise NoSameLengthError(\
                        "The current structure has %s residues and data that you want to assign has %s !!!"%(len(sequence), len(new_data)))
          c = 0
          for index in sequence:
              self.GetRes(index).UpDateValue('rfact',new_data[c])
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
          out_data.write("REMARK %s writen by me. \n"%self.name)
          #for index in [ int(i.resi) for i in self.pdbdata ][1:-1]:
          for index in [ int(i.resi) for i in self.pdbdata ]:
              res = self.GetRes(index)
              for atn in res.atomnames:
                  atom = res.GetAtom(atn)
                  line = "ATOM"
                  line += "%7s"%atom.atom_number
                  line += "%5s"%atn
                  line += "%4s"%res.resn
                  line += "%2s"%res.chain
                  line += "%4s"%res.resi
                  line += "    "
                  line += "%8.3f"%atom.coord[0]
                  line += "%8.3f"%atom.coord[1]
                  line += "%8.3f"%atom.coord[2]
                  line += "%6.2f"%atom.occup
                  line += "%6.2f"%atom.rfact
                  line += "         "
                  line += "%3s"%atom.element
                  out_data.write("%s\n"%line)
          if flag_trj:
             out_data.write("ENDMDL\n")
             return out_data
          else:
             out_data.write("END\n")

class Trajectory(object):
      """Handles trajectory files. My trajectory file format. """
      def __init__(self, name, frames=None, length=None):
          self.name = name
          if frames is None:
             self.frames = []
             self.length = length

      def __iter__(self):
          return self

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
          self.ResetIter()
          for cnt in range(self.length):
              frm = self.next()
              outfile = frm.WriteToFile(outfile,True)
          outfile.write("END\n")

      def AddFrame(self,PdbStruct):
          self.frames.append(PdbStruct)

      def ReadTraj(self,file_to_read,every=1):
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
                         temp.AddPdbData(frame)
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

          for index in self.frames[0].GetSeqInd():
              temp_ob = self.frames[0].GetRes(index)
              resi = temp_ob.resi
              resn = temp_ob.resn
              chain = temp_ob.chain
              data.append(Residue(resi,resn,chain))
              residue = data[res_count]
              for atn in ['N','CA','C']:
                  atn_count += 1
                  temp_coor = []
                  temp_rfact = []
                  for i in set_frames:
                      fr = self.frames[i]
                      res = fr.GetRes(index)
                      atom = getattr(res,atn)
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

