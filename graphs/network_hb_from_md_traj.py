#!/usr/bin/env python2.7
import numpy as np
import sys
sys.path.append("/Users/marcelino/Things_Tools/scripts_python/pdb_manipulation")
sys.path.append("/Users/marcelino/Things_Tools/scripts_python/math_tricks")
import read_pdb_tools as rpt
import math_vect_tools as mymath

dictionary = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D', \
              'CYS':'C','GLU':'E','GLN':'Q','GLY':'G', \
              'HIS':'H','ILE':'I','LEU':'L','LYS':'K', \
              'MET':'M','PHE':'F','PRO':'P','SER':'S', \
              'THR':'T','TRP':'W','TYR':'Y','VAL':'V',\
              'HIE':'H','HID':'H','HIP':'H'}

class Network(object):
      # Network
      def __init__(self, name):
          self.name = name
          self.nodes = []
          self.arists = []
          self.end = 0
          self.current = 0

      def __iter__(self):
          return self

      def next(self): # Python 3: def __next__(self)
          if self.current > self.end:
             raise StopIteration
          else:
             self.current += 1
             return self.nodes[self.current - 1]

      def reset_iter( self , start = 0):
          self.current = start

      def count_nodes(self):
          length = len(self.nodes)
          self.num_nodes = length
          return length

      def count_arists(self):
          length = len(self.arists)
          self.num_arists = length
          return length

      def add_node(self,node):
          self.nodes.append(node)
          self.end += 1

      def add_arist(self,arist):
          self.arists.append(arist)

      def get_node_by_ndx(self,ndx):
          return self.nodes[ndx]

      def get_arist_by_ndx(self,ndx):
          return self.arist[ndx]

      def set_nodes_on_frame(self,rfr,mute=True):

          def get_offset(frame_obj):
              seq = frame_obj.GetResSeq()
              for i in range(len(seq)):
                  try:
                       dictionary[seq[i]]
                       break
                  except KeyError:
                       continue
              return frame_obj.GetSeqInd()[0]+i

          self.offset = get_offset(rfr)
          c_empty = 0
          while rfr.current < rfr.seqlength:
                res = rfr.next()
                try:
                    if dictionary[ res.resn ] == 'P':
                       h_pos = [0,0,0]
                except KeyError:
                    c_empty += 1
                    if not mute:
                       print "The residue "
                       res.PrintResSummary()
                       print "it is not a regular one.\
                             It was skipped. Plesea check if that it's ok."
                    continue
                try:
                    h_pos = res.GetAtom("H").coord ## add function that adds proton
                                                   ## in case it is not present.
                except IndexError:
                    res_prev = rfr.GetRes(int(res.resi)-1)
                    res.Add_h2n(res_prev.GetAtom("C").coord)
                    h_pos = res.GetAtom("H").coord
                    if not mute:
                       print "The residue "
                       res.PrintResSummary()
                       print "Does no have H.\
                              A hydrogen was added based on geometric definition."

                n_pos = res.GetAtom("N").coord
                ca_pos = res.GetAtom("CA").coord
                c_pos = res.GetAtom("C").coord
                o_pos = res.GetAtom("O").coord
                at_pos = np.array([n_pos,h_pos,ca_pos,c_pos,o_pos])
                r_info = "%s_%s_%s"%(res.resn,res.resi,res.chain)
                self.add_node(Node(r_info,rfr.current-1-c_empty,at_pos))

      def set_arists_on_nodes(self):
          for n in range(self.count_nodes()):
              c = self.get_node_by_ndx(n) # Acceptor
              for t in range(self.count_nodes()):
                  if t == n:
                     continue
                  cc = self.get_node_by_ndx(t) # Donor
                  if cc.name.find('PRO')==0:
                     continue
                  ch = 1./mymath.distance(cc.coord[1],c.coord[3]) # H from donor to C from aceptor
                  cn = 1./mymath.distance(cc.coord[0],c.coord[3]) # N from donor to C from aceptor
                  oh = 1./mymath.distance(cc.coord[1],c.coord[4]) # H from donor to O from aceptor
                  on = 1./mymath.distance(cc.coord[0],c.coord[4]) # N from donor to O from aceptor
                  const = 27.888 # 0.084 from two charges times 332 kcal/mol. DSSP definition.
                  energy = const*(on+ch-oh-cn)
                  n2o = c.coord[4]-cc.coord[0]
                  n2h = cc.coord[1]-cc.coord[0]
                  ang = mymath.angle(n2o,n2h)*180./np.pi
                  #if c.name == "ASP_85_A" and cc.name == "GLN_88_A" :
                  #   print c.name,cc.name,"  ",energy,"  ",ang

                  #if energy < -0.5 and ang < 63 and 1./on < 4.0:
                  if ang <= 30 and 1./on < 3.5:
                     cc_name = str(n)+"_"+str(t) # Accepted from Donor, A[n,t].
                     cc.conne[str(n)] = energy
                     #print c.name,cc.name,"  ",energy,"  ",ang
                     self.add_arist(Arist(cc_name,energy))

      def write_arist_short(self,of,iteration,short=True):
          for i in range(self.count_arists()):
                f = self.arists[i]
                fout = f.output_format(self.offset)
                if short:
                   line = "%-4s"%fout
                else:
                   line = "%-13s"%fout+"%.3f"%f.dist
                of.write('%s\n'%line+str(iteration)+" ")

      def write_conne_short(self,of,short=True):
          while self.current < self.end:
                f = self.next()
                sor = sorted([ int(i) for i in f.conne.keys() ])
                if short:
                   com = [ "%-4s"%i for i in sor ]
                else:
                   com = [ "%-4s"%i+"("+"%.3f"%f.conne[str(i)]+")" for i in sor ]
                line =  "%-12s"%f.name
                line += "%-6s : "%f.ndx
                line += ' '.join(com)
                outfile.write('%s\n'%line)

class Node(object):
      #Node of network
      def __init__(self, name, ndx ,coord):
          self.name  = name
          self.ndx   = ndx
          self.coord = np.array(coord)
          self.conne = {}

class Arist(object):
      #Connection of nodes
      def __init__( self, name , dist ):
          self.name = name
          self.dist = dist
      def output_format(self,os):
          n = self.name.split('_')
          return "%s_%s"%(int(n[0])+os,int(n[1])+os)

