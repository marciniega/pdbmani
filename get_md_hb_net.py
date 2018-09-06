#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("math_tricks/")
sys.path.append("graphs/")
import read_pdb_tools as rpt
import math_vect_tools as mymath
import network_hb_from_md_traj as pdbnet

assert( len(sys.argv)>1)
infile = sys.argv[1]
outfile = open('hh_%s.txt'%infile.split('.')[0],'w')

trj = rpt.Trajectory("first")
trj.ReadTraj("%s"%infile, every=1)
trj.PrintTrajInfo()

while trj.current < trj.length:
      ref = trj.next()
      net = pdbnet.Network("first")
      net.set_nodes_on_frame(ref, mute=True)
      net.set_arists_on_nodes()
      #line = "# frame: %-4s  Nodes: %-6s  Edges: %-10s"%(trj.current ,net.count_nodes() ,net.count_arists())
      #print line
      #outfile.write('%s\n# Edge   Distance[A]\n'%line)
      net.write_arist_short(outfile, trj.current, short=True)

