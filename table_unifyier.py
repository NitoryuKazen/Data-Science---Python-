from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp
import numpy as np
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as scp 
import scipy.interpolate as sci
import photutils.isophote as phi
import photutils.aperture as php
import numpy.ma as ma
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")


###############################################
rff,eta=parinfo=[[] for i in range(2)]#rff,a1-rrf
cluster=[]
tipo=[]
############################################
with open('pargal_L07_compact_astro.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_astro.dat','r')

with open('pargal_L07_compact_astro.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_astro.dat','r')

with open('../isophotest/graph_stats.dat','r') as inp2:
	ninp2=len(inp2.readlines())
inp2=open('../isophotest/graph_stats.dat','r')

with open('data_indiv_ecd.dat','r') as inp3:
	ninp3=len(inp3.readlines())
inp3=open('data_indiv_ecd.dat','r')

graph_rff=open('graph_stats_rff.dat','w')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls3=inp3.readline()
	ll3=ls3.split()
	cluster.append(ll1[0])
	rff.append(float(ll1[7]))
	eta.append(float(ll1[7]) - float(ll1[9]))
	tipo.append(ll3[7])
	
cl=[]
linha=[]
for j in range(0,ninp2):
	ls2=inp2.readline()
	ll2=ls2.split()
	cl.append(ll2[0])
	linha.append(ls2[:-2])	

for ii in range(len(cl)):
	graph_rff.write('%s %f %f %s \n'%(linha[ii],rff[cluster.index(cl[ii])],eta[cluster.index(cl[ii])], tipo[cluster.index(cl[ii])]))
