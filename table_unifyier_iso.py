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
linha=[]
############################################
with open('iso_geral_values_coef_astro.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('iso_geral_values_coef_astro.dat','r')

with open('pargal_simples.dat','r') as inp3:
	ninp3=len(inp3.readlines())
inp3=open('pargal_simples.dat','r')

iso_tipo=open('iso_geral_values_coef_astro_tipo.dat','w')

for j in range(0,ninp3):
	ls3=inp3.readline()
	ll3=ls3.split()
	cluster.append(ll3[0])
	tipo.append(ll3[1])
cl=[]

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	linha.append(ls1[:-2])
	cl.append(ll1[0])	

for ii in range(len(cl)):
	iso_tipo.write('%s %s \n'%(linha[ii],tipo[cluster.index(cl[ii])]))
	
	
	
	
	
