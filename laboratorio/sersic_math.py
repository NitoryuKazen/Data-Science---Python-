from math import sin,cos,tan,pi,floor,log10,sqrt,atan2
import numpy as np
import pyfits
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
import scipy.optimize as scp 
import scipy.interpolate as sci
from matplotlib.patches import Ellipse
import datetime 
#############################
# SERSIC UNICO

def sersicfit(x,n,mag,effre):
	b = 2.*n - (1./3) + (4./(405*n)) + (46./(25515*(n**2))) + (131./(1148175*(n**3))) - (2194697./(30690717750*(n**4)))
#	mag = -2.5*(np.log10(sigma_e))

	return mag + 2.5*np.log10(np.exp(1.))*(b*(((x/effre)**(1./n))-1))

def sersicfitn4(x,mag,effre):
	b = 2*4 - (1./3) + (4/(405*4)) + (46/(25515*(4**2))) + (131/(1148175*(4**3))) - (2194697/(30690717750*(4**4)))
#	mag = -2.5*(np.log10(sigma_e))

	return mag + 2.5*np.log10(np.exp(1))*(b*(((x/effre)**(1./4))-1))

####################################

#SKY FUNCTION

def skyradfunc(x,a,b,c):
	return a*np.exp(-x/b)+c

def calc_sky(image,mask,maskb,xcenter,ycenter,cluster):
	vsky=[]
	dsky=[]
	for j in range(image.shape[0]):
		for i in range(image.shape[1]):
			if mask[j,i]==0 and maskb[j,i]==0:
				vsky.append(image[j,i])
				dsky.append(((j-ycenter)**2+(i-xcenter)**2)**0.5)
	if len(dsky) <= 3:
		skyvalue=+0
	else:
		popt,pcov=scp.curve_fit(skyradfunc,dsky,vsky,p0=[200,100,100])
		skyvalue =+ skyradfunc(np.max(dsky),*popt)
	return skyvalue

##################################################################33
# FLAG DA MASCARA
def masktest(image,mask):
	nmask=0.
	npix=0.
	for i in range(int(0.25*image.shape[0]),int(0.75*image.shape[0])):
		for j in range(int(0.25*image.shape[1]),int(0.75*image.shape[1])):
			if mask[i,j] == 0:
				npix+=1.
			if mask[i,j] == 1:
				nmask+=1
	flagvalue=0
	if nmask/(npix+nmask) >= 0.2:
		flagvalue+=1
	else:
		flagvalue+=0
	
	return flagvalue
#####################################################################
#FLAG DOS PARAMETROS DE SAIDA DO SERSICFIT 
def sersictest(n,mag,re,maxr_norm):
	
	###
	nvalue=0
	if 0. < n < 30.:
		nvalue=+0
	else:
		nvalue=+1
	###
	magvalue=0
	if 0. < mag < 30.:
		magvalue=+0
	else:
		magvalue=+1
	####
	revalue=0
	if 5. < re < 1000.:
		revalue=+0
	else:
		revalue=+1
		
	maxrrevalue=0 
	if maxr_norm < 1.8:
		maxrrevalue=+0
	else:
		maxrrevalue=+1
	
	return nvalue,magvalue,revalue,maxrrevalue
################################################################

def deltatest(delta):
	deltavalue=0
	if delta < 0.3:
		deltavalue=+0
	else:
		deltavalue=+1
	
	return deltavalue

################################################################
def cutaster(x):
 tempx=''
 for item in x:
 	if item != '*':
 		tempx+=item
 return tempx

###########################/home/andrelpkaipper/Documentos/Projetos###########

with open('/home/andrelpkaipper/Documentos/Projetos/L07/pargal_L07.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('/home/andrelpkaipper/Documentos/Projetos/L07/pargal_L07.dat','r')

with open('pargal_type_split_clean.dat','r') as inp2:
	ninp2=len(inp2.readlines())
inp2=open('pargal_type_split_clean.dat','r')

par_info=open('pre_parser_info_clean_r4.dat','a')

'''
NOTES

pre_parser_info.dat -> primeira rodada com os cortes anteriores sem modificacoes
pre_parser_info_2.dat -> segun da rodada com cortes corrigidos na mascara 
pre_parser_info_3.dat -> terceira rodada com o corte em raio efetivo iniciando em 5.
pre_parser_info_4.dat -> quarta rodada com o adicional corte para r/re na 1/4 <1.8 
pre_parser_info_5.dat -> quinta rodada baseada em cortes verticais
pre_parser_info_6.dat -> sexta rodada baseada em cortes verticais com 50 bcgs rigorosamente 

pre_parser_info_clean.dat -> novas rodadas com nenhuma espiral ou outro tipo.
pre_parser_info_clean_r4.dat -> nova rodada mas com o n=4. 
'''

typ=[]
rnmin=[]
rnmax=[]
rnminn4=[]
rnmaxn4=[]
fmask=[]
fn=[]
fmag=[]
fre=[]
frmax=[]
ok=[]
delta=[]
verf=open('pre_parser_info_clean_r4.dat','r')
for item in verf.readlines():	
	ok.append(item.split()[0])
	typ.append(item.split()[1])
	fmask.append(int(item.split()[2]))
	fn.append(int(item.split()[3]))
	fmag.append(int(item.split()[4]))
	fre.append(int(item.split()[5]))
	rnmin.append(float(item.split()[7]))
	rnmax.append(float(item.split()[8]))
	rnminn4.append(float(item.split()[18]))
	rnmaxn4.append(float(item.split()[19]))
	frmax.append(int(item.split()[6]))
	delta.append(int(item.split()[21]))
	
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls2=inp2.readline()
	ll2=ls2.split()
	tipo=ll2[1]
	tipozhao=ll2[2]
	cluster=ll1[0]
	re=float(ll1[2])
	ar=float(ll1[5])
	pa=float(ll1[6])
	
######################################################################################################
	if cluster in ok:
		pass
	else:
		print cluster
		os.chdir('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/')

		ajust1 = pyfits.getdata('ajust-bcg-r.fits',1)
		mask = pyfits.getdata('bcg_r_mask.fits')#MASK CONTAMINANTES
		mask_b = pyfits.getdata('bcg_r_mask_b.fits')#MASK BCG
		header = pyfits.getheader('ajust-bcg-r.fits',2)
		
		efre = float(header['1_RE'].split()[0].replace('*',''))
		magni = float(header['1_MAG'].split()[0].replace('*',''))
		ngg = float(header['1_N'].split()[0].replace('*',''))
		xc=float(header['1_XC'].split()[0].replace('*',''))
		yc=float(header['1_YC'].split()[0].replace('*',''))
		skyg=float(header['2_SKY'].split()[0])

		indice=0
		if ngg <= 4.:
			indice+=ngg
		else:
			indice+=4.
			
		sky=calc_sky(ajust1,mask,mask_b,xc,yc,cluster)
		flagmask=masktest(ajust1,mask)
		flagdelta=deltatest(float(ll1[10]))
		raio=[]
		fluxo=[]
		mi=[]
		mj=[]		
		gj=[]
		for i in range(0,ajust1.shape[0]):
			for j in range(0,ajust1.shape[1]):
				dtheta=-atan2(i-yc,j-xc)+(pa*3.141592/180.+3.141592/2.)
				cor=ar/((ar*cos(dtheta))**2+(sin(dtheta))**2)**0.5
				if 5. < np.sqrt((i-yc)**2 + (j-xc)**2) and mask[i,j] == 0:
					raio.append((float(np.sqrt((i-yc)**2 + (j-xc)**2))/cor))
					fluxo.append(ajust1[i,j]-sky)
					mj.append(j)
					mi.append(i)
					 
#################################################################
	# RAIO EFETIVO CORRIGIDO 

		rbin = [min(raio)]
		while rbin[-1] < max(raio):
			rbin.append(rbin[-1]*1.1)

		flux=[[] for _ in xrange(len(rbin)-1)]
		for q in range(len(raio)):	
			for a in range(len(rbin)-1):
				if (rbin[a]) <= (raio[q]) < (rbin[a+1]):
					flux[a].append(fluxo[q])
							
		flux_mag=[]
		rbin_true=[]
		ebar=[]
		for d in range(len(flux)-1):
			if len(flux[d]) != 0:
				if sum(flux[d]) > 0:
					flux_mag.append(22.5-2.5*np.log10(np.average(flux[d])))
					rbin_true.append((float(rbin[d])+float(rbin[d+1]))/2)
					ebar.append((2.5*np.std(flux[d]))/(np.log(10)*np.sqrt(len(flux[d]))*np.average(flux[d])))				
				else:
					break	
					
###############################################################################################
#OBTENCAO PERFIL DE SERSIC E INTERPOLACAO
		
		if len(rbin_true) <= 3:
			par_info.write('%s \t %s \t %i \t 1 \t 1 \t 1 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0\t 0 \t 0 \t 0 \t 0 \t 0 \t 0\n'%(cluster,tipo,flagmask))

			pass
		else:	
			rprofile,rcov = scp.curve_fit(sersicfit,rbin_true,flux_mag,p0=[indice,magni,efre],sigma=ebar,maxfev=100000000)
			rprofilen4,rcovn4 = scp.curve_fit(sersicfitn4,rbin_true,flux_mag,p0=[magni,efre],sigma=ebar,maxfev=100000000)
			
			rbin_norm=[]
			rbin_normn4=[]
			for r in range(len(rbin_true)):
				rbin_normn4.append(rbin_true[r]/rprofilen4[1])
				rbin_norm.append(rbin_true[r]/rprofile[2])
			normprofile,normcov = scp.curve_fit(sersicfitn4,rbin_normn4,flux_mag,p0=[rprofilen4[0],1.],sigma=ebar,maxfev=100000000)
			rremax=np.power(max(rbin_normn4),0.25)
			profileflags=sersictest(4.,rprofile[0],rprofile[1],rremax)
###############################################			
			par_info.write('%s \t %s \t %i \t %i \t %i \t %i \t %i \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %s \t %i \n'%(cluster,tipo,flagmask,profileflags[0],profileflags[1],profileflags[2],profileflags[3],min(rbin_norm),max(rbin_norm),4.,normprofile[0],normprofile[1],rprofile[0],rprofile[1],rprofile[2],4.,rprofilen4[0],rprofilen4[1],min(rbin_normn4),max(rbin_normn4),tipozhao,flagdelta))
			clval=open('/home/andrelpkaipper/Documentos/Projetos/photutils/sersic_profile/'+tipo+'/'+cluster+'_perfil_values.dat','w')
			for q in range(len(rbin_normn4)):
				clval.write('%f \t %f \t %f \n'%(rbin_normn4[q],flux_mag[q],ebar[q]))
			
			figr=plt.figure(figsize=(8,8))
			plt.errorbar(np.power(rbin_normn4,1),flux_mag,yerr=ebar,fmt='none',label='test')
			plt.scatter(np.power(rbin_normn4,1),flux_mag,c='gray',edgecolors='black')
			plt.plot(np.power(rbin_normn4,1),sersicfitn4(rbin_normn4,*normprofile),'r-')
			plt.ylim([max(flux_mag)+0.15,min(flux_mag)-0.15])
			plt.xlim([np.power(min(rbin_normn4),1)-0.15,np.power(max(rbin_normn4),1)+0.15])
			plt.text(rbin_normn4[rbin_normn4.index(max(rbin_normn4))-3],flux_mag[10],' n = 4.\n $\mu_{eff}$ = '+str(round(normprofile[0],5))+'\n $R_{e}$ = '+str(round(rprofilen4[1],5))+'',bbox=dict(facecolor='white'))
			plt.ylabel(r'$\mu$ (mag)')
			plt.xlabel('(R/$R_{e}$)')
			figr.savefig('/home/andrelpkaipper/Documentos/Projetos/photutils/sersic_profile/'+tipo+'/'+cluster+'_linear.png')
			plt.close(figr)
			
			figr4=plt.figure(figsize=(8,8))
			plt.errorbar(np.power(rbin_normn4,0.25),flux_mag,yerr=ebar,fmt='none')
			plt.scatter(np.power(rbin_normn4,0.25),flux_mag,c='gray',edgecolors='black')
			plt.plot(np.power(rbin_normn4,0.25),sersicfitn4(rbin_normn4,*normprofile),'r-')
			plt.ylim([max(flux_mag)+0.15,min(flux_mag)-0.15])
			plt.xlim([np.power(min(rbin_normn4),0.25)-0.15,np.power(max(rbin_normn4),0.25)+0.15])
			plt.text(np.power(rbin_normn4[rbin_normn4.index(max(rbin_normn4))-3],0.25),flux_mag[10],' n = 4.\n $\mu_{eff}$ = '+str(round(normprofile[0],5))+'\n $R_{e}$ = '+str(round(rprofilen4[1],5))+'',bbox=dict(facecolor='white'))
			plt.ylabel(r'$\mu$ (mag)')
			plt.xlabel(r'$(R/R_e)^{1/4}$')
			figr4.savefig('/home/andrelpkaipper/Documentos/Projetos/photutils/sersic_profile/'+tipo+'/'+cluster+'_norm_r4.png')
			plt.close(figr4)
			
raios=open('raios_tipos_clean_n4.dat','w')
min_rn=[[] for i in range(3)]
max_rn=[[] for i in range(3)]

min_rnn4=[[] for i in range(3)]
max_rnn4=[[] for i in range(3)]

kind=['eliptica','assimetrica','envelope']

for i in range(len(typ)):
	if [fn[i],fmag[i],fre[i],fmask[i],frmax[i]] == [0,0,0,0,0]:
		if typ[i] in kind:
			min_rn[kind.index(typ[i])].append(rnmin[i])
			max_rn[kind.index(typ[i])].append(rnmax[i])
			min_rnn4[kind.index(typ[i])].append(rnminn4[i])
			max_rnn4[kind.index(typ[i])].append(rnmaxn4[i])

raios.write('%f \t %f \t %f \t %f \t %f \t %f \n %f \t %f \t %f \t %f \t %f \t %f'%(np.median(min_rnn4[0]),np.median(max_rnn4[0]),np.median(min_rnn4[1]),
np.median(max_rnn4[1]),np.median(min_rnn4[2]),np.median(max_rnn4[2]),np.median(min_rn[0]),
np.median(max_rn[0]),np.median(min_rn[1]),np.median(max_rn[1]),np.median(min_rn[2]),np.median(max_rn[2])))
