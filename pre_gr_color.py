from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
import datetime 
from astropy.io import fits
import numpy.ma as ma
from numpy.ma import MaskedArray as mask

#######################
def codband(band):
	if band=='u':
		cob=0
	elif band=='g':
		cob=1
	elif band=='r':
		cob=2
	elif band=='i':
		cob=3
	elif band=='z':
		cob=4
	return cob

###################################
def gain(camcol,cband,run):
	if int(run)<1100:
		gainvec=[[1.62,3.32,4.71,5.165,4.745],[1.595,3.855,4.6,6.565,5.155],[1.59,3.845,4.72,4.86,4.885],[1.6,3.995,4.76,4.885,4.775],[1.47,4.05,4.725,4.64,3.48],[2.17,4.035,4.895,4.76,4.69]]
	else:
		gainvec=[[1.62,3.32,4.71,5.165,4.745],[1.825,3.855,4.6,6.565,5.155],[1.59,3.845,4.72,4.86,4.885],[1.6,3.995,4.76,4.885,4.775],[1.47,4.05,4.725,4.64,3.48],[2.17,4.035,4.895,4.76,4.69]]
	ga=gainvec[int(camcol)-1][cband]

	return ga

#########################################

def downframes(cluster,run,camcol,field):

	frame = 'http://data.sdss3.org/sas/dr12/boss/photoObj/frames/301/%d/%d/frame-r-%06d-%d-%04d.fits.bz2' % (run,camcol,run,camcol,field)
	psf = 'http://data.sdss3.org/sas/dr12/boss/photo/redux/301/%d/objcs/%d/psField-%06d-%d-%04d.fit' % (run,camcol,run,camcol,field)
	call('wget -r -nd -q --directory-prefix='+cluster+' '+frame,shell=True)
	call('wget -r -nd -q --directory-prefix='+cluster+' '+psf,shell=True)
	call('cp -i /home/andre/Documentos/L07/galfit /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/read_PSF /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/base_default.sex /home/andre/Documentos/L07/'+cluster, shell=True)
	call('bunzip2 */frame*.bz2',shell=True)
	return	
#########################################################################################################################
ok=[]
inpcheck = open('data_indiv_ginfo.dat','r+')
for item in inpcheck.readlines():	
	ok.append(item.split()[0])
with open('data_indiv_clean.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_clean.dat','r')
band='r'
cband=codband(band)
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	cluster=ll1[0]
	run=str(ll1[2]).zfill(6)
	camcol=str(ll1[3])
	field=str(ll1[4]).zfill(4)
	x0=float(ll1[5])
	y0=float(ll1[6])
	if cluster in ok:
		pass
	else:		
		with open('base_default.sex','r') as inp2:
			ninp2=len(inp2.readlines())
		inp2=open('base_default.sex','r')
		out1=open(cluster+'/base_default.sex','w')
		for j in range(0,ninp2):
			ls2=inp2.readline()
			ll2=ls2.split()
			if len(ll2)>0 and ll2[0]=='CATALOG_NAME':

				ll2[1]=ll1[0]+'/out_sex_large.cat'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='DETECT_MINAREA':
				ll2[1]='100'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])

			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]='128'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='CHECKIMAGE_NAME':
				ll2[1]=cluster+'/check1_large.fits,'+cluster+'/check2_large.fits,'+cluster+'/check3_large.fits'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			else:
				out1.write('%s' % ls2)
		inp2.close()
		out1.close()
		call('sex '+cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits -c '+cluster+'/base_default.sex',shell=True)

		header = fits.open(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',memmap=True)[0].header
		data = fits.open(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',memmap=True)[0].data
	
		CRPIX1=float(header['CRPIX1'])
		CRPIX2=float(header['CRPIX2'])
		CRVAL1=float(header['CRVAL1'])
		CRVAL2=float(header['CRVAL2'])
		CD1_1=float(header['CD1_1'])
		CD1_2=float(header['CD1_2'])
		CD2_1=float(header['CD2_1'])
		CD2_2=float(header['CD2_2'])
		EXPTIME=float(header['EXPTIME'])
		NMGY=float(header['NMGY'])

#############################################3
# obtencao do numero da bcg e formatacao do tamanho da imagem 

		with open(cluster+'/out_sex_large.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_large.cat','r')
		dist=1000.
		xb=-1.
		for i in range(0,ninfa):
			lsa=infa.readline()
			lla=lsa.split()
			if lla[0]!='#':
				X=float(lla[7])
				Y=float(lla[8])
				XI=(CD1_1*(X-CRPIX1)+CD1_2*(Y-CRPIX2))*pi/180.
				ETA=(CD2_1*(X-CRPIX1)+CD2_2*(Y-CRPIX2))*pi/180.
				p=(XI**2+ETA**2)**0.5
				c=np.arctan(p)
				RA=np.arctan(XI*sin(c)/(p*cos(CRVAL2*pi/180.)*cos(c)-ETA*sin(CRVAL2*pi/180.)*sin(c)))*180./pi+CRVAL1
				DEC=np.arcsin(cos(c)*sin(CRVAL2*pi/180.)+ETA*sin(c)*cos(CRVAL2*pi/180.)/p)*180./pi
				subt=float(lla[12])
				if ((RA-x0)**2+(DEC-y0)**2)**0.5<dist:
					xb=float(lla[0])
					dist=((RA-x0)**2+(DEC-y0)**2)**0.5
					ARA=RA
					ADEC=DEC
					AX=X
					AY=Y
					thetabcg=float(lla[11])
					siz=1.5*float(lla[9])*float(lla[4])
					razax=float(lla[10])/float(lla[9])
					mag=float(lla[2])
					countbcg=float(lla[1])
		infa.close()
		infx=int(floor(AX))-int(floor(siz))
		supx=int(floor(AX))+int(floor(siz))
		infy=int(floor(AY))-int(floor(siz))
		supy=int(floor(AY))+int(floor(siz))
		mxc=int(floor(siz))
		myc=int(floor(siz))
		print(infx,infy,supx,supy)
	#INFX	
		if infx<=0:
			mxc=int(siz+infx)
			minfx=1
		elif infx>0:
			minfx=infx
	#INFY:
		if infy<=0:
			myc=int(siz+infy)
			minfy=1
		elif infy>0:
			minfy=infy
	#SUPX

		if supx>data.shape[1]:
			msupx=data.shape[1]
		elif supx<=data.shape[1]:
			msupx=supx
	#SUPY
	
		if supy>data.shape[0]:
			msupy=data.shape[0]
		elif supy<=data.shape[0]:
			msupy=supy		

		inpcheck.write('%s %i %i %i %i %i %i \n'%(ls1[:-2],mxc,myc,minfx,msupx,minfy,msupy))
		#inpcheck.close()
