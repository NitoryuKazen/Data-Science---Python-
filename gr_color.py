from math import sin,cos,tan,pi,floor,log10,sqrt
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
import datetime 
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

	frame = 'http://data.sdss3.org/sas/dr12/boss/photoObj/frames/301/%d/%d/frame-g-%06d-%d-%04d.fits.bz2' % (run,camcol,run,camcol,field)
	print 'Downloading FRAME for '+cluster+'...'
	call('wget -r -nd -q --directory-prefix='+cluster+' '+frame,shell=True)
	call('bunzip2 */frame*.bz2',shell=True)
	return
######################################################################
with open('data_indiv_ecd_nodelta.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_ecd_nodelta.dat','r')
band='g'
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
	if os.path.isfile(cluster+'/bcg_g_mask.fits'):
		pass
	else:
		downframes(str(ll1[0]),int(ll1[2]),int(ll1[3]),int(ll1[4]))
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

		header = pyfits.getheader(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',0)
		data = pyfits.getdata(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',0)
	
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
		xc=int(floor(siz))
		yc=int(floor(siz))
	#INFX	
		if infx<=0:
			xc=int(siz+infx)
			ninfx=1
		elif infx>0:
			ninfx=infx
	#INFY:
		if infy<=0:
			yc=int(siz+infy)
			ninfy=1
		elif infy>0:
			ninfy=infy
	#SUPX

		if supx>data.shape[1]:
			nsupx=data.shape[1]
		elif supx<=data.shape[1]:
			nsupx=supx
	#SUPY
	
		if supy>data.shape[0]:
			nsupy=data.shape[0]
		elif supy<=data.shape[0]:
			nsupy=supy		
##########################################################

		inp2=open('base_default.sex','r')
		out1=open(cluster+'/base_default.sex','w')
	
		for j in range(0,ninp2):
			ls2=inp2.readline()
			ll2=ls2.split()
			if len(ll2)>0 and ll2[0]=='CATALOG_NAME':
				ll2[1]=cluster+'/out_sex_small.cat'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='DETECT_MINAREA':
				ll2[1]='3'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]='64'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='CHECKIMAGE_NAME':
				ll2[1]=cluster+'/check1_small.fits,'+cluster+'/check2_small.fits,'+cluster+'/check3_small.fits'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			else:
				out1.write('%s' % ls2)
		inp2.close()
		out1.close()
		call('sex '+cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits -c '+cluster+'/base_default.sex',shell=True)
	
#################################################################################
# obtencao do numero da bcg do small.cat
		with open(cluster+'/out_sex_small.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_small.cat','r')
		dist=100000.
		yb0=-1.
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
				if ((RA-x0)**2+(DEC-y0)**2)**0.5<dist:
					yb=float(lla[0])
					dist=((RA-x0)**2+(DEC-y0)**2)**0.5
					VRA=RA
					VDEC=DEC
					VX=X
					VY=Y
#######################################################
	# MASCARA COMECA AQUI
######################################
		# mask normal 

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10a=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] == yb:
					data10a[i,j] = 0 
				elif data10a[i,j] == 0:
					data10a[i,j] = 0 
				elif data10a[i,j] != yb and data10a[i,j] != 0:
					data10a[i,j] = 1

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11a=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11a[i,j] == xb:
					data11a[i,j] = 0 
				elif data11a[i,j] == 0:
					data11a[i,j] = 0 
				elif data11a[i,j] != xb and data11a[i,j] != 0:
					data11a[i,j] = 1

		mask_miss=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] != 0 or data11a[i,j] != 0:
					mask_miss[i,j] = 1
				else:
					mask_miss[i,j] = 0
###################################################
	# mask da bcg 

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10b=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] == yb:
					data10b[i,j] = 1 
				elif data10b[i,j] == 0:
					data10b[i,j] = 0 
				elif data10b[i,j] != yb and data10b[i,j] != 0:
					data10b[i,j] = 0

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11b=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11b[i,j] == xb:
					data11b[i,j] = 1 
				elif data11b[i,j] == 0:
					data11b[i,j] = 0 
				elif data11b[i,j] != xb and data11b[i,j] != 0:
					data11b[i,j] = 0

		mask_dirty=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] != 0 or data11b[i,j] != 0:
					mask_dirty[i,j] = 1
				else:
					mask_dirty[i,j] = 0 

		#MASK CLEANER
		
		mask_b=copy.deepcopy(mask_dirty)
		coords_distances=[[],[],[]]
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if mask_dirty[i,j] ==1:
					coords_distances[0].append(i)
					coords_distances[1].append(j)
					coords_distances[2].append(np.sqrt((i-yc)**2+(j-xc)**2))
		contiguos=[[int(yc)],[int(xc)]]
		sorteddists=sorted(coords_distances[2])
		sorteddists.append(sorteddists[-1]+1)
		for ii in range(len(sorteddists)):
			for i in range(len(coords_distances[0])):
				if sorteddists[ii]<=coords_distances[2][i]<sorteddists[ii+1]:
					ypix=coords_distances[0][i]
					xpix=coords_distances[1][i]
					laterals=[]
					for k in range(-1,2):
						for kk in range(-1,2):
							laterals.append((ypix+k,xpix+kk))
					contiguos_array=[]
					for k in range(len(contiguos[0])):
						contiguos_array.append((contiguos[0][k],contiguos[1][k]))
					if len(set(laterals) & set(contiguos_array))>0:
						contiguos[0].append(ypix)
						contiguos[1].append(xpix)
						mask_b[ypix,xpix]=1
					else:
						mask_b[ypix,xpix]=0
		pyfits.writeto(cluster+'/bcg_'+band+'_mask_b.fits',mask_b,header=header,clobber=True)
		
		mask=copy.deepcopy(mask_miss)
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if mask_b[i,j] == 0 and mask_dirty[i,j] == 1:
					mask[i,j] = 1
				else:
					mask[i,j] = mask_miss[i,j]
		pyfits.writeto(cluster+'/bcg_'+band+'_mask.fits',mask,header=header,clobber=True)

		data1 = copy.deepcopy(data[ninfy:nsupy,ninfx:nsupx])
		data2 = (((data1-subt)*EXPTIME)/NMGY)
		pyfits.writeto(cluster+'/bcg_'+band+'.fits',data2,header=header,clobber=True)

