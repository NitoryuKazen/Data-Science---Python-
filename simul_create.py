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

def cutaster(x):
 tempx=''
 for item in x:
 	if item != '*':
 		tempx+=item
 return tempx

###########################################

def simul_build(cluster,X,Y):

#PROGRAMA PARTE 1 

	par = open('simul_indiv_L07_clean.dat','a')
	
	os.chdir(cluster+'/')

	mask = pyfits.getdata('bcg_r_mask.fits')
	mask2= pyfits.getdata('bcg_r_mask_b.fits')
	#
	data = pyfits.getdata('ajust-bcg-r.fits',1)
	data1 = pyfits.getdata('ajust-bcg-r.fits',2)	
	header = pyfits.getheader('ajust-bcg-r.fits',2)
	header1=pyfits.getheader('ajust-bcg-r.fits',1)
	#
	data_exp = pyfits.getdata('ajust-bcg-r-exp.fits',1)
	data1_exp = pyfits.getdata('ajust-bcg-r-exp.fits',2)	
	header_exp = pyfits.getheader('ajust-bcg-r-exp.fits',2)
	header1_exp=pyfits.getheader('ajust-bcg-r-exp.fits',1)
	#
	data0=copy.deepcopy(data1)
	#	
	data0_exp=copy.deepcopy(data1_exp)
	vec=[]		
	for i in range(0,data.shape[0]):
		for o in range(0,data.shape[1]):
			if mask[i,o] == 0 and mask2[i,o] == 0:
				vec.append(data[i,o])
	rows=data.shape[0]
	cols=data.shape[1]
	print cols,rows, len(vec)

	####

	if len(vec) == 0 or len(vec) > 500000 or cols > len(vec):
		par.write('%s\n'%(ll1[0]))
	else:		
		x=np.random.rand(rows, len(vec)-1).argpartition(cols,axis=1)[:,:cols]+1
		print x.shape
		for i in range(0,data.shape[0]):
			for j in range(0,data.shape[1]):
				data0[i,j]+=vec[x[i,j]]
				data0_exp[i,j]+=vec[x[i,j]]
		print 'aqui'
	
	pyfits.writeto('/mnt/Dados1/andre/WHL/'+ll1[0]+'/bcg-simul-n.fits',data0,header=header1,clobber=True)
	pyfits.writeto('/mnt/Dados1/andre/WHL/'+ll1[0]+'/bcg-simul-exp.fits',data0_exp,header=header1_exp,clobber=True)


	inp0.close()
##############
	par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(ll1[0],chi2,re1,mag1,n1,axis1,pa1,float(rff),float(assim[0]),float(assim[5])))
	par.close()
	call('rm bcg_r.fits',shell=True)
	call('rm sigma-r.fits',shell=True)
	call('rm galfit.*',shell=True)
	call('rm bcg_r_psf_b.fits',shell=True)
	os.chdir('../')
	return
######################################################################

ok=[]
out4 = open('testemask.dat','a')
with open('simul_indiv_L07_clean.dat','r') as inp2:
	for item in inp2.readlines():	
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
	if cluster in ok or os.path.isfile('/'+cluster+'/ajust-bcg-exp.fits') == False:
		pass
	else:
		print datetime.datetime.now(), ik, cluster
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
		print siz, xc , yc, infx, infy, supx, supy
		
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

		print 'centro bcg em x',xc
		print 'centro bcg em y',yc
		print 'ninfx',ninfx
		print 'ninfy',ninfy 
		print 'nsupx',nsupx
		print 'nsupy',nsupy
		print siz+infy 
		
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
			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]=str(siz/10.)
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

########################################################## 
# inicio do sigma.fits:

		header.append(card='GAIN',useblanks=True, bottom=False, end=False)
		header['GAIN'] = gain(camcol,cband,run)
	
		sigma = copy.deepcopy(data[ninfy:nsupy,ninfx:nsupx])
		sig = ((sigma*EXPTIME)/NMGY)*gain(camcol,cband,run)	

		vec = []
		for i in range(0,nsupy-ninfy):
			for o in range(0,nsupx-ninfx):
				if mask[i,o] == 0 and mask_b[i,o] == 0 : 
					vec.append(sig[i,o]) 	
		z = np.sum(np.power(vec-np.average(vec),2))/(len(vec)-1.)
		print z
		print len(vec)
		sig1 = (np.sqrt(np.absolute(sig)+z))/gain(camcol,cband,run)
		pyfits.writeto(cluster+'/sigma-'+band+'.fits',sig1,header=header,clobber=True)

########################################################
# imagem da bcg usada pelo galfit, == ajust_..[1]
		data1 = copy.deepcopy(data[ninfy:nsupy,ninfx:nsupx])
		data2 = (((data1-subt)*EXPTIME)/NMGY)
		pyfits.writeto(cluster+'/bcg_'+band+'.fits',data2,header=header,clobber=True)

##########################################################
# prepare psf file

		call('./read_PSF '+cluster+'/psField-'+str(run).zfill(6)+'-'+str(camcol)+'-'+str(field).zfill(4)+'.fit '+str(cband+1)+' '+str(AX)+' '+str(AY)+' '+cluster+'/bcg_'+band+'_psf.fits',shell=True)
	
		headerpsf = pyfits.getheader(cluster+'/bcg_'+band+'_psf.fits',0)
		datapsf = pyfits.getdata(cluster+'/bcg_'+band+'_psf.fits',0)
		datapsf=np.subtract(datapsf,1000.)
		pyfits.writeto(cluster+'/bcg_'+band+'_psf_b.fits',datapsf,clobber=True)

# zeropoints:

		NMGY = float(header['NMGY'])
		a=22.5 
		b=2.5
		c=0.1569166
		zeropoints = a+(b*log10(1./NMGY))
		mag = mag+zeropoints - 2.5*(log10(EXPTIME)) + 2.5*(log10(c))

#arquivo constr.all:

		ou3=open(cluster+'/constr.all','w')
		ou3.write(' 1 x -5 5 \n 1 y -5 5') 

# finish calculations and print input data to galfit

		ou1=open(cluster+'/feedme.r','w')
		ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00    1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'% ('bcg_'+band+'.fits','ajust-bcg-'+band+'.fits','sigma-'+band+'.fits','bcg_'+band+'_psf_b.fits','bcg_'+band+'_mask.fits',1,nsupx-ninfx,1,nsupy-ninfy,zeropoints,xc,yc,mag,siz/4.5,razax,thetabcg))
		ou1.close()
		ou3.close()

		os.chdir(cluster+'/')
		call('rm check*',shell=True)
		call('rm bcg_r_psf.fits',shell=True)
		call('./galfit feedme.r',shell=True)

		os.chdir('../')
		simul_buil(ll1[0],xc,yc)
###########################
