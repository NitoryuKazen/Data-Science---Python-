from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
from subprocess import call
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

	frame = 'http://data.sdss3.org/sas/dr12/boss/photoObj/frames/301/%d/%d/frame-g-%06d-%d-%04d.fits.bz2' % (run,camcol,run,camcol,field)
	print('Downloading FRAME for '+cluster+'...')
	call('wget -r -nd -q --directory-prefix='+cluster+' '+frame,shell=True)
	call('bunzip2 */frame*.bz2',shell=True)
	return
#########################################################################################################################
ok=[]
inpcheck = open('bandag_checklist.dat','r+')
for item in inpcheck.readlines():	
	ok.append(item.split()[0])
with open('data_indiv_ginfo.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_ginfo.dat','r')
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
	if cluster in ok:
		pass
	else:
		print(cluster)
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
		print(infx,supx,infy,supy)
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

		dx,dy=int(ll1[7])-mxc,int(ll1[8])-myc
		if dx == dy:
			nsiz=dx
			if msupy == data.shape[0]:
				if msupx == data.shape[1]:
					ninfx=minfx-nsiz
					nsupx=msupx
					ninfy=minfy-nsiz
					nsupy=msupy
					xc=int(ll1[7])
					yc=int(ll1[8])
				else:
					ninfx=minfx-nsiz
					nsupx=msupx+nsiz
					ninfy=minfy-nsiz
					nsupy=msupy
					xc=int(ll1[7])
					yc=int(ll1[8])

				
			elif msupx == data.shape[1]:
				ninfx=minfx-nsiz
				nsupx=msupx
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
				
			elif minfx == 1:
				ninfx=minfx
				nsupx=msupx+nsiz
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])

			elif minfy == 1:
				ninfx=minfx-nsiz
				nsupx=msupx+nsiz
				ninfy=minfy
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			

			else:
				ninfx=minfx-nsiz
				nsupx=msupx+nsiz
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])

		elif dx > dy:
			nsiz=dx
			msiz=dy

			if minfy == 1:
				ninfx=minfx-nsiz
				nsupx=msupx+nsiz
				ninfy=abs(msiz)+1
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
				if minfx == 1:
					ninfx=abs(nsiz)+1
					nsupx=msupx+nsiz
					ninfy=abs(msiz)+1
					nsupy=msupy+nsiz
					xc=int(ll1[7])
					yc=int(ll1[8])
				
			elif msupy == data.shape[0]:
				if minfx == 1:
					ninfx=abs(msiz)+1
					nsupx=msupx+nsiz
					ninfy=minfy-nsiz
					nsupy=msupy
					xc=int(ll1[7])
					yc=int(ll1[8])
					
				else:
					ninfx=minfx-nsiz
					nsupx=msupx+nsiz
					ninfy=minfy-nsiz
					nsupy=msupy
					xc=int(ll1[7])
					yc=int(ll1[8])
			elif minfx == 1:

				if nsiz == 0:
					ninfx=minfx
					nsupx=msupx+msiz
					ninfy=minfy-msiz
					nsupy=msupy+msiz
					xc=int(ll1[7])
					yc=int(ll1[8])
				else:
					ninfx=abs(nsiz)+1
					nsupx=msupx+msiz
					ninfy=minfy-msiz
					nsupy=msupy+msiz
					xc=int(ll1[7])
					yc=int(ll1[8])
			
			elif minfy-msiz < 0:

				ninfx=minfx-nsiz
				nsupx=msupx+nsiz
				ninfy=1
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			
		
			else:
				ninfx=minfx-nsiz
				nsupx=msupx+nsiz
				ninfy=minfy-msiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			
		elif dx < dy:
			nsiz=dy
			msiz=dx
			if minfx == 1:
				ninfx=abs(msiz)+1
				nsupx=msupx+nsiz
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			elif msupx == data.shape[1]:
				ninfx=minfx-nsiz
				nsupx=msupx
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			elif msupy == data.shape[0]:
				ninfx=minfx
				nsupx=msupx+nsiz
				ninfy=minfy-nsiz
				nsupy=msupy
				xc=int(ll1[7])
				yc=int(ll1[8])
			elif minfy == 1:
				ninfx=minfx-msiz
				nsupx=msupx+msiz
				ninfy=abs(nsiz)+1
				nsupy=msupy+msiz
				xc=int(ll1[7])
				yc=int(ll1[8])
			else:
				ninfx=minfx
				nsupx=msupx+nsiz
				ninfy=minfy-nsiz
				nsupy=msupy+nsiz
				xc=int(ll1[7])
				yc=int(ll1[8])
		
		print(minfx,msupx,minfy,msupy,mxc,myc)
		#print(ll1[9],ll1[10],ll1[11],ll1[12],ll1[7],ll1[8])
		print(ninfx,nsupx,ninfy,nsupy,xc,yc)
		print(int(ll1[7])-mxc,int(ll1[8])-myc)
		
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
		#####################

		cks = fits.open(cluster+'/check3_small.fits')[0].data
		ckl = fits.open(cluster+'/check3_large.fits')[0].data

		data_check_small = cks[ninfy:nsupy,ninfx:nsupx]
		data_check_large = ckl[ninfy:nsupy,ninfx:nsupx]
		
		pre_mask_miss = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		pre_mask_dirty = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		
		############
		#PRE MASK BCG
		pre_mask_miss[(data_check_small == yb)] = 1
		pre_mask_miss[(data_check_large == xb)] = 1
		#PRE MASK NORMAL
		pre_mask_dirty[(data_check_small != yb) & (data_check_small != 0)] = 1
		pre_mask_dirty[(data_check_large != xb) & (data_check_large != 0)] = 1

		##################################3		
		#MASK CLEANER
		#
		mask_miss = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		mask_dirty = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		#
		bcg_pixeis = np.nonzero(pre_mask_miss)
		coords_distances=[bcg_pixeis[0],bcg_pixeis[1],np.sqrt((bcg_pixeis[0]-yc)**2+(bcg_pixeis[1]-xc)**2)]
		###########################################################
		cont=[[int(yc)],[int(xc)]]
		sorteddists=sorted(coords_distances[2])
		sorteddists.append(sorteddists[-1]+1)
		##############################################################3		
		for ii in range(len(sorteddists)):
			for i in range(len(coords_distances[0])):
				if sorteddists[ii]<=coords_distances[2][i]<sorteddists[ii+1]:
					ypix,xpix=coords_distances[0][i],coords_distances[1][i]
					laterals=[]
					for k in range(-1,2):
						for kk in range(-1,2):
							laterals.append((ypix+k,xpix+kk))
					contiguos_array=[]
					for k in range(len(cont[0])):
						contiguos_array.append((cont[0][k],cont[1][k]))
		############################################################3
					if len(set(laterals) & set(contiguos_array))>0:
						cont[0].append(ypix)
						cont[1].append(xpix)
					
		contiguos=tuple(np.asarray(cont))
		
		mask_miss[contiguos] = 1
		pre_mask_miss[contiguos]=0

		mask_dirty[pre_mask_dirty == 1] = 1
		mask_dirty[pre_mask_miss == 1] = 1
				
		fits.writeto(cluster+'/bcg_'+band+'_mask_b.fits',mask_miss,header=header,overwrite=True)
		fits.writeto(cluster+'/bcg_'+band+'_mask.fits',mask_dirty,header=header,overwrite=True)

		data1 = data[ninfy:nsupy,ninfx:nsupx]
		data2 = (((data1-subt)*EXPTIME)/NMGY)

		fits.writeto(cluster+'/bcg_'+band+'.fits',data2, header=header,overwrite=True)
		
		
		datar=fits.open(cluster+'/ajust-bcg-r.fits')[1].data
		maskr=fits.open(cluster+'/bcg_r_mask.fits')[0].data
		maskr_b=fits.open(cluster+'/bcg_r_mask_b.fits')[0].data
		headerr=fits.open(cluster+'/bcg_r_mask.fits')[0].header
#############################################################################################################
#############################################################################################################
		####################################
		if datar.shape[0] == data2.shape[0]:
			shape0=datar.shape[0]
			if datar.shape[1] == data2.shape[1]:
				shape1=datar.shape[1]
			elif datar.shape[1] > data2.shape[1]:
				shape1=data2.shape[1]
			elif datar.shape[1] < data2.shape[1]:
				shape1=datar.shape[1]
		####################################
		elif datar.shape[1] == data2.shape[1]:
			shape1=datar.shape[1]
			if datar.shape[0] > data2.shape[0]:
				shape0=data2.shape[0]
			elif datar.shape[0] < data2.shape[0]:
				shape0=datar.shape[0]
		####################################
		elif datar.shape[1] > data2.shape[1]:
			shape1=data2.shape[1]
			if datar.shape[0] > data2.shape[0]:
				shape0=data2.shape[0]
			elif datar.shape[0] < data2.shape[0]:
				shape0=datar.shape[0]
		####################################
		elif datar.shape[1] < data2.shape[1]:
			shape1=datar.shape[1]
			if datar.shape[0] > data2.shape[0]:
				shape0=data2.shape[0]
			if datar.shape[0] < data2.shape[0]:
				shape0=datar.shape[0]
		##################################
		blank_miss=np.zeros((shape0,shape1),dtype=np.intc)
		blank_dirty=np.zeros((shape0,shape1),dtype=np.intc)
		#
		datag_mod=data2[0:shape0,0:shape1]
		maskg_mod=mask_dirty[0:shape0,0:shape1]
		maskg_b_mod=mask_miss[0:shape0,0:shape1]
		#
		datar_mod=datar[0:shape0,0:shape1]
		maskr_mod=maskr[0:shape0,0:shape1]
		maskr_b_mod=maskr_b[0:shape0,0:shape1]
		#
		blank_miss[(maskr_b_mod==1) | (maskg_b_mod==1)] = 1
		blank_dirty[(maskr_mod==1) | (maskg_mod==1)] = 1
		##
		fits.writeto(cluster+'/bcg_r.fits',datar_mod,header=headerr,overwrite=True)
		fits.writeto(cluster+'/bcg_gr_mask.fits',blank_dirty,header=header,overwrite=True)
		fits.writeto(cluster+'/bcg_gr_mask_b.fits',blank_miss,header=header,overwrite=True)
		fits.writeto(cluster+'/bcg_g.fits',datag_mod,header=header,overwrite=True)

		inpcheck.write('%s \n'%(cluster))
		call('rm '+cluster+'/check*',shell=True)
###########################
