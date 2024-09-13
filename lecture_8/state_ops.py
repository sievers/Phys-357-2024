import numpy as np

#let's write down the kets for each
#primary direction in the z-basis
zp=np.asarray([1.,0])
zm=np.asarray([0,1.])

xp=np.asarray([1,1])/np.sqrt(2)
xm=np.asarray([1,-1])/np.sqrt(2)

yp=np.asarray([1,1J])/np.sqrt(2)
ym=np.asarray([1,-1J])/np.sqrt(2)

#let's build up the matrices of bras and kets, which
#will come in handy 
zket=np.zeros([2,2],dtype='complex')
zket[:,0]=zp
zket[:,1]=zm

xket=np.zeros([2,2],dtype='complex')
xket[:,0]=xp
xket[:,1]=xm

zket=np.zeros([2,2],dtype='complex')
zket[:,0]=yp
zket[:,1]=ym

#what should the associated bra matrices look like?

#what does a 90-degree rotation about the z-axis look like?
#that has to take +x to +y, and +y to -x
