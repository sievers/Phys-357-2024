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

yket=np.zeros([2,2],dtype='complex')
yket[:,0]=yp
yket[:,1]=ym


#let's work out z->x basis
#+x projection operator
xplus_proj=np.outer(np.asarray([1,0]),np.conj(xp.T))
xminus_proj=np.outer(np.asarray([0,1]),np.conj(xm.T))
z2x=xplus_proj+xminus_proj


print('plus x in x is ',z2x@xp)
print('minus x in x is ',z2x@xm)

print('plus z in x is ',z2x@zp)
y_in_x=z2x@yp

#if I write a complex numer as a+ib
#I can also write it as c*exp(i theta)
#np.angle(a+ib) returns theta
theta=np.angle(y_in_x[0])
y_in_x=y_in_x*np.exp(-1J*theta)
print('plus y in x is ',y_in_x)

#we want to take <y| y>
print('normalization: ',np.dot(np.conj(y_in_x),y_in_x))


print("\n Doing rotation about z now")
M1=np.zeros([2,2],dtype='complex')
M1[:,0]=xp
M1[:,1]=yp

M2=np.zeros([2,2],dtype='complex')
M2[:,0]=yp
M2[:,1]=xm

R=M2@np.linalg.inv(M1)
print('rotation matrix is: ',R)

print('R@ym is ',R@ym)
print('R@xm is ',R@xm)
assert(1==0)
                     

#what should the associated bra matrices look like?

#what does a 90-degree rotation about the z-axis look like?
#that has to take +x to +y, and +y to -x
