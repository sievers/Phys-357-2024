#code to work out raising/lowering operators of
#angular momentum for arbitrary spin, and use that
#to work out Jx,Jy and from there work out rotation
#about the z-axis, and the eigenkets of Jx and Jy in
#the Jz basis.  We'll assume hbar=1 throughout for simplicity.

import numpy as np
from matplotlib import pyplot as plt
plt.ion()
np.set_printoptions(precision=4) #settings to make arrays easier to read
np.set_printoptions(suppress=True)

j=4/2  #set our max m, reminder that total angular momentum is j(j+1)hbar^2
m=np.linspace(j,-j,int(2*j)+1) #this will give us our allowed m's
print(m)  #double check we got that right, in particular, spacing should be 1


#we'll write a routine to generate a rotation matrix by an angle theta
#about an axis set by an angular momentum operator
def R_from_J(J,th):
    #rotation about an axis is exp(-i th J/hbar)
    e,v=np.linalg.eigh(J)
    e_new=-1j*th*e
    return v@np.diag(np.exp(e_new))@v.conj().T


#phases can be rather random out of e.g.
#eigenvector routines, but we're usually
#used to seeing the first element be real
#and positive.  This routine will do that
#note - only use this on matrices where the
#phase is not defined, otherwise you're
#changing physics!
def de_angle(mat):
    mm=mat.copy()
    #this will loop over every column
    for i in range(mat.shape[1]):
        th=np.angle(mat[0,i]) #return theta from a=c exp(i theta)
        mm[:,i]=mm[:,i]/np.exp(1J*th)
    return mm



#now generate our raising operator
#we do need to be a bit careful here with indexing and
#how the m's are ordered.  If the m's go from high to low
#like we are used to writing, then our raising operator takes
#the i^th element of our state and puts it in the (i-1)^th.
#It does nothing to i=0 (where m=j), so we start our loop
#at i=1, or m=j-1
n=len(m)
Jp=np.zeros([n,n])
for i in range(1,len(m)):
    #Reminder, this comes from J- J+ = J^2-Jz^2-hJz
    #and we know that J- is the adjoint of J+.  Putting
    #this on a state will give us j(j+1) from the J^2, and
    #m^2+m from Jz^2+hJz 
    val=np.sqrt(j*(j+1)-m[i]*(m[i]+1))
    Jp[i-1,i]=val  #we want the i'th element to end up in the (i-1)'th spot
Jm=Jp.conj().T

Jx=(Jp+Jm)/2
ex,vx=np.linalg.eigh(Jx)
ex=np.flipud(ex) #syntactic sugar to make eigenvalue ordering the same
vx=np.fliplr(vx)

Jy=(Jp-Jm)/2J
ey,vy=np.linalg.eigh(Jy)
print('y eigenvalues: ',ey)
ey=np.flipud(ey)
vy=np.fliplr(vy)

Jz=np.diag(m)
Jyp=Jz+1J*Jx

#we can raise m for our Jy states with the y raising operator Jyp
vy_raised=Jyp@vy

#let's look at Jyp in the y-basis.  In z, we know that the raising operator
#for y must do something shift x-basis to y-basis, raise, then take y-basis
#back to x-basis

ybra=np.conj(vy).T  #this will rotate from z-basis to y-basis since
                    #we represented these basis vectors in the z-basis
yket=vy #just make this clear
#if I want to calculate Jyp in the z-basis, I convert a state from
#z to y with ybra, raise in the y basis, which is just our same-basis
#raising operator, then convert back to z-basis

Jyp_pred=yket@Jp@ybra

xbra=np.conj(vx).T
xket=vx

Jxp=Jy+1J*Jz
Jxp_pred=xket@Jp@xbra

for i in range(len(m)):
    psi_raised=Jxp@vx[:,i]
    jx_raised=Jx@psi_raised
    top=psi_raised.conj().T@jx_raised
    bot=psi_raised.conj().T@psi_raised
    print('eigenvalue for raised state ',i,' is ',top/bot)

Ry=R_from_J(Jy,np.pi/2) #rotate by 90 degrees about the y-axis

zket=np.eye(len(m))
xket_pred=Ry@zket

to_check=xbra@xket_pred
to_check2=Ry@Ry@zket

plt.clf();plt.imshow(np.abs(to_check2));plt.show()
Rz=R_from_J(Jz,np.pi/2)
to_check=Rz@xket
plt.clf();plt.imshow(np.abs(ybra@to_check));plt.show()

#now we'll rotate the z-basis into an arbitrary direction
theta=np.pi/2
phi=np.pi/2
#one way to do this is rotate the zkets by angle theta about the y-axis
#because then we're pointing in the phi=0 direction but an angle theta from
#vertical
Ry_th=R_from_J(Jy,theta)
kets_tmp=Ry_th@zket #this gives us a set of state along z-x plane but phi=0

Rz_phi=R_from_J(Jz,phi)
kets_rot=Rz_phi@kets_tmp

y_rep=ybra@kets_rot
plt.clf()
plt.imshow(np.abs(y_rep))
plt.show()
