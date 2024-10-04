import numpy as np
#code to work out raising/lowering operators of
#angular momentum for arbitrary spin, and use that
#to work out Jx,Jy and from there work out rotation
#about the z-axis, and the eigenkets of Jx and Jy in
#the Jz basis.  We'll assume hbar=1 throughout for simplicity.

j=6/2  #set our max m, reminder that total angular momentum is j(j+1)hbar^2
m=np.linspace(j,-j,int(2*j)+1) #this will give us our allowed m's
print(m)  #double check we got that right, in particular, spacing should be 1

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

#let's do a stern-gerlach!
xbra=vx.conj().T
#Psi=np.asarray([1,0,0])
Psi=0*m
#Psi[0]=1
Psi[int(j)]=1
Psi_new=xbra@Psi
print("Stern-gerlach probabilities: ",np.abs(Psi_new)**2)

Rz=np.diag(np.exp(-1J*m*np.pi/2))
ypred=Rz@vx
ybra=vy.conj().T
test=ybra@ypred
