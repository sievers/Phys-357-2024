import numpy as np
#code to work out raising/lowering operators of
#angular momentum for arbitrary spin, and use that
#to work out Jx,Jy and from there work out rotation
#about the z-axis, and the eigenkets of Jx and Jy in
#the Jz basis.  We'll assume hbar=1 throughout for simplicity.

j=4/2  #set our max m, reminder that total angular momentum is j(j+1)hbar^2
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
    
Jm=np.conj(Jp.T) #because they're adjoints of each other
print("raising operator: ")
print(Jp) #double check this looks sensible

#we now produce Jx and Jy given the z-axis raising/lowering operators
Jx=(Jm+Jp)/2
Jy=(Jp-Jm)/2J

#double check commutation relation
comm=Jx@Jy-Jy@Jx
Jz=np.diag(m)
err=np.mean(np.abs(comm-1J*Jz))
print('commutation relation error: ',err)  #should be very small!

ex,vx=np.linalg.eigh(Jx)
#by default, eigh sorts by smallest first, but we're used to
#largest first, so flip the eigenvalues/eigenvectors to our
#usual ordering.  This isn't required, but indexing becomes a
#nightmare if you don't do this *and* you try to compare states
#from different bases.
ex=np.flipud(ex)  
vx=np.fliplr(vx)
print("x eigenvalues: ",ex)  #should match m


ey,vy=np.linalg.eigh(Jy)
ey=np.flipud(ey)
vy=np.fliplr(vy)
print("y eigenvalues: ",ey)

#now generate a 90 degree rotation about the z-axis, so we
#can check that the x eigenstates do indeed turn into the y eigenstates
th=np.pi/2
Rz=np.diag(np.exp(-1J*m*th))

xket=vx
#a challenge of numerical eigenvalues here.  If you have
#A= V Lam V^dag for Hermitian A, you can multiply any column
#of V by a random phase. so we have no idea about sensible
#overall phases for our y eigenkets.  This isn't an issue for
#x because Jx is real (because it's the sum or raising/lowering,
#and we constructed raising/lowering to be real by selecting the
#real, positive square root of sqrt(j(j+1)-m(m+1))
#the y states cannot be real, though, so they will have an
#essentially random phase applied to them that in general will
#not line up with the phase we get if we rotate |x> states into |y>.
#Instead, get the y kets by rotating the x kets, but we'll check
#that agrees with the y eigenvectors up to overall phases
yket=Rz@xket
ybra=yket.conj().T
y_braket=ybra@vy
print('y-basis check.  (Abs should be 1, but phases might be odd): ')
print(np.diag(y_braket))
yerr=np.mean(np.abs(y_braket-np.diag(np.diag(y_braket))))
print('off-diagonal mean (should be zero): ',yerr)

#if we rotate about y by 90 degrees, we should take
#z to x, but with proper phases.  let's check that consistency now as well.

Ry=vy@np.diag(np.exp(-1J*ey*np.pi/2))@(vy.conj().T)
zket=np.eye(len(m))
xket=Ry@zket
xbra=xket.conj().T
print('diag of xbra@vx: ')
print(np.diag(xbra@vx))
#now we get the yket by rotating our xkets:
yket=Rz@xket
ybra=yket.conj().T
print('diag of ybra@vy:')
print(np.diag(ybra@vy))

#now, if everything is consistent, Rx should take z to y 
#but with correct phases with a -90 degree rotation
Rx=vx@np.diag(np.exp(1J*ex*np.pi/2))@(vx.conj().T)
ypred=Rx@zket
print('checked phases are ',np.diag(ypred.conj().T@yket))
#ah well, phases are hard...  The amplitudes all check out, as expected,
#but I haven't (yet) tried forcing internal consistency as we did
#for spin-1/2.  
