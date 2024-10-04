import numpy as np

zp=np.asarray([1,0])
xp=np.asarray([1,1])/np.sqrt(2)
zm=np.asarray([0,1])
xm=np.asarray([1,-1])/np.sqrt(2)

A_in=np.zeros([2,2])
A_in[:,0]=zp
A_in[:,1]=xp
A_out=np.zeros([2,2])
A_out[:,0]=xp
A_out[:,1]=zm

#A_out=Ry@A_in, so Ry=A_out@inv(A_in)
Ry=A_out@np.linalg.inv(A_in)
print("Ry the first way is: ")
print(Ry)

#this is the 90 degree rotation about your own axis
R=np.diag(np.exp(-1J*np.pi/4*np.asarray([1,-1])))

#make the ykets
yket=np.asarray([[1,1],[1J,-1J]])/np.sqrt(2)
ybra=yket.conj().T
#convert z basis to y-basis, rotate about y-axis in y-basis
#and convert back to z-basis
Ry2=yket@R@ybra
print("Ry2 is:")
print(Ry2)

print("difference check (should be nearly zero):")
print(np.sum(np.abs(Ry-Ry2)))
