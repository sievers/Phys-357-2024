import numpy as np

zp=np.asarray([1,0])
xp=np.asarray([1,1])/np.sqrt(2)
zm=np.asarray([0,1])

ket_in=np.zeros([2,2])
ket_in[:,0]=zp
ket_in[:,1]=xp

ket_out=np.zeros([2,2])
ket_out[:,0]=xp
ket_out[:,1]=zm

#we have ket_out=R@ket_in, so R=ket_out@inv(ket_in)
Ry=ket_out@np.linalg.inv(ket_in)
print(Ry)
xm=Ry@zm
print('my new xm is ',xm)
print('my new zp is ',Ry@xm)

ket_in2=np.zeros([2,2])
ket_in2[:,0]=zm
ket_in2[:,1]=xm
ket_out2=np.zeros([2,2])
ket_out2[:,0]=xm
ket_out2[:,1]=-zp
Ry2=ket_out2@np.linalg.inv(ket_in2)
print("new rotation matrix: ")
print(Ry2)
e,v=np.linalg.eig(Ry2)
print("abs(evals): ",np.abs(e))
