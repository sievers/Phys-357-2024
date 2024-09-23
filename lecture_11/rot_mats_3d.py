import numpy as np
th=0.01
Rz=np.asarray([[np.cos(th),0 -np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])

Ry=np.asarray([[np.cos(th**2),0,np.sin(th**2)],[0,1,0],[-np.sin(th**2),0,np.cos(th**2)]])
Rx=np.asarray([[1,0,0],[0,np.cos(th),np.sin(th)],[0,-np.sin(th),np.cos(th)]])

RzRx=Rz@Rx-Rx@Rz

pred=np.eye(3)-Ry
print('commutator is: ')
print(RzRx)
print('')
print('(I-Ry(theta)^2')
print(pred)
