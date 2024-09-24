import numpy as np
th=0.01

#we want Rz to be
#  cos(th) -sin(th) 0
#  sin(th) cos(th)  0
#     0       0     1
Rz=np.asarray([[np.cos(th),0 -np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
#And Rx should be
#     1      0         0
#     0     cos(th)  sin(th)
#     0    -sin(th)  cos(th)
Rx=np.asarray([[1,0,0],[0,np.cos(th),np.sin(th)],[0,-np.sin(th),np.cos(th)]])
#calculate the commutator
RzRx=Rz@Rx-Rx@Rz


#and based on our calculations, the result of this should come from 
#the rotation about y by theta squared, which is
#    cos(th^2)  0   sin(th^2)
#        0      1       0
#   -sin(th^2)  0   cos(th^2) 
Ry=np.asarray([[np.cos(th**2),0,np.sin(th**2)],[0,1,0],[-np.sin(th**2),0,np.cos(th**2)]])

#the actual predicted commutator is I-Ry(th^2), so we calculate it here
pred=np.eye(3)-Ry
print('commutator is: ')
print(RzRx)
print('')
print('(I-Ry(theta)^2')
print(pred)


max_val=np.max(np.abs(RzRx))
max_err=np.max(np.abs(RzRx-pred))
print('max term in commutator is ',max_val)
print('max term in error is ',max_err)
print('ratio of max error to max value is ',max_err/max_val,' for theta ',th)

