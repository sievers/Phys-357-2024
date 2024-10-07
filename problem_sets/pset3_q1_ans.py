import numpy as np

#problem 1:  show the angular momentum operators commute as expected
#we'll set hbar=1 here
Jz=np.diag([1,-1])/2
Jx=np.asarray([[0,1],[1,0]])/2
Jy=np.asarray([[0,-1J],[1J,0]])/2

JxJy=Jx@Jy-Jy@Jx
JyJz=Jy@Jz-Jz@Jy
JzJx=Jz@Jx-Jx@Jz

print('problem 1: ')
print('[Jx,Jy]-iJz: ',np.sum(np.abs(JxJy-1J*Jz)))
print('[Jy,Jz]-iJx: ',np.sum(np.abs(JyJz-1J*Jx)))
print('[Jz,Jx]-iJy: ',np.sum(np.abs(JzJx-1J*Jy)))


