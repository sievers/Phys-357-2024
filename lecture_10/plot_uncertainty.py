import numpy as np
from matplotlib import pyplot as plt

p=np.linspace(0,1,1001)
lam=np.linspace(0,1,1001)
esqr=np.outer(p,lam**2)+np.outer(1-p,(0*lam+1)**2)
e=np.outer(p,lam)+np.outer(1-p,(0*lam+1))
var=esqr-e**2
plt.ion()
plt.clf()
plt.imshow(np.sqrt(var),extent=[0,1,0,1])
plt.colorbar()
plt.xlabel(r'$\lambda_{small}/\lambda_{large}$')
plt.ylabel(r'$P_{large}$')
plt.title('Uncertainty in 2-state System')
plt.show()
plt.savefig('uncertainty_2state.png')
