import numpy as np
import schrod

x=np.linspace(-10,10,6001)
V=0.5*x**2

states=schrod.States(x,V)
print('first ten energy levels are: ',states.E[:10])
