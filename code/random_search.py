def himmelblau(x) :
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


import matplotlib.pyplot as plt
import numpy as np

fig,ax = plt.subplots()
x = np.linspace(-4,4,200)
X,Y = np.meshgrid(x,x)
Z = himmelblau([X,Y])

map = ax.contourf(X,Y,Z,cmap="plasma",levels=1000)
fig.colorbar(map)

plt.title("Random Search on Himmelblau 2D function")

n = 25

x1 = np.random.uniform(-4,4,n)
x2 = np.random.uniform(-4,4,n)
plt.scatter(x1,x2,s=10)

plt.savefig("assets/img/chap_2/plots/random_search.png")




