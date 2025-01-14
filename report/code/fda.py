from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold, Minimizer,IThreshold
from zellij.strategies.fractals import ILS, DBA
from zellij.strategies.fractals.sampling import PHS
from zellij.strategies.tools import Hypersphere, DistanceToTheBest, MoveUp, Min
from zellij.utils.converters import FloatMinMax, ArrayDefaultC
from zellij.utils.benchmarks import Rosenbrock, himmelblau

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

dim = 2

fun = himmelblau
folder = "code/fda"
if not Path(folder).exists():
  lf = Loss(
      objective=[Minimizer("obj")],
      only_score=False,
  )(fun)

  values = ArrayVar(converter=ArrayDefaultC())
  for i in range(dim):
      values.append(
          FloatVar(f"float_{i+1}", -4, 4, converter=FloatMinMax())
      )

  sp = Hypersphere(values, save_points=True)

  explor = PHS(sp, inflation=1.75)
  exploi = ILS(sp, inflation=1.75)
  stop1 = Threshold(None, "current_calls", 3)  # set target to None, DBA will automatically asign it.
  stop2 = IThreshold(exploi,"step", 1e-16)  # set target to None, DBA will automatically asign it.
  stop3 = Threshold(lf, "calls",50)

  #ts = MoveUp(sp, 5)

  dba = DBA(
      sp, MoveUp(sp, 5), (explor, stop1), (exploi, stop2), scoring=DistanceToTheBest(lf)
  )
  exp = Experiment(dba, lf, stop3, verbose=True, save=folder)
  exp.run()
  print(f"Best solution: f({lf.best_point})={lf.best_score}")

def objective(x) :
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmel_plot():
    fig,ax = plt.subplots()
    x = np.linspace(-4,4,200)
    X,Y = np.meshgrid(x,x)
    Z = objective([X,Y])

    map = ax.contourf(X,Y,Z,cmap="plasma",levels=1000)
    fig.colorbar(map)

    plt.title("FDA (Hypersphere) on Himmelblau function")

himmel_plot()
X = pd.read_csv("code/fda/outputs/all_evaluations.csv")

lvl1= X[X.level == 1]
x1 = lvl1["float_1"]
x2 = lvl1["float_2"]
plt.scatter(x1,x2,s = 10, label = "level 1")

lvl2= X[X.level == 2]
x1 = lvl2["float_1"]
x2 = lvl2["float_2"]
plt.scatter(x1,x2,s = 10, label = "level 2")

plt.legend()
plt.savefig("assets/img/chap_2/plots/fda.png")