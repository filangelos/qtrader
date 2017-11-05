import numpy as np
import matplotlib.pyplot as plt
import qtrader.data as qd

N = 3

t = np.linspace(0, 10, 500)

X = np.tile(t, (N, 1))

pipe = qd.Pipeline([qd.Sinusoidal, qd.Noise])

data = pipe.transform(X)

for cur in data:
    plt.plot(cur)

plt.show()
