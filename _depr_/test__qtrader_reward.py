# scientific computing
import numpy as np

import qtrader as qt

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 5.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# set random seed
np.random.seed(0)


# cardinality of stocks universe
m = 3
# number of timestamps
N = 500

# time index
t = np.linspace(0, 10, N)

# raw matrix
X = np.tile(t, (m, 1))

# transformation pipeline - toy data
pipe = qt.data.Pipeline([qt.data.Sinusoidal, qt.data.Noise])

# closing prices
U = pipe.transform(X)

# truncate negative prices
U[U < 0] = 0

# price relative matrix
# y_{t} = u_{t} \oslash u_{t-1} = [ 1, u_{1, t} / u_{1, t-1}, ..., u_{m, t} / u_{m, t-1} ]
# t: time index
# m: cardinality of stocks universe
_Y = U[:, 1:] / (U[:, :-1] + 1e-6)
# append ones
Y = np.r_[np.ones((1, N - 1)), _Y]

# visualize price relative vectors versus time
for j, y in enumerate(Y):
    plt.plot(y, label=j)
plt.legend()

# random weights
_W = np.random.randn(Y.shape[1], Y.shape[0])

# normalise weights to sum to 1
W = np.apply_along_axis(qt.utils.softmax, 1, _W)

# logarithmic rate of return
# r_{t} = ln(y_{t}*w_{t-1})
R = np.log(np.dot(W, Y).diagonal())

plt.figure()

plt.plot(R)
plt.title("Reward over Time")

plt.show()
