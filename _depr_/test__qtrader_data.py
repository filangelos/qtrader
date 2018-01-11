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

# closing price vectors
U = pipe.transform(X)

# visualize price vectors versus time
for j, u in enumerate(U):
    plt.plot(t, u, label=j)

plt.legend()
plt.show()
