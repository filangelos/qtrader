import numpy as np

import gym


class PortfolioVector(gym.Space):
    """OpenAI Gym Spaces Portfolio Vector Data Structure."""

    def __init__(self, num_instruments):
        """Constructs a `PortfolioVector` object.

        Parameters
        ----------
        num_instruments: int
            Cardinality of universe
        """
        self.low = -np.ones(num_instruments, dtype=float) * np.inf
        self.high = np.ones(num_instruments, dtype=float) * np.inf

    def sample(self):
        """Draw random `PortfolioVector` sample."""
        _vec = np.random.uniform(0, 1.0, self.shape[0])
        return _vec / np.sum(_vec)

    def contains(self, x, tolerance=1e-5):
        """Assert if `x` in space."""
        shape_predicate = x.shape == self.shape
        range_predicate = (x >= self.low).all() and (x <= self.high).all()
        budget_constraint = np.abs(x.sum() - 1.0) < tolerance
        return shape_predicate and range_predicate and budget_constraint

    @property
    def shape(self):
        """Shape of `PortfolioVector` object."""
        return self.low.shape

    def __repr__(self):
        return "PortfolioVector" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and \
            np.allclose(self.high, other.high)
