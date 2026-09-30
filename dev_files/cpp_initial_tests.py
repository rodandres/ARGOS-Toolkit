import numpy as np

from argos._cpp import rk45_test
from argos.solvers.RK45 import rk45


def dynamics(t, y):
    return y


y0 = np.array([1.0])

cpp = rk45_test(
    y0,
    0.0,
    1.0,
    0.01,
)


py = rk45(
    dynamics,
    (0.0, 1.0),
    y0,
    h0=0.01,
    h_adaptative=False,
)


print("C++:", cpp)
print("Python:", py.y[:, -1])
print("Exact:", np.exp(1.0))
print("Error:", np.abs(cpp - py.y[:, -1]))