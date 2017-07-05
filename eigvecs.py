# A collection of mutually unbiased bases. Yes, I realize this is an absolutely terrible
# way of storing such things but I haven't implemented their generation in Balthasar yet.
# They are ordered according to their associated monomials' slope.

import numpy as np
from math import sqrt

eigenvectors_2 = [[[1., 0], [0, 1.]],
                  [[1./sqrt(2), 1j / sqrt(2)], [1./sqrt(2), -1j / sqrt(2)]],
                  [[1./sqrt(2), 1./sqrt(2)], [1./sqrt(2), -1./sqrt(2)]]]

w = np.exp((2 * np.pi * 1j) / 3)
eigenvectors_3 = [[[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
                  [[1/sqrt(3), (w*w)/sqrt(3), (w*w)/sqrt(3)], [1/sqrt(3), 1/sqrt(3), w/sqrt(3)], [1/sqrt(3), w/sqrt(3), 1/sqrt(3)]],
                  [[1/sqrt(3), w/sqrt(3), w/sqrt(3)], [1/sqrt(3), (w*w)/sqrt(3), 1/sqrt(3)], [1/sqrt(3), 1/sqrt(3), (w*w)/sqrt(3)]],
                  [[1/sqrt(3), 1/sqrt(3), 1/sqrt(3)], [1/sqrt(3), w/sqrt(3), (w*w)/sqrt(3)], [1/sqrt(3), (w*w)/sqrt(3), w/sqrt(3)]]]

eigenvectors_4 = [[[1., 0, 0, 0], [0, 0, 0, 1.], [0, 0, 1., 0], [0, 1., 0, 0]], 
                  [[0.+0.5*1j, 0.-0.5*1j, -0.5, -0.5], [0.-0.5*1j, 0.-0.5*1j, -0.5, 0.5], 
                   [0.+0.5*1j, 0.-0.5*1j, 0.5,0.5], [0.+0.5*1j, 0.+0.5*1j, -0.5, 0.5]], 
                  [[0.+0.5*1j, -0.5, 0.-0.5*1j, -0.5], [0.-0.5*1j, -0.5, 0.-0.5*1j,0.5], 
                   [0.-0.5*1j, -0.5, 0.+0.5*1j, -0.5], [0.+0.5*1j, -0.5, 0.+0.5*1j, 0.5]], 
                  [[0.5, 0.+0.5*1j, 0.+0.5*1j, -0.5], [0.5, 0.-0.5*1j, 0.+0.5*1j, 0.5], 
                   [0.5, 0.-0.5*1j, 0.-0.5*1j, -0.5], [-0.5, 0.-0.5*1j, 0.+0.5*1j, -0.5]], 
                  [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]]]
