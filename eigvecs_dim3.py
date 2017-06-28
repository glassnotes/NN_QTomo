import numpy as np

from math import sqrt

w = np.exp((2 * np.pi * 1j) / 3)

eigenvectors = [[[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
                [[1/sqrt(3), (w*w)/sqrt(3), (w*w)/sqrt(3)], [1/sqrt(3), 1/sqrt(3), w/sqrt(3)], [1/sqrt(3), w/sqrt(3), 1/sqrt(3)]],
                [[1/sqrt(3), w/sqrt(3), w/sqrt(3)], [1/sqrt(3), (w*w)/sqrt(3), 1/sqrt(3)], [1/sqrt(3), 1/sqrt(3), (w*w)/sqrt(3)]],
                [[1/sqrt(3), 1/sqrt(3), 1/sqrt(3)], [1/sqrt(3), w/sqrt(3), (w*w)/sqrt(3)], [1/sqrt(3), (w*w)/sqrt(3), w/sqrt(3)]]]
