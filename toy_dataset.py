import numpy as np


def toy_dataset(I=10, N=15):
  p0 = [0.8, 0.2]

  tp = [[0.75, 0.25], [0.4, 0.6]]

  mu1 = [-1.5, 1.3]
  mu2 = [1.5, -1.8]

  s1 = [[0.8, 0.17], [0.17, 0.8]]
  s2 = [[0.8, -0.15], [-0.15, 0.8]]

  dataset = []
  for i in range(I):
    r = np.random.rand()
    if r < p0[0]:
      # state 1
      z = 0
    else:
      # state 2
      z = 1
    tmp = []
    for n in range(N):
      if z == 0:
        tmp.append(np.random.multivariate_normal(mu1, s1, size=1))
        r = np.random.rand()
        if r > tp[0][0]:
          z = 1
      else:  # z == 1
        tmp.append(np.random.multivariate_normal(mu2, s2, size=1))
        r = np.random.rand()
        if r > tp[1][1]:
          z = 0
    dataset.append(tmp)

  return np.squeeze(
    np.transpose(np.array(dataset, dtype=np.float64), axes=[0, 1, 3, 2]), -1)
