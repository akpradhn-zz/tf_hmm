import numpy as np


def toy_dataset(I=10, N=15):
  models = [0.6, 0.4]

  p0 = [0.8, 0.2]
  p0_ = [0.8, 0.2]

  tp = [[0.8, 0.2], [0.4, 0.6]]
  tp_ = [[0.5, 0.5], [0.5, 0.5]]

  mu1 = [1.5, 1.3]
  mu1_ = [-1.5, 1.1]

  mu2 = [1.5, -1.8]
  mu2_ = [1.8, 1.4]

  s1 = [[1.0, 0.2], [0.2, 1.0]]
  s1_ = [[1.0, -0.2], [-0.2, 1.0]]

  s2 = [[1.0, -0.5], [-0.5, 0.9]]
  s2_ = [[1.2, 0.0], [0.0, 0.8]]

  dataset = []
  model = []
  for i in range(I):
    m = np.random.rand()
    if m < models[0]:
      model.append(0)
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
    else:
      model.append(1)
      r = np.random.rand()
      if r < p0_[0]:
        # state 1
        z = 0
      else:
        # state 2
        z = 1
      tmp = []
      for n in range(N):
        if z == 0:
          tmp.append(np.random.multivariate_normal(mu1_, s1_, size=1))
          r = np.random.rand()
          if r > tp_[0][0]:
            z = 1
        else:  # z == 1
          tmp.append(np.random.multivariate_normal(mu2_, s2_, size=1))
          r = np.random.rand()
          if r > tp_[1][1]:
            z = 0
    dataset.append(tmp)

  return np.squeeze(
    np.transpose(np.array(dataset, dtype=np.float64), axes=[0, 1, 3, 2]),
    -1), np.array(model)
