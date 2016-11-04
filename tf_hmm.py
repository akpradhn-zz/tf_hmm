import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import time


class HiddenMarkovModel(object):
  def __init__(self, states=3, data_dim=2, time_steps=None, reports=False,
               code_number=None):
    self._states = states
    self._data_dim = data_dim
    self._time_steps = None
    self._reports = reports
    self._code_number = code_number
    #    self._dataset_members = None

    # numpy variables
    self._p0 = np.ones([1, self._states], dtype=np.float64)/self._states
    self._tp = np.ones([self._states, self._states],
                       dtype=np.float64)/self._states
    self._mu = np.random.rand(self._states, self._data_dim)
    self._cov = np.array(
      [np.identity(self._data_dim, dtype=np.float64)]*self._states)

    # graph nodes
    self._alpha = None
    self._betta = None
    self._b_p = None
    self._c = None
    self._emissions = None
    self._posterior = None
    self._gamma = None
    self._xi = None
    self._new_p0 = None
    self._new_tp = None
    self._new_mu = None
    self._new_cov = None

    # the whole graph
    self._graph = tf.Graph()

    if time_steps is not None:
      self._time_steps = time_steps
      self._create_the_computational_graph(reports=self._reports)

  # front end functions #######################################################

  def expectation_maximization(self, dataset, max_steps=1, epsilon=0.1,
                               codes=None):
    self._recreate_the_computational_graph(dataset)
    if codes is not None:
      idx_list = []
      for i in range(len(codes)):
        if codes[i] == self._code_number:
          idx_list.append(i)
      dataset = dataset[idx_list]
    converged = False
    tic = time.time()
    with tf.Session(graph=self._graph) as sess:
      step = 0
      posterior = None
      sess.run(tf.initialize_all_variables())
      while (not converged) and (step < max_steps):
        # for step in range(max_steps):
        # if self._reports:
        # print('epoch : %d'%(step+1))
        feed_dict = {self._dataset_tf: dataset, self._mu_tf: self._mu,
                     self._cov_tf: self._cov, self._p0_tf: self._p0,
                     self._tp_tf: self._tp}
        posterior_old = posterior
        self._p0, self._tp, self._mu, self._cov, posterior = sess.run(
          [self._new_p0, self._new_tp, self._new_mu, self._new_cov,
           self._posterior],
          feed_dict=feed_dict)

        if step > 0:
          d_posterior = np.linalg.norm(posterior-posterior_old, 1)
          if d_posterior < epsilon:
            converged = True
        step += 1
      toc = time.time()
      if self._reports:
        if converged:
          print(
            'the training process has been converged in %d steps in %.1f sec'%(
              step, toc-tic))
        else:
          print('warning : the training process has not been converged. The '
                'maximum number of steps has been reached in %.1f sec.'%(
                  toc-tic))

  # writer = tf.train.SummaryWriter('./logs', sess.graph)
  #    merged = tf.merge_all_summaries()

  def posterior(self, dataset):
    self._recreate_the_computational_graph(dataset)
    with tf.Session(graph=self._graph) as sess:
      sess.run(tf.initialize_all_variables())
      feed_dict = {self._dataset_tf: dataset, self._mu_tf: self._mu,
                   self._cov_tf: self._cov, self._p0_tf: self._p0,
                   self._tp_tf: self._tp}
      return np.squeeze(sess.run(self._posterior, feed_dict=feed_dict))

  def plot(self, dataset=None):
    # matplotlib.use('Agg')
    plt.style.use('fivethirtyeight')
    font = matplotlib.font_manager.FontProperties(weight='bold', size=14)
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    z = matplotlib.mlab.bivariate_normal(x_mesh, y_mesh,
                                         sigmax=self.cov[0, 0, 0],
                                         sigmay=self.cov[0, 1, 1],
                                         mux=self.mu[0, 0],
                                         muy=self.mu[0, 1],
                                         sigmaxy=self.cov[0, 0, 1])
    plt.annotate('state #'+str(1),
                 (self.mu[0, 0]+0.5*np.sqrt(self.cov[0, 0, 0]),
                  self.mu[0, 1]+0.5*np.sqrt(self.cov[0, 1, 1])),
                 fontproperties=font, color='purple')
    for k in range(1, self._states):
      z += matplotlib.mlab.bivariate_normal(x_mesh, y_mesh,
                                            sigmax=self.cov[k, 0, 0],
                                            sigmay=self.cov[k, 1, 1],
                                            mux=self.mu[k, 0],
                                            muy=self.mu[k, 1],
                                            sigmaxy=self.cov[k, 0, 1])
      plt.annotate('state #'+str(k+1),
                   (self.mu[k, 0]+0.5*np.sqrt(self.cov[k, 0, 0]),
                    self.mu[k, 1]+0.5*np.sqrt(self.cov[k, 1, 1])),
                   fontproperties=font, color='purple')
      # plt.contourf(x_mesh, y_mesh, z, alpha=0.5, cmap='Blues')

    plt.contourf(x_mesh, y_mesh, z, alpha=0.6, cmap='Blues')
    plt.savefig('1')
    plt.show()

  def save(self, filename):
    if 'hmm' not in filename:
      filename += '_hmm'
    f = open(filename,'w')
    np.savez(filename, self._p0, self._tp, self._mu, self._cov)
    f.close()

  def load(self, filename):
    #f = open(filename,'r')
    if 'hmm' not in filename:
      filename += '_hmm'
    if '.npz' not in filename:
      filename += '.npz'
    np_file = np.load(filename)
    self._p0 = np_file['arr_0']
    self._tp = np_file['arr_1']
    self._mu = np_file['arr_2']
    self._cov = np_file['arr_3']

  # getters
  @property
  def p0(self):
    return np.squeeze(self._p0)

  @property
  def tp(self):
    return self._tp

  @property
  def mu(self):
    return self._mu

  @property
  def cov(self):
    return self._cov

  # end of front end functions ################################################


  # implementation functions ##################################################
  def _create_the_computational_graph(self, reports=False):
    with self._graph.as_default():
      tic = time.time()
      #    self._N = tf.placeholder(tf.int32)
      self._p0_tf = tf.placeholder(tf.float64, shape=[1, self._states])
      self._tp_tf = tf.placeholder(tf.float64,
                                   shape=[self._states, self._states])
      self._mu_tf = tf.placeholder(tf.float64,
                                   shape=[self._states, self._data_dim])
      self._cov_tf = tf.placeholder(tf.float64,
                                    shape=[self._states, self._data_dim,
                                           self._data_dim])
      self._dataset_tf = tf.placeholder(tf.float64,
                                        shape=[None,
                                               self._time_steps,
                                               self._data_dim])
      self._emissions_eval()
      self._forward()
      self._backward()
      self._expectation()
      self._maximization()
      toc = time.time()
      if reports:
        print('the computational graph has been created in %.1f sec'%(toc-tic))

  def _recreate_the_computational_graph(self, dataset):
    dataset_shape = dataset.shape
    if not dataset_shape[1] == self._time_steps:
      self._time_steps = dataset_shape[1]
      tic = time.time()
      self._graph = tf.Graph()
      self._create_the_computational_graph()
      toc = time.time()
      if self._reports:
        print('the computation graph has been recreated in %.1f sec'%(toc-tic))

  def _emissions_eval(self):
    with tf.variable_scope('emissions_eval'):
      x_expanded = tf.expand_dims(self._dataset_tf, -2)
      x_m_mu = tf.sub(x_expanded, self._mu_tf)
      # calculate S^(-1)
      inv_cov = tf.matrix_inverse(self._cov_tf)

      # calculate S^(-1) (x-mu)
      inv_cov_x_m_mu = tf.reduce_sum(
        tf.mul(tf.expand_dims(x_m_mu, -1), inv_cov),
        reduction_indices=-1)
      # normalization constant
      c = tf.pow((2*np.pi)**self._data_dim*tf.matrix_determinant(self._cov_tf),
                 -0.5)
      # calculate c exp{(x-mu)^T S^(-1)(x-mu)} shape : (I, N, states)
      self._emissions = c*tf.exp(
        -0.5*tf.reduce_sum(tf.mul(x_m_mu, inv_cov_x_m_mu),
                           reduction_indices=[-1]))

  def _forward(self):
    with tf.variable_scope('forward'):
      # alpha shape : (N, I, states)
      # c shape : (N, I)
      alpha_list = []
      c_list = []
      a_tmp = tf.mul(self._emissions[:, 0, :],
                     tf.squeeze(self._p0_tf))
      c_tmp = tf.expand_dims(tf.reduce_sum(a_tmp, reduction_indices=-1), -1)
      alpha_list.append(a_tmp)
      c_list.append(c_tmp)
      # for n = 1..N
      for n in range(1, self._time_steps):
        # calculate alpha[n-1] tp
        alpha_tp = tf.matmul(alpha_list[n-1], self._tp_tf)
        # calculate p(x|z) \sum_z alpha[n-1] tp
        a_tmp = tf.mul(tf.squeeze(self._emissions[:, n, :]), alpha_tp)
        c_tmp = tf.expand_dims(tf.reduce_sum(a_tmp, reduction_indices=-1), -1)
        alpha_list.append(a_tmp/c_tmp)
        c_list.append(c_tmp)
      self._alpha = tf.pack(alpha_list)
      self._c = tf.pack(c_list)
      self._posterior = tf.reduce_sum(tf.log(self._c), reduction_indices=0)

  def _backward(self):
    with tf.variable_scope('backward'):
      betta_list = []
      b_p_list = []
      shape = tf.shape(self._dataset_tf)[0]
      dims = tf.pack([shape, self._states])
      b_tmp_ = tf.fill(dims, 1.0)
      b_tmp = tf.ones_like(b_tmp_, dtype=tf.float64)
      betta_list.append(b_tmp)
      # b_tmp = tf.ones_like()
      # betta shape : (N, I, states)
      # c shape : (N, I)
      # self._betta = tf.Variable(
      #  tf.ones([self._time_steps, self._dataset_members, self._states],
      #          dtype=tf.float64),
      #  dtype=tf.float64,
      #  name='betta_backward')
      # self._b_p = tf.Variable(
      #  tf.zeros([self._time_steps-1, self._dataset_members, self._states],
      #           dtype=tf.float64),
      #  dtype=tf.float64,
      #  name='b_p')
      # set betta[n] := 1
      # It's already set

      # for n = N..2

      for n in range(self._time_steps-2, -1, -1):
        # calculate betta[n+1] p(x|z)
        b_p_tmp = tf.mul(betta_list[0],
                         tf.squeeze(self._emissions[:, n+1, :]))
        # calculate \sum_z  tp betta[n+1] p(x|z)
        b_tmp = tf.matmul(b_p_tmp, self._tp_tf, transpose_b=True)
        betta_list.insert(0, b_tmp/self._c[n+1])
        b_p_list.insert(0, b_p_tmp)
        # self._betta = tf.scatter_update(self._betta,
        #                                tf.Variable(n, dtype=tf.int32),
        #                                b_tmp/self._c[n+1])
        # self._b_p = tf.scatter_update(self._b_p,
        #                              tf.Variable(n, dtype=tf.int32),
        #                              b_p_tmp)
      self._betta = tf.pack(betta_list)
      self._b_p = tf.pack(b_p_list)

  def _expectation(self):
    with tf.variable_scope('expectation'):
      # gamma shape : (N, I, states)
      self._gamma = tf.mul(self._alpha, self._betta, name='gamma')
      # xi shape : (N-1, I, states,states)
      # shape = tf.shape(self._dataset_tf)[0]
      # dims = tf.pack([shape, self._states])
      # xi_tmp_ = tf.fill(dims, 1.0)
      # xi_tmp = tf.ones_like(xi_tmp_, dtype=tf.float64)
      # self._xi = tf.Variable(
      #  tf.zeros([self._time_steps-1, self._dataset_members, self._states,
      #            self._states],
      #           dtype=tf.float64),
      #  dtype=tf.float64,
      #  name='xi')
      # for n = 2..N
      xi_list = []
      for n in range(1, self._time_steps):
        # betta[n] p(x_n|z_n)
        # done, b_p from the backward algorithm

        # alpha[n-1] betta[n] p(x_n|z_n)
        a_b_p = tf.batch_matmul(
          tf.expand_dims(self._alpha[n-1]/self._c[n], -1),
          tf.expand_dims(self._b_p[n-1], -1), adj_y=True)
        xi_tmp = tf.mul(a_b_p, self._tp_tf)
        xi_list.append(xi_tmp)
        # self._xi = tf.scatter_update(self._xi,
        #                             tf.Variable(n-1, dtype=tf.int32), xi_tmp)
      self._xi = tf.pack(xi_list)

  def _maximization(self):
    with tf.variable_scope('maximization'):
      # print('gamma shape : ' + str(self._gamma.get_shape()))
      gamma_mv = tf.reduce_mean(self._gamma, reduction_indices=1,
                                name='gamma_mv')
      xi_mv = tf.reduce_mean(self._xi, reduction_indices=1, name='xi_mv')
      # update the initial state probabilities
      # new_p0 = tf.transpose(tf.expand_dims(gamma_mv[0], -1))
      self._new_p0 = tf.transpose(tf.expand_dims(gamma_mv[0], -1))
      # update the transition matrix
      # first calculate sum_n=2^{N} xi_mean[n-1,k , n,l]
      sum_xi_mean = tf.squeeze(tf.reduce_sum(xi_mv, reduction_indices=0))

      self._new_tp = tf.transpose(
        sum_xi_mean/tf.reduce_sum(sum_xi_mean, reduction_indices=0))

      x_t = tf.transpose(self._dataset_tf, perm=[1, 0, 2], name='x_transpose')
      gamma_x = tf.batch_matmul(tf.expand_dims(self._gamma, -1),
                                tf.expand_dims(x_t, -1), adj_y=True)
      sum_gamma_x = tf.reduce_sum(gamma_x, reduction_indices=[0, 1])
      mu_tmp_t = tf.transpose(sum_gamma_x)/tf.reduce_sum(self._gamma,
                                                         reduction_indices=[0,
                                                                            1])
      self._new_mu = tf.transpose(mu_tmp_t)

      # update the covariances
      # gamma shape : (N, I, states)
      # x shape : (I, N, dim)
      # mu shape : (states, dim)
      x_expanded = tf.expand_dims(self._dataset_tf, -2)
      # calculate (x - mu) tensor : expected shape (I, N, states, dim)
      x_m_mu = tf.sub(x_expanded, self._new_mu)
      # calculate (x - mu)(x - mu)^T : expected shape (I, N, states, dim, dim)
      x_m_mu_2 = tf.batch_matmul(tf.expand_dims(x_m_mu, -1),
                                 tf.expand_dims(x_m_mu, -2))
      gamma_r = tf.transpose(self._gamma, perm=[1, 0, 2])
      gamma_x_m_mu_2 = tf.mul(x_m_mu_2,
                              tf.expand_dims(tf.expand_dims(gamma_r, -1), -1))
      self._new_cov = tf.reduce_sum(gamma_x_m_mu_2,
                                    reduction_indices=[0, 1])/tf.expand_dims(
        tf.expand_dims(
          tf.reduce_sum(gamma_r, reduction_indices=[0, 1]), -1), -1)

# end of implementation functions #############################################
