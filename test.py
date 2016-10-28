from tf_hmm import HiddenMarkovModel
from toy_dataset import toy_dataset
import tensorflow as tf
import time

hmm = HiddenMarkovModel(2, 2, time_steps=64, reports=True)

dataset = toy_dataset(30, 64)

hmm.expectation_maximization(dataset, max_steps=10)

print('### p0 ###')
print(hmm._p0)
print('### tp ###')
print(hmm._tp)
print('### mu ###')
print(hmm._mu)
print('### cov ###')
print(hmm._cov)

