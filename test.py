from tf_hmm import HiddenMarkovModel
from toy_dataset import toy_dataset
import tensorflow as tf
import time

hmm = HiddenMarkovModel(2, 2, time_steps=64, reports=True, code_number=1)

# training using the 1st, 2nd, 3rd, 4th, 8th and 9th members of the dataset
codes = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
dataset = toy_dataset(10, 64)

print(hmm.posterior(dataset))
hmm.expectation_maximization(dataset, max_steps=100, codes=codes)
hmm.plot()

print('### p0 ###')
print(hmm.p0)
print('### tp ###')
print(hmm.tp)
print('### mu ###')
print(hmm.mu)
print('### cov ###')
print(hmm.cov)
print()

hmm.save('example')

print(hmm.posterior(dataset))
