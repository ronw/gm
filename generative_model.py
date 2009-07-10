import abc

class generative_model(object):
    __metaclass__ = abc.ABCMeta

    def eval(self, obs):
        pass
    def logprob(self, obs):
        pass
    def decode(self, obs):
        pass
    def rvs(self, n=1):
        pass
    def init(self, obs):
        pass
    def train(self, obs, iter=10):
        pass
