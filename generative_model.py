import abc

import numpy as np

class GenerativeModel(object):
    __metaclass__ = abc.ABCMeta

    def lpdf(self, obs, *args, **kwargs):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : array_like, length n
            List of data points.

        Returns
        -------
        logprob : array_like
            Log probabilities of each data point in `obs`.
        """
        logprob,posteriors = self.eval(obs, *args, **kwargs)
        return logprob

    def pdf(self, obs, *args, **kwargs):
        """Compute the probability of `obs` under the model.

        Underflow can be avoided by using lpdf instead.

        Parameters
        ----------
        obs : array_like, length n
            List of data points.

        Returns
        -------
        prob : array_like, shape (n,)
            Probabilities of each data point in `obs`.

        See Also
        --------
        lpdf : Compute the log-probability under the model.
    
        """
        return np.exp(self.lpdf(obs, *args, **kwargs))

    @abc.abstractmethod
    def decode(self, obs):
        pass

    @abc.abstractmethod
    def eval(self, obs):
        pass

    @abc.abstractmethod
    def rvs(self, n=1):
        pass

    @abc.abstractmethod
    def init(self, obs):
        pass

    @abc.abstractmethod
    def train(self, obs, iter=10):
        pass
