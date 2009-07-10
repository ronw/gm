import abc
import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.cluster

from gmm import *

#implemented_classes = [_GaussianHMM, _GMMHMM]

#def HMM(emission_type='gaussian', *args, **kwargs):
def HMM(emission_type='gaussian', *args, **kwargs):
    supported_emission_types = dict([(x.emission_type, x)
                                     for x in implemented_classes])
    if emission_type == supported_emission_types.keys():
        return suppoerted_emission_types[emission_type](*args, **kwargs)
    else:
        raise ValueError, 'Unknown emission_type'

class _BaseHMM(object):
    """Hidden Markov Model abstract base class.
    
    See the instance documentation for details specific to a particular object.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, nstates=1, emission_type='gaussian'):
        self._nstates = nstates
        self.emission_type = emission_type

        self.startprob = np.tile(1.0 / nstates, nstates)
        self.transmat = np.tile(1.0 / nstates, (nstates, nstates))

    def eval(self, obs):
        """Compute the log probability under the model and compute posteriors

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            Sequence of ndim-dimensional data points.  Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logprob : array_like, shape (n,)
            Log probabilities of each data point in `obs`
        posteriors: array_like, shape (n, nstates)
            Posterior probabilities of each state for each
            observation
        """
        lpr = self._forward_pass()
        posteriors = np.exp(lpr
                            - np.tile(logprob[:,np.newaxis], (1, self._nstates)))
        return logprob, posteriors

    def logprob(self, obs):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            Sequence of ndim-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n,)
            Log probabilities of each data point in `obs`
        """
        logprob,posteriors = self.eval(obs)
        return logprob

    def decode(self, obs):
        """Find most likely states for each point in `obs`.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.

        Returns
        -------
        components : array_like, shape (n,)
            Index of the most likelihod states for each observation
        """
        logprob, posteriors = self.eval(obs)
        return posteriors.argmax(axis=1)
        
    def rvs(self, n=1):
        """Generate random samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        obs : array_like, shape (n, ndim)
            List of samples
        """
        pass

    def init(self, obs, params='stmc', **kwargs):
        """Initialize model parameters from data using the k-means algorithm

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to 'stmc'.
        **kwargs :
            Keyword arguments to pass through to the k-means function 
            (scipy.cluster.vq.kmeans2)

        See Also
        --------
        scipy.cluster.vq.kmeans2
        """
        if 's' in params:
            self.startprob = np.tile(1.0 / self._nstates, self._nstates)
        if 't' in params:
            shape = (self._nstates, self._nstates)
            self.startprob = np.tile(1.0 / self._nstates,  shape)

    def train(self, obs, iter=10, min_covar=1.0, verbose=False, thresh=1e-2,
              params='stmc'):
        """Estimate model parameters with the Baum-Welch algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        iter : int
            Number of EM iterations to perform.
        min_covar : float
            Floor on the diagonal of the covariance matrix to prevent
            overfitting.  Defaults to 1.0.
        verbose : bool
            Flag to toggle verbose progress reports.  Defaults to False.
        thresh : float
            Convergence threshold.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to 'stmc'.

        Returns
        -------
        logprob : list
            Log probabilities of each data point in `obs` for each iteration
        """

        T = time.time()
        logprob = []
        for i in xrange(iter):
            # Expectation step
            curr_logprob,posteriors = self.eval(obs)
            logprob.append(curr_logprob.sum())

            if verbose:
                currT = time.time()
                print ('Iteration %d: log likelihood = %f (took %f seconds).'
                       % (i, logprob[-1], currT - T))
                T = currT

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
                if verbose:
                    print 'Converged at iteration %d.' % i
                break

            # Maximization step
            self._mstep(posteriors)
        return logprob

    @property
    def nstates(self):
        """Number of states in the model."""
        return self._nstates

    @property
    def startprob(self):
        """Mixing startprob for each state."""
        return np.exp(self._log_startprob)

    @startprob.setter
    def startprob(self, startprob):
        if len(startprob) != self._nstates:
            raise ValueError, 'startprob must have length nstates'
        if not almost_equal(np.sum(startprob), 1.0):
            raise ValueError, 'startprob must sum to 1.0'
        
        self._log_startprob = np.log(np.array(startprob).copy())

    @property
    def transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    @transmat.setter
    def transmat(self, startprob):
        if transmat.shape != (self._nstates, self._nstates):
            raise ValueError, 'transmat must have shape (nstates, nstates)'
        if not almost_equal(np.sum(startprob), 1.0):
            raise ValueError, 'each row of transmat must sum to 1.0'
        
        self._log_transmat = np.log(np.array(transmat).copy())

    def _forward_pass(self, fun=np.add):
        pass

    def _forward_pass_viterbi(self, *args):
        return _forward_pass(*args, fun=np.max)

    def _backward_pass(self):
        pass

    def _prune_lattice_frame(self):
        pass

    def _mstep(self):
        pass


class _GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Attributes
    ----------
    cvtype : string (read-only)
        String describing the type of covariance parameters used by
        the model.  Must be one of 'spherical', 'tied', 'diag', 'full'.
    ndim : int (read-only)
        Dimensionality of the Gaussian components.
    nstates : int (read-only)
        Number of states in the model.
    transmat : array, shape (`nstates`, `nstates`)
        Matrix of transition probabilities between states.
    startprob : array, shape ('nstates`,)
        Initial state occupation distribution.
    means : array, shape (`nstates`, `ndim`)
        Mean parameters for each state.
    covars : array
        Covariance parameters for each state.  The shape depends on
        `cvtype`:
            (`nstates`,)                if 'spherical',
            (`ndim`, `ndim`)            if 'tied',
            (`nstates`, `ndim`)         if 'diag',
            (`nstates`, `ndim`, `ndim`) if 'full'

    Methods
    -------
    eval(obs)
        Compute the log likelihood of `obs` under the HMM.
    decode(obs)
        Find most likely state sequence for each point in `obs` using the
        Viterbi algorithm.
    rvs(n=1)
        Generate `n` samples from the HMM.
    init(obs)
        Initialize HMM parameters from `obs`.
    train(obs)
        Estimate HMM parameters from `obs` using the Baum-Welch algorithm.

    Examples
    --------
    >>> hmm = HMM(nstates=2, ndim=1)

    See Also
    --------
    model : Gaussian mixture model
    """

    emission_type = 'gaussian'

    def __init__(self, nstates=1, ndim=1, cvtype='diag'):
        """Create a hidden Markov model

        Initializes parameters such that every state has
        zero mean and identity covariance.

        Parameters
        ----------
        ndim : int
            Dimensionality of the states.
        nstates : int
            Number of states.
        cvtype : string (read-only)
            String describing the type of covariance parameters to
            use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
            Defaults to 'diag'.
        """

        super(_GaussianHMM, self).__init__(nstates,
                                           emission_type=self.emission_type)
        self._nstates = nstates
        self._ndim = ndim
        self._cvtype = cvtype

        self.startprob = np.tile(1.0 / nstates, nstates)
        self.transmat = np.tile(1.0 / nstates, (nstates, nstates))
        self.means = np.zeros((nstates, ndim))
        self.covars = _distribute_covar_matrix_to_match_cvtype(
            np.eye(ndim), cvtype, nstates)

    # Read-only properties.
    @property
    def cvtype(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._cvtype

    @property
    def ndim(self):
        """Dimensionality of the states."""
        return self._ndim

    @property
    def means(self):
        """Mean parameters for each state."""
        return self._means

    @means.setter
    def means(self, means):
        means = np.array(means)
        if means.shape != (self._nstates, self._ndim):
            raise ValueError, 'means must have shape (nstates, ndim)'
        self._means = means.copy()

    @property
    def covars(self):
        """Covariance parameters for each state."""
        return self._covars

    @covars.setter
    def covars(self, covars):
        covars = np.array(covars)
        _validate_covars(covars, self._cvtype, self._nstatessssss, self._ndim)
        self._covars = np.array(covars).copy()

    def _forward_pass(fun=np.add):
        pass

    def _backward_pass():
        pass

    def _prune_lattice_frame():
        pass
    
    def rvs(self, n=1):
        weight_pdf = self.startprob
        weight_cdf = np.cumsum(weight_pdf)

        obs = np.empty((n, self._ndim))
        for x in xrange(n):
            rand = np.random.rand()
            c = (weight_cdf > rand).argmax()
            if self._cvtype == 'tied':
                cv = self._covars
            else:
                cv = self._covars[c]
            samp = sample_gaussian(self._means[c], cv, self._cvtype)

            obs[x] = sample_gaussian(self._means[c], cv, self._cvtype)[:]
        return obs

    def init(self, obs, params='stmc', **kwargs):
        """Initialize model parameters from data using the k-means algorithm

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to 'stmc'.
        **kwargs :
            Keyword arguments to pass through to the k-means function 
            (scipy.cluster.vq.kmeans2)

        See Also
        --------
        scipy.cluster.vq.kmeans2
        """
        if 'm' in params:
            self._means,tmp = sp.cluster.vq.kmeans2(obs, self._nstates, **kwargs)
        if 's' in params:
            self.startprob = np.tile(1.0 / self._nstates, self._nstates)
        if 't' in params:
            shape = (self._nstates, self._nstates)
            self.startprob = np.tile(1.0 / self._nstates,  shape)
        if 'c' in params:
            cv = np.cov(obs.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars = _distribute_covar_matrix_to_match_cvtype(
                cv, self._cvtype, self._nstates)


    def train(self, obs, iter=10, min_covar=1.0, verbose=False, thresh=1e-2,
              params='stmc'):
        """ Estimate model parameters with the expecatation-maximization
        algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        iter : int
            Number of EM iterations to perform.
        min_covar : float
            Floor on the diagonal of the covariance matrix to prevent
            overfitting.  Defaults to 1.0.
        verbose : bool
            Flag to toggle verbose progress reports.  Defaults to False.
        thresh : float
            Convergence threshold.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to 'stmc'.

        Returns
        -------
        logprob : list
            Log probabilities of each data point in `obs` for each iteration
        """
        covar_mstep_fun = {'spherical': _covar_mstep_spherical,
                           'diag': _covar_mstep_diag,
                           #'tied': self._covar_mstep_tied,
                           #'full': self._covar_mstep_full,
                           'tied': lambda *args: _covar_mstep_slow(*args,
                               cvtype='tied'),
                           'full': lambda *args: _covar_mstep_slow(*args,
                               cvtype='full'),
                           }[self._cvtype]

        T = time.time()
        logprob = []
        for i in xrange(iter):
            # Expectation step
            curr_logprob,posteriors = self.eval(obs)
            logprob.append(curr_logprob.sum())

            if verbose:
                currT = time.time()
                print ('Iteration %d: log likelihood = %f (took %f seconds).'
                       % (i, logprob[-1], currT - T))
                T = currT

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
                if verbose:
                    print 'Converged at iteration %d.' % i
                break

            # Maximization step
            w = posteriors.sum(axis=0)
            avg_obs = np.dot(posteriors.T, obs)
            norm = np.tile(1.0 / w[:,np.newaxis], (1, self._ndim))

            if 'w' in params:
                self.startprob = w / w.sum()
            if 'm' in params:
                self._means = avg_obs * norm
            if 'c' in params:
                self._covars = covar_mstep_fun(obs, posteriors, avg_obs, norm,
                                               min_covar)

        return logprob

    def _covar_mstep_diag(self, obs, posteriors, avg_obs, norm, min_covar):
        # For column vectors:
        # covars_c = average((obs(t) - means_c) (obs(t) - means_c).T,
        #                    startprob_c)
        # (obs(t) - means_c) (obs(t) - means_c).T
        #     = obs(t) obs(t).T - 2 obs(t) means_c.T + means_c means_c.T
        #
        # But everything here is a row vector, so all of the
        # above needs to be transposed.
        avg_obs2 = np.dot(posteriors.T, obs * obs) * norm
        avg_means2 = self._means**2
        avg_obs_means = self._means * avg_obs * norm
        return avg_obs2 - 2 * avg_obs_means + avg_means2 + min_covar

    def _covar_mstep_spherical(self, *args):
        return self._covar_mstep_diag(*args).mean(axis=1)

    def _covar_mstep_full(self, obs, posteriors, avg_obs, norm, min_covar):
        print "THIS IS BROKEN"
        # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
        # Distribution"
        avg_obs2 = np.dot(obs.T, obs)
        #avg_obs2 = np.dot(obs.T, avg_obs)
        cv = np.empty((self._nstates, self._ndim, self._ndim))
        for c in xrange(self._nstates):
            wobs = obs.T * np.tile(posteriors[:,c], (self._ndim, 1))
            avg_obs2 = np.dot(wobs, obs) / posteriors[:,c].sum()
            mu = self._means[c][np.newaxis]
            cv[c] = (avg_obs2 - np.dot(mu, mu.T)
                     + min_covar * np.eye(self._ndim))
        return cv

    def _covar_mstep_tied2(self, *args):
        return self._covar_mstep_full(*args).mean(axis=0)

    def _covar_mstep_tied(self, obs, posteriors, avg_obs, norm, min_covar):
        print "THIS IS BROKEN"
        # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
        # Distribution"
        avg_obs2 = np.dot(obs.T, obs)
        avg_means2 = np.dot(self._means.T, self._means)
        return (avg_obs2 - avg_means2 + min_covar * np.eye(self._ndim))

    def _covar_mstep_slow(self, obs, posteriors, avg_obs, norm, min_covar,
                          cvtype):
        w = posteriors.sum(axis=0)
        covars = np.zeros(self.covars.shape)
        for c in xrange(self._nstates):
            mu = self._means[np.newaxis,c]
            #cv = np.dot(mu.T, mu)
            avg_obs2 = np.zeros((self._ndim, self._ndim))
            for t,o in enumerate(obs):
                avg_obs2 += posteriors[t,c] * np.dot(o[np.newaxis,:].T,
                                                     o[np.newaxis,:])
            cv = (avg_obs2 / w[c]
                  - 2 * np.dot(avg_obs[np.newaxis,c].T / w[c], mu)
                  + np.dot(mu.T, mu)
                  + min_covar * np.eye(self._ndim))

            if cvtype == 'spherical':
                covars[c] = np.diag(cv).mean()
            elif cvtype == 'diag':
                covars[c] = np.diag(cv)
            elif cvtype == 'full':
                covars[c] = cv
            elif cvtype == 'tied':
                covars += cv / self._nstates
        return covars


class _GMMHMM(_BaseHMM):
    emission_type = 'gmm'
