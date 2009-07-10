import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.cluster

ZEROLOGPROB = -1e9

def almost_equal(actual, desired, decimal=7):
    return abs(desired - actual) < 0.5 * 10**(-decimal)

def logsum(A, axis=None):
    """ Computes the sum of A assuming A is in the log domain.

    Returns log(sum(exp(A), axis)) while minimizing the possibility of
    over/underflow.
    """
    Amax = A.max(axis)
    Anorm = Amax
    if axis and A.ndim > 1:
        shape = list(A.shape)
        shape[axis] = 1
        Amax.shape = shape
        shape = np.ones(A.ndim)
        shape[axis] = A.shape[axis]
        Anorm = np.tile(Amax, shape)
    Asum = np.log(np.sum(np.exp(A - Anorm), axis))
    Asum += Amax.reshape(Asum.shape)
    return Asum

def _lmvnpdfdiag(obs, means=0.0, covars=1.0):
    nobs, ndim = obs.shape

    # (x-y).T A (x-y) = x.T A x - 2x.T A y + y.T A y
    lpr = -0.5 * (np.tile((np.sum((means**2) / covars, 1)
                           + np.sum(np.log(covars), 1))[np.newaxis,:], (nobs,1))
                  - 2 * np.dot(obs, (means / covars).T)
                  + np.dot(obs**2, (1.0 / covars).T)
                  + ndim * np.log(2 * np.pi))
    return lpr

def _lmvnpdfspherical(obs, means=0.0, covars=1.0):
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:,np.newaxis]

    return _lmvnpdfdiag(obs, means, np.tile(cv, (1, obs.shape[-1])))

def _lmvnpdftied(obs, means, covars):
    nobs, ndim = obs.shape
    nmix = len(means)

    # (x-y).T A (x-y) = x.T A x - 2x.T A y + y.T A y
    icv = np.linalg.inv(covars)
    lpr = -0.5 * (np.tile(np.sum(obs * np.dot(obs, icv), 1), (nmix, 1)).T
                  - 2 * np.dot(np.dot(obs, icv), means.T)
                  + np.tile(np.sum(means * np.dot(means, icv), 1), (nobs, 1))
                  + ndim * np.log(2 * np.pi) + np.log(np.linalg.det(covars)))
    return lpr

def _lmvnpdffull(obs, means, covars):
    # FIXME: this representation of covars is going to lose for caching
    nobs, ndim = obs.shape
    nmix = len(means)
    lpr = np.empty((nobs,nmix))
    for c, (mu, cv) in enumerate(itertools.izip(means, covars)):
        icv = np.linalg.inv(cv)
        lpr[:,c] = -0.5 * (ndim * np.log(2 * np.pi) + np.log(np.linalg.det(cv)))
        for o, currobs in enumerate(obs):
            dzm = (currobs - mu)
            lpr[o,c] += -0.5 * np.dot(np.dot(dzm, icv), dzm.T)
        #dzm = (obs - mu)
        #lpr[:,c] = -0.5 * (np.dot(np.dot(dzm, np.linalg.inv(cv)), dzm.T)
        #                   + np.log(2 * np.pi) + np.linalg.det(cv)).diagonal()
    return lpr

def lmvnpdf(obs, means, covars, cvtype='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    obs : array_like, shape (O, D)
        List of D-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (C, D)
        List of D-dimensional mean vectors for C Gaussians.  Each row
        corresponds to a single mean vector.
    covars : array_like
        List of C covariance parameters for each Gaussian.  The shape
        depends on `cvtype`:
            (C,)      if 'spherical',
            (D, D)    if 'tied',
            (C, D)    if 'diag',
            (C, D, D) if 'full'
    cvtype : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (O, C)
        Array containing the log probabilities of each data point in
        `obs` under each of the C multivariate Gaussian distributions.
    """
    lmvnpdf_dict = {'spherical': _lmvnpdfspherical,
                    'tied': _lmvnpdftied,
                    'diag': _lmvnpdfdiag,
                    'full': _lmvnpdffull}
    return lmvnpdf_dict[cvtype](obs, means, covars)


def sample_gaussian(mean, covar, cvtype='diag', n=1):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (ndim,)
        Mean of the distribution.
    covars : array_like
        Covariance of the distribution.  The shape depends on `cvtype`:
            scalar  if 'spherical',
            (D)     if 'diag',
            (D, D)  if 'tied', or 'full'
    cvtype : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.
    n : int
        Number of samples to generate.

    Returns
    -------
    obs : array, shape (n, ndim)
        Randomly generated sample
    """
    ndim = len(mean)
    rand = np.random.randn(ndim, n)
    if n == 1:
        rand.shape = (ndim,)

    if cvtype == 'spherical':
        rand *= np.sqrt(covar)
    elif cvtype == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        U,s,V = np.linalg.svd(covar)
        sqrtS = np.diag(np.sqrt(s))
        sqrt_covar = np.dot(U, np.dot(sqrtS, V))
        rand = np.dot(sqrt_covar, rand)

    return (rand.T + mean).T


class GMM(object):
    """ Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Attributes
    ----------
    cvtype : string (read-only)
        String describing the type of covariance parameters used by
        the GMM.  Must be one of 'spherical', 'tied', 'diag', 'full'.
    ndim : int (read-only)
        Dimensionality of the mixture components.
    nmix : int (read-only)
        Number of mixture components.
    weights : array, shape (`nmix`,)
        Mixing weights for each mixture component.
    means : array, shape (`nmix`, `ndim`)
        Mean parameters for each mixture component.
    covars : array
        Covariance parameters for each mixture component.  The shape
        depends on `cvtype`:
            (`nmix`,)                if 'spherical',
            (`ndim`, `ndim`)         if 'tied',
            (`nmix`, `ndim`)         if 'diag',
            (`nmix`, `ndim`, `ndim`) if 'full'

    Methods
    -------
    eval(obs)
        Compute the log likelihood of `obs` under the GMM.
    decode(obs)
        Find most likely mixture components for each point in `obs`.
    rvs(n=1)
        Generate `n` samples from the GMM.
    init(obs)
        Initialize GMM parameters from `obs`.
    train(obs)
        Estimate GMM parameters from `obs` using the EM algorithm.

    Examples
    --------
    >>> gmm = GMM(nmix=2, ndim=1)
    >>> gmm.train(numpy.concatenate((numpy.random.randn(100),
    ...                              10*numpy.random.randn(100))))
    >>> gmm.train(numpy.concatenate((20 * [0], 20 * [10])))
    >>> gmm.weights
    array([0.5, 0.5])
    >>> gmm.means
    array([0., 10.])
    >>> gmm.covars
    array([1., 1.])
    >>> gmm.lpdf([0, 5, 10])
    array([, , ])
    """

    def __init__(self, nmix=1, ndim=1, cvtype='diag'):
        """Create a Gaussian mixture model

        Initializes parameters such that every mixture component has
        zero mean and identity covariance.

        Parameters
        ----------
        ndim : int (read-only)
            Dimensionality of the mixture components.
        nmix : int (read-only)
            Number of mixture components.
        cvtype : string (read-only)
            String describing the type of covariance parameters to
            use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
            Defaults to 'diag'.
        """

        self._nmix = nmix
        self._ndim = ndim
        self._cvtype = cvtype

        self.weights = np.tile(1.0 / nmix, nmix)

        self.means = np.zeros((nmix, ndim))

        if cvtype == 'spherical':
            self.covars = np.ones(nmix)
        elif cvtype == 'tied':
            self.covars = np.eye(ndim)
        elif cvtype == 'diag':
            self.covars = np.ones((nmix, ndim))
        elif cvtype == 'full':
            self.covars = np.tile(np.eye(ndim), (nmix, 1, 1))
        else:
            raise (ValueError,
                   "cvtype must be one of 'spherical', 'tied', 'diag', 'full'")

    # Read-only properties.
    @property
    def cvtype(self):
        """Covariance type of the GMM.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._cvtype

    @property
    def ndim(self):
        """Dimensionality of the mixture components."""
        return self._ndim

    @property
    def nmix(self):
        """Number of mixture components in the GMM."""
        return self._nmix

    @property
    def weights(self):
        """Mixing weights for each mixture component."""
        return np.exp(self._log_weights)

    @weights.setter
    def weights(self, weights):
        if len(weights) != self.nmix:
            raise ValueError, 'weights must have length nmix'
        if not almost_equal(np.sum(weights), 1.0):
            raise ValueError, 'weights must sum to 1.0'
        
        self._log_weights = np.log(np.array(weights).copy())
        self._log_weights[np.isinf(self._log_weights)] = ZEROLOGPROB
                          

    @property
    def means(self):
        """Mean parameters for each mixture component."""
        return self._means

    @means.setter
    def means(self, means):
        means = np.array(means)
        if means.shape != (self.nmix, self.ndim):
            raise ValueError, 'means must have shape (nmix, ndim)'
        self._means = means.copy()

    @property
    def covars(self):
        """Covariance parameters for each mixture component."""
        return self._covars

    @covars.setter
    def covars(self, covars):
        covars = np.array(covars)
        if self._cvtype == 'spherical':
            if len(covars) != self.nmix:
                raise ValueError, "'spherical' covars must have length nmix"
            elif np.any(covars <= 0):
                raise ValueError, "'spherical' covars must be non-negative"
        elif self._cvtype == 'tied':
            if covars.shape != (self.ndim, self.ndim):
                raise ValueError, "'tied' covars must have shape (ndim, ndim)"
            elif (not np.all(almost_equal(covars, covars.T))
                  or np.any(np.linalg.eigvalsh(covars) <= 0)):
                raise (ValueError,
                       "'tied' covars must be symmetric, positive-definite")
        elif self._cvtype == 'diag':
            if covars.shape != (self.nmix, self.ndim):
                raise ValueError, "'diag' covars must have shape (nmix, ndim)"
            elif np.any(covars <= 0):
                raise ValueError, "'diag' covars must be non-negative"
        elif self._cvtype == 'full':
            if covars.shape != (self.nmix, self.ndim, self.ndim):
                raise (ValueError,
                       "'full' covars must have shape (nmix, ndim, ndim)")
            for n,cv in enumerate(covars):
                if (not np.all(almost_equal(cv, cv.T))
                    or np.any(np.linalg.eigvalsh(cv) <= 0)):
                    raise (ValueError,
                           "component %d of 'full' covars must be symmetric,"
                           "positive-definite" % n)
        self._covars = np.array(covars).copy()
    
    def eval(self, obs):
        """Compute the log probability under the GMM.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.

        Returns
        -------
        logprob : array_like, shape (n,)
            Log probabilities of each data point in `obs`
        posteriors: array_like, shape (n, nmix)
            Posterior probabilities of each mixture component for each
            observation
        """
        lpr = (lmvnpdf(obs, self._means, self._covars, self._cvtype)
               + np.tile(self._log_weights, (len(obs), 1)))
        logprob = logsum(lpr, axis=1)
        posteriors = np.exp(lpr
                            - np.tile(logprob[:,np.newaxis], (1, self._nmix)))
        return logprob, posteriors

    def decode(self, obs):
        """Find most likely mixture components for each point in `obs`.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.

        Returns
        -------
        components : array_like, shape (n,)
            Index of the most likelihod mixture components for each observation
        """
        logprob, posteriors = self.eval(obs)
        return posteriors.argmax(axis=1)
        
    def rvs(self, n=1):
        """Generate random samples from the GMM.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        obs : array_like, shape (n, ndim)
            List of samples
        """
        weight_pdf = self.weights
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

    def init(self, obs, iter=10, params='wmc'):
        """Initialize GMM parameters from data using the k-means algorithm

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        iter : int
            Number of k-means iterations to perform.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 'w' for weights,
            'm' for means, and 'c' for covars.  Defaults to 'wmc'.
        """
        if 'm' in params:
            self._means,labels = sp.cluster.vq.kmeans2(obs, self._nmix, iter=5)
        if 'w' in params:
            self.weights = np.tile(1.0 / self._nmix, self._nmix)
        if 'c' in params:
            cv = np.cov(obs.T)
            if self._cvtype == 'spherical':
                self._covars = np.tile(np.diag(cv).mean(), self._nmix)
            elif self._cvtype == 'tied':
                self._covars = cv
            elif self._cvtype == 'diag':
                self._covars = np.tile(np.diag(cv), (self._nmix, 1))
            elif self._cvtype == 'full':
                self._covars = np.tile(cv, (self._nmix, 1, 1))

    def train(self, obs, iter=10, min_covar=1.0, verbose=False, thresh=1e-2,
              params='wmc'):
        """ Estimate GMM parameters with the expecatation-maximization algorithm.

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
            process.  Can contain any combination of 'w' for weights,
            'm' for means, and 'c' for covars.  Defaults to 'wmc'.

        Returns
        -------
        logprob : list
            Log probabilities of each data point in `obs` for each iteration
        """
        covar_mstep_fun = {'spherical': self._covar_mstep_spherical,
                           'diag': self._covar_mstep_diag,
                           #'tied': self._covar_mstep_tied,
                           #'full': self._covar_mstep_full,
                           'tied': lambda *args: self._covar_mstep_slow(*args,
                               cvtype='tied'),
                           'full': lambda *args: self._covar_mstep_slow(*args,
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
                self.weights = w / w.sum()
            if 'm' in params:
                self._means = avg_obs * norm
            if 'c' in params:
                self._covars = covar_mstep_fun(obs, posteriors, avg_obs, norm,
                                               min_covar)

        return logprob

    def _covar_mstep_diag(self, obs, posteriors, avg_obs, norm, min_covar):
        # For column vectors:
        # covars_c = average((obs(t) - means_c) (obs(t) - means_c).T,
        #                    weights_c)
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
        cv = np.empty((self._nmix, self._ndim, self._ndim))
        for c in xrange(self._nmix):
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
        for c in xrange(self._nmix):
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
                covars += cv / self._nmix
        return covars


