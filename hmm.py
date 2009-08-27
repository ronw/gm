import abc
import itertools
import logging
import time

import numpy as np
import scipy as sp
import scipy.cluster

from generative_model import GenerativeModel
from gmm import *
from gmm import _distribute_covar_matrix_to_match_cvtype, _validate_covars

ZEROLOGPROB = -1e200

log = logging.getLogger('gm.hmm')

def HMM(emission_type='gaussian', *args, **kwargs):
    """Create an HMM object with the given emission_type."""
    supported_emission_types = dict([(x.emission_type, x)
                                     for x in _BaseHMM.__subclasses__()])
    if emission_type in supported_emission_types.keys():
        return supported_emission_types[emission_type](*args, **kwargs)
    else:
        raise ValueError, 'Unknown emission_type'

class _BaseHMM(GenerativeModel):
    """Hidden Markov Model abstract base class.
    
    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Attributes
    ----------
    nstates : int (read-only)
        Number of states in the model.
    transmat : array, shape (`nstates`, `nstates`)
        Matrix of transition probabilities between states.
    startprob : array, shape ('nstates`,)
        Initial state occupation distribution.
    labels : list, len `nstates`
        Optional labels for each state.

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

    See Also
    --------
    gmm : Gaussian mixture model
    """
    __metaclass__ = abc.ABCMeta

    # This class implements the public interface to all HMMs that
    # derive from it, including all of the machinery for the
    # forward-backward and Viterbi algorithms.  Subclasses need only
    # implement the abstractproperty emission_type, and the
    # abstractmethods _generate_sample_from_state(),
    # _compute_obs_log_likelihood(), _init(), _init_sufficient_statistics(),
    # _accumulate_sufficient_statistics(), and _mstep() which depend
    # on the specific emission distribution.
    #
    # Subclasses will probably also want to implement properties for
    # the emission distribution parameters to expose them publically.

    @abc.abstractproperty
    def emission_type(self):
        """String identifier for the emission distribution used by this HMM"""
        pass

    def __init__(self, nstates=1):
        self._nstates = nstates
        self.startprob = np.tile(1.0 / nstates, nstates)
        self.transmat = np.tile(1.0 / nstates, (nstates, nstates))
        self.labels = [None] * nstates

    def eval(self, obs, maxrank=None, beamlogprob=-np.Inf):
        """Compute the log probability under the model and compute posteriors

        Implements rank and beam pruning in the forward-backward
        algorithm to speed up inference in large models.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            Sequence of ndim-dimensional data points.  Each row
            corresponds to a single point in the sequence.
        maxrank : int
            Maximum rank to evaluate for rank pruning.  If not None,
            only consider the top `maxrank` states in the inner
            sum of the forward algorithm recursion.  Defaults to None
            (no rank pruning).  See The HTK Book for more details.
        beamlogprob : float
            Width of the beam-pruning beam in log-probability units.
            Defaults to -numpy.Inf (no beam pruning).  See The HTK
            Book for more details.

        Returns
        -------
        logprob : array_like, shape (n,)
            Log probabilities of the sequence `obs`
        posteriors: array_like, shape (n, nstates)
            Posterior probabilities of each state for each
            observation

        See Also
        --------
        lpdf : Compute the log probability under the model
        decode : Find most likely state sequence corresponding to a `obs`
        """
        framelogprob = self._compute_obs_log_likelihood(obs)
        logprob, fwdlattice = self._do_forward_pass(framelogprob, maxrank,
                                                    beamlogprob)
        bwdlattice = self._do_backward_pass(framelogprob, fwdlattice, maxrank,
                                            beamlogprob)
        gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        #posteriors = np.exp(gamma - logprob)
        posteriors = np.exp(gamma.T - logsum(gamma, axis=1)).T
        return logprob, posteriors

    def lpdf(self, obs, maxrank=None, beamlogprob=-np.Inf):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            Sequence of ndim-dimensional data points.  Each row
            corresponds to a single data point.
        maxrank : int
            Maximum rank to evaluate for rank pruning.  If not None,
            only consider the top `maxrank` states in the inner
            sum of the forward algorithm recursion.  Defaults to None
            (no rank pruning).  See The HTK Book for more details.
        beamlogprob : float
            Width of the beam-pruning beam in log-probability units.
            Defaults to -numpy.Inf (no beam pruning).  See The HTK
            Book for more details.

        Returns
        -------
        logprob : array_like, shape (n,)
            Log probabilities of each data point in `obs`

        See Also
        --------
        eval : Compute the log probability under the model and compute posteriors
        decode : Find most likely state sequence corresponding to a `obs`
        """
        framelogprob = self._compute_obs_log_likelihood(obs)
        logprob, fwdlattice =  self._do_forward_pass(framelogprob, maxrank,
                                                     beamlogprob)
        return logprob

    def decode(self, obs, maxrank=None, beamlogprob=-np.Inf):
        """Find most likely state sequence corresponding to `obs`.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        maxrank : int
            Maximum rank to evaluate for rank pruning.  If not None,
            only consider the top `maxrank` states in the inner
            sum of the forward algorithm recursion.  Defaults to None
            (no rank pruning).  See The HTK Book for more details.
        beamlogprob : float
            Width of the beam-pruning beam in log-probability units.
            Defaults to -numpy.Inf (no beam pruning).  See The HTK
            Book for more details.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM
        components : array_like, shape (n,)
            Index of the most likelihood states for each observation

        See Also
        --------
        eval : Compute the log probability under the model and compute posteriors
        lpdf : Compute the log probability under the model
        """
        framelogprob = self._compute_obs_log_likelihood(obs)
        logprob, state_sequence = self._do_viterbi_pass(framelogprob, maxrank,
                                                        beamlogprob)
        return logprob, state_sequence
        
    def rvs(self, n=1):
        """Generate random samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        obs : array_like, length `n`
            List of samples
        """

        startprob_pdf = self.startprob
        startprob_cdf = np.cumsum(startprob_pdf)
        transmat_pdf = self.transmat
        transmat_cdf = np.cumsum(transmat_pdf, 1);

        # Initial state.
        rand = np.random.rand()
        currstate = (startprob_cdf > rand).argmax()
        obs = [self._generate_sample_from_state(currstate)]

        for x in xrange(n-1):
            rand = np.random.rand()
            currstate = (transmat_cdf[currstate] > rand).argmax()
            obs.append(self._generate_sample_from_state(currstate))

        return np.array(obs)

    def init(self, obs, params='stmc', **kwargs):
        """Initialize model parameters from data using the k-means algorithm

        Parameters
        ----------
        obs : array_like, shape (nobs, n, ndim)
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
        self._init(obs, params, **kwargs)

    def train(self, obs, iter=10, thresh=1e-2, params='stmc', maxrank=None,
              beamlogprob=-np.Inf, **kwargs):
        """Estimate model parameters with the Baum-Welch algorithm.

        Parameters
        ----------
        obs : array_like, shape (nobs, n, ndim)
            List of ndim-dimensional data points.  Each row corresponds to a
            single data point.
        iter : int
            Number of iterations to perform.
        thresh : float
            Convergence threshold.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to 'stmc'.
        maxrank : int
            Maximum rank to evaluate for rank pruning.  If not None,
            only consider the top `maxrank` states in the inner
            sum of the forward algorithm recursion.  Defaults to None
            (no rank pruning).  See The HTK Book for more details.
        beamlogprob : float
            Width of the beam-pruning beam in log-probability units.
            Defaults to -numpy.Inf (no beam pruning).  See The HTK
            Book for more details.

        Returns
        -------
        logprob : list
            Log probabilities of each data point in `obs` for each iteration
        """

        T = time.time()
        logprob = []
        for i in xrange(iter):
            # Expectation step
            stats = self._init_sufficient_statistics()
            curr_logprob = 0
            for seq in obs:
                framelogprob = self._compute_obs_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob, maxrank,
                                                        beamlogprob)
                bwdlattice = self._do_backward_pass(framelogprob, fwdlattice,
                                                    maxrank, beamlogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsum(gamma, axis=1)).T

                curr_logprob += lpr
                self._accumulate_sufficient_statistics(stats, seq, framelogprob,
                                                       posteriors, fwdlattice,
                                                       bwdlattice)
            logprob.append(curr_logprob)

            currT = time.time()
            log.info('Iteration %d: log likelihood = %f (took %f seconds).'
                      % (i, logprob[-1], currT - T))
            T = currT

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
                log.info('Converged at iteration %d.' % i)
                break

            # Maximization step
            self._mstep(stats, params, **kwargs)
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
        
        self._log_startprob = np.log(np.asarray(startprob).copy())

    @property
    def transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    @transmat.setter
    def transmat(self, transmat):
        if np.asarray(transmat).shape != (self._nstates, self._nstates):
            raise ValueError, 'transmat must have shape (nstates, nstates)'
        if not np.all(almost_equal(np.sum(transmat, axis=1), 1.0)):
            raise ValueError, 'each row of transmat must sum to 1.0'
        
        self._log_transmat = np.log(np.asarray(transmat).copy())

    def _do_viterbi_pass(self, framelogprob, maxrank=None, beamlogprob=-np.Inf):
        nobs = len(framelogprob)
        lattice = np.zeros((nobs, self._nstates))
        traceback = np.zeros((nobs, self._nstates), dtype=np.int) 

        lattice[0] = self._log_startprob + framelogprob[0]
        for n in xrange(1, nobs):
            idx = self._prune_states(lattice[n-1], maxrank, beamlogprob)
            pr = self._log_transmat[idx].T + lattice[n-1,idx]
            lattice[n]   = np.max(pr, axis=1) + framelogprob[n]
            traceback[n] = np.argmax(pr, axis=1)
        lattice[lattice <= ZEROLOGPROB] = -np.Inf;
        
        # Do traceback.
        reverse_state_sequence = []
        s = lattice[-1].argmax()
        for frame in reversed(traceback):
            reverse_state_sequence.append(s)
            s = frame[s]

        reverse_state_sequence.reverse()
        return logsum(lattice[-1]), np.array(reverse_state_sequence)

    def _do_forward_pass(self, framelogprob, maxrank=None, beamlogprob=-np.Inf):
        nobs = len(framelogprob)
        fwdlattice = np.zeros((nobs, self._nstates))

        fwdlattice[0] = self._log_startprob + framelogprob[0]
        for n in xrange(1, nobs):
            idx = self._prune_states(fwdlattice[n-1], maxrank, beamlogprob)
            fwdlattice[n] = (logsum(self._log_transmat[idx].T
                                    + fwdlattice[n-1,idx], axis=1)
                             + framelogprob[n])
        fwdlattice[fwdlattice <= ZEROLOGPROB] = -np.Inf;

        return logsum(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob, fwdlattice, maxrank=None,
                          beamlogprob=-np.Inf):
        nobs = len(framelogprob)
        bwdlattice = np.zeros((nobs, self._nstates))
        for n in xrange(nobs - 1, 0, -1):
            # Do HTK style pruning (p. 137 of HTK Book version 3.4).
            # Don't bother computing backward probability if
            # fwdlattice * bwdlattice is more than a certain distance
            # from the total log likelihood.
            idx = self._prune_states(bwdlattice[n] + fwdlattice[n], None,
                                     -50)
                                     #beamlogprob)
                                     #-np.Inf)
            bwdlattice[n-1] = logsum(self._log_transmat[idx].T
                                     + bwdlattice[n,idx] + framelogprob[n,idx],
                                     axis=1)
        bwdlattice[bwdlattice <= ZEROLOGPROB] = -np.Inf;

        return bwdlattice

    def _prune_states(self, lattice_frame, maxrank, beamlogprob):
        """ Returns indices of the active states in `lattice_frame`
        after rank and beam pruning.
        """
        # Beam pruning
        threshlogprob = logsum(lattice_frame) + beamlogprob
        
        # Rank pruning
        if maxrank:
            # How big should our rank pruning histogram be?
            nbins = 3 * len(lattice_frame)

            lattice_min = lattice_frame[lattice_frame > ZEROLOGPROB].min() - 1
            hst, cdf = np.histogram(lattice_frame, bins=nbins, new=True,
                                    range=(lattice_min, lattice_frame.max()))
        
            # Want to look at the high ranks.
            hst = hst[::-1].cumsum()
            cdf = cdf[::-1]

            rankthresh = cdf[hst >= min(maxrank, self._nstates)].max()
      
            # Only change the threshold if it is stricter than the beam
            # threshold.
            threshlogprob = max(threshlogprob, rankthresh)
    
        # Which states are active?
        state_idx, = np.nonzero(lattice_frame >= threshlogprob)
        return state_idx

    @abc.abstractmethod
    def _compute_obs_log_likelihood(self, obs):
        pass
    
    @abc.abstractmethod
    def _generate_sample_from_state(self, state):
        pass

    @abc.abstractmethod
    def _init(self, obs, params, **kwargs):
        if 's' in params:
            self.startprob = np.tile(1.0 / self._nstates, self._nstates)
        if 't' in params:
            shape = (self._nstates, self._nstates)
            self.transmat = np.tile(1.0 / self._nstates, shape)

    @abc.abstractmethod
    def _init_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self._nstates),
                 'trans': np.zeros((self._nstates, self._nstates))}
        return stats

    @abc.abstractmethod
    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob, 
                                         posteriors, fwdlattice, bwdlattice):
        stats['nobs'] += 1
        stats['start'] += posteriors[0]

        zeta = np.zeros((self._nstates, self._nstates))
        for t in xrange(len(framelogprob) - 1):
            zeta = (np.tile(fwdlattice[t], (self._nstates, 1)).T
                    + self._log_transmat.T
                    + np.tile(framelogprob[t + 1] + bwdlattice[t + 1],
                              (self._nstates, 1)))
            stats['trans'] += np.exp(zeta - logsum(zeta))

    @abc.abstractmethod
    def _mstep(self, stats, params, **kwargs):
        if 's' in params:
            self.startprob = stats['start'] / stats['nobs']
        if 't' in params:
            self.transmat = normalize(stats['trans'], axis=1)


class GaussianHMM(_BaseHMM):
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
    labels : list, len `nstates`
        Optional labels for each state.

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
    >>> hmm = HMM('gaussian', nstates=2, ndim=1)

    See Also
    --------
    gmm : Gaussian mixture model
    """

    emission_type = 'gaussian'

    def __init__(self, nstates=1, ndim=1, cvtype='diag'):
        """Create a hidden Markov model.

        Initializes parameters such that every state has zero mean and
        identity covariance.

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

        super(GaussianHMM, self).__init__(nstates)
        self._ndim = ndim
        self._cvtype = cvtype
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
        means = np.asarray(means)
        if means.shape != (self._nstates, self._ndim):
            raise ValueError, 'means must have shape (nstates, ndim)'
        self._means = means.copy()

    @property
    def covars(self):
        """Covariance parameters for each state."""
        return self._covars

    @covars.setter
    def covars(self, covars):
        covars = np.asarray(covars)
        _validate_covars(covars, self._cvtype, self._nstates, self._ndim)
        self._covars = covars.copy()

    def _compute_obs_log_likelihood(self, obs):
        return lmvnpdf(obs, self._means, self._covars, self._cvtype)

    def _generate_sample_from_state(self, state):
        if self._cvtype == 'tied':
            cv = self._covars
        else:
            cv = self._covars[state]
        return sample_gaussian(self._means[state], cv, self._cvtype)

    def _init(self, obs, params='stmc', **kwargs):
        super(GaussianHMM, self)._init(obs, params=params)

        if 'm' in params:
            self._means, tmp = sp.cluster.vq.kmeans2(obs[0], self._nstates,
                                                     **kwargs)
        if 'c' in params:
            cv = np.cov(obs[0].T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars = _distribute_covar_matrix_to_match_cvtype(
                cv, self._cvtype, self._nstates)

    def _init_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._init_sufficient_statistics()
        stats['obs'] = np.zeros((self._nstates, self._ndim))
        if self._cvtype in ('spherical', 'diag'):
            stats['obs**2'] = np.zeros((self._nstates, self._ndim))
        elif self._cvtype in ('tied', 'full'):
            stats['obs.T*obs'] = np.zeros((self._nstates, self._ndim,
                                           self._ndim))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        w = posteriors.sum(axis=0)
        norm = np.tile(1.0 / w, (self._ndim, 1)).T

        stats['obs'] += np.dot(posteriors.T, obs) * norm

        if self._cvtype in ('spherical', 'diag'):
            stats['obs**2'] += np.dot(posteriors.T, obs**2) * norm
        elif self._cvtype in ('tied', 'full'):
            for t, o in enumerate(obs):
                obsTobs = np.outer(o, o)
                for c in xrange(self._nstates):
                    stats['obs.T*obs'][c] += posteriors[t,c] * obsTobs / w[c]
                    
    def _mstep(self, stats, params, min_covar=1.0, **kwargs):
        super(GaussianHMM, self)._mstep(stats, params)

        if 'm' in params:
            self._means = stats['obs'] / stats['nobs']

        if 'c' in params:
            if self._cvtype in ('spherical', 'diag'):
                cv = (stats['obs**2'] / stats['nobs']
                      - 2 * self._means * stats['obs'] / stats['nobs']
                      + self._means ** 2 + min_covar)
                if self._cvtype == 'spherical':
                    self._covars = cv.mean(axis=1)
                elif self._cvtype == 'diag':
                    self._covars = cv
            elif self._cvtype in ('tied', 'full'):
                self._covars[:] = 0
                for c in xrange(self._nstates):
                    cv = (stats['obs.T*obs'][c] / stats['nobs']
                          - 2 * np.outer(stats['obs'][c] / stats['nobs'],
                                         self._means[c])
                          + np.outer(self._means[c], self._means[c])
                          + min_covar * np.eye(self._ndim))
                    if self._cvtype == 'tied':
                        self._covars += cv / self._nstates
                    elif self._cvtype == 'full':
                        self._covars[c] = cv


class GMMHMM(_BaseHMM):
    emission_type = 'gmm'

