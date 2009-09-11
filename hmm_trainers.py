import abc
import time

import numpy as np

import hmm
from gmm import *

# FIXME - remove trainers from HMM classes so that we can easily swap
# in adaptation, etc.
# trainer=DefaultTrainer
# class DefaultTrainer(BaseHMMTrainer):
#     def train(hmm, ...
# can have instance variances
# class MAPAdaptationTrainer


class HMMTrainer(object):
    """Abstract base class for HMM training algorithms."""

    __metaclass__ = abc.ABCMeta

    def train(self, hmm, obs, iter=10, thresh=1e-2, params='stmpc',
              maxrank=None, beamlogprob=-np.Inf, **kwargs):
        """Estimate model parameters.

        Parameters
        ----------
        hmm : HMM object
            HMM to train.
        obs : list
            List of array-like observation sequences (shape (n_i, ndim)).
        iter : int
            Number of iterations to perform.
        thresh : float
            Convergence threshold.
        params : string
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'm' for means, and 'c' for covars.
            Defaults to all parameters ('stmpc').
        maxrank : int
            Maximum rank to evaluate for rank pruning.  If not None,
            only consider the top `maxrank` states in the inner
            sum of the forward algorithm recursion.  Defaults to None
            (no rank pruning).  See "The HTK Book" for more details.
        beamlogprob : float
            Width of the beam-pruning beam in log-probability units.
            Defaults to -numpy.Inf (no beam pruning).  See The HTK
            Book for more details.

        Returns
        -------
        logprob : list
            Log probabilities of the training data after each iteration.
        """

        T = time.time()
        logprob = []
        for i in xrange(iter):
            # Expectation step
            stats = self._initialize_sufficient_statistics(hmm)
            curr_logprob = 0
            for seq in obs:
                framelogprob = hmm._compute_log_likelihood(seq)
                lpr, fwdlattice = hmm._do_forward_pass(framelogprob, maxrank,
                                                       beamlogprob)
                bwdlattice = hmm._do_backward_pass(framelogprob, fwdlattice,
                                                   maxrank, beamlogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsum(gamma, axis=1)).T

                curr_logprob += lpr
                self._accumulate_sufficient_statistics(hmm, stats, seq,
                                                       framelogprob, posteriors,
                                                       fwdlattice, bwdlattice)
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
            self._do_mstep(hmm, stats, params, **kwargs)

        return logprob

    def _initialize_sufficient_statistics(self, hmm):
        pass

    def _accumulate_sufficient_statistics(self, hmm, stats, seq, framelogprob, 
                                          posteriors, fwdlattice, bwdlattice):
        pass
    
    def _do_mstep(self, hmm, stats, params, **kwargs):
        pass


class BaseHMMBaumWelchTrainer(HMMTrainer):
    """Abstract base class for HMM trainers.

    Uses the Baum-Welch algorithm to train the startprob and transmat
    parameters.
    """

    def _initialize_sufficient_statistics(self, hmm):
        stats = {'nobs': 0,
                 'start': np.zeros(hmm._nstates),
                 'trans': np.zeros((hmm._nstates, hmm._nstates))}
        return stats

    def _accumulate_sufficient_statistics(self, hmm, stats, seq, framelogprob, 
                                          posteriors, fwdlattice, bwdlattice):
        stats['nobs'] += 1
        stats['start'] += posteriors[0]

        zeta = np.zeros((hmm._nstates, hmm._nstates))
        for t in xrange(len(framelogprob) - 1):
            zeta = (np.tile(fwdlattice[t], (hmm._nstates, 1)).T
                    + hmm._log_transmat.T
                    + np.tile(framelogprob[t + 1] + bwdlattice[t + 1],
                              (hmm._nstates, 1)))
            stats['trans'] += np.exp(zeta - logsum(zeta))

    def _do_mstep(self, hmm, stats, params, **kwargs):
        if 's' in params:
            hmm.startprob = stats['start'] / stats['nobs']
        if 't' in params:
            hmm.transmat = normalize(stats['trans'], axis=1)


class GaussianHMMBaumWelchTrainer(BaseHMMBaumWelchTrainer):
    """Standard Baum-Welch trainer for HMMs with Gaussian emissions."""

    def _initialize_sufficient_statistics(self, hmm):
        stats = super(GaussianHMMBaumWelchTrainer,
                      self)._initialize_sufficient_statistics(hmm)

        stats['obs'] = np.zeros((hmm._nstates, hmm._ndim))
        if hmm._cvtype in ('spherical', 'diag'):
            stats['obs**2'] = np.zeros((hmm._nstates, hmm._ndim))
        elif hmm._cvtype in ('tied', 'full'):
            stats['obs.T*obs'] = np.zeros((hmm._nstates, hmm._ndim,
                                           hmm._ndim))
        return stats

    def _accumulate_sufficient_statistics(self, hmm, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GaussianHMMBaumWelchTrainer,
              self)._accumulate_sufficient_statistics(hmm, stats, obs,
                                                      framelogprob, posteriors,
                                                       fwdlattice, bwdlattice)
        w = posteriors.sum(axis=0)
        norm = np.tile(1.0 / w, (hmm._ndim, 1)).T

        stats['obs'] += np.dot(posteriors.T, obs) * norm

        if hmm._cvtype in ('spherical', 'diag'):
            stats['obs**2'] += np.dot(posteriors.T, obs**2) * norm
        elif hmm._cvtype in ('tied', 'full'):
            for t, o in enumerate(obs):
                obsTobs = np.outer(o, o)
                for c in xrange(hmm._nstates):
                    stats['obs.T*obs'][c] += posteriors[t,c] * obsTobs / w[c]
                    
    def _do_mstep(self, hmm, stats, params, min_covar=1.0, **kwargs):
        super(GaussianHMMBaumWelchTrainer, self)._do_mstep(hmm, stats, params)

        if 'm' in params:
            hmm._means = stats['obs'] / stats['nobs']

        if 'c' in params:
            if hmm._cvtype in ('spherical', 'diag'):
                cv = (stats['obs**2'] / stats['nobs']
                      - 2 * hmm._means * stats['obs'] / stats['nobs']
                      + hmm._means ** 2 + min_covar)
                if hmm._cvtype == 'spherical':
                    hmm._covars = cv.mean(axis=1)
                elif hmm._cvtype == 'diag':
                    hmm._covars = cv
            elif hmm._cvtype in ('tied', 'full'):
                hmm._covars[:] = 0
                for c in xrange(hmm._nstates):
                    cv = (stats['obs.T*obs'][c] / stats['nobs']
                          - 2 * np.outer(stats['obs'][c] / stats['nobs'],
                                         hmm._means[c])
                          + np.outer(hmm._means[c], hmm._means[c])
                          + min_covar * np.eye(hmm._ndim))
                    if hmm._cvtype == 'tied':
                        hmm._covars += cv / hmm._nstates
                    elif hmm._cvtype == 'full':
                        hmm._covars[c] = cv


class GaussianHMMMAPTrainer(GaussianHMMBaumWelchTrainer):
    """HMM trainer based on maximum-a-posteriori (MAP) adaptation.
    """

    # FIXME means and covars aren't being normalized correctly
    def __init__(self, priorhmm, weights={}):
        self.priorhmm = priorhmm

        for field in ['means', 'covars']:
            if field not in weights:
                weights[field] = 1.0
        self.weights = weights

    def _do_mstep(self, hmm, stats, params, **kwargs):
        # Based on Huang, Acero, Hon, "Spoken Language Processing", p. 443 - 445
        if 's' in params:
            hmm.startprob = ((self.priorhmm.startprob - 1 + stats['start'])
                             / np.sum(self.priorhmm.startprob - 1
                                      + stats['start']))
        if 't' in params:
            transmat = np.empty(hmm.transmat.shape)
            for n in xrange(hmm.nstates):
                transmat[n] = ((self.priorhmm.transmat[n] - 1
                                + stats['trans'][n])
                               / np.sum(self.priorhmm.transmat[n] - 1
                                        + stats['trans'][n]))
            #hmm.transmat = normalize(transmat, axis=1)
            hmm.transmat = transmat

        if 'm' in params:
            weight = self.weights['means']
            hmm._means = ((weight * self.priorhmm.means + stats['obs'])
                          / (weight + stats['nobs']))

        if 'c' in params:
            wM = self.weights['means']
            wC = self.weights['covars']
            meandiff = hmm._means - self.priorhmm.means
            if hmm._cvtype in ('spherical', 'diag'):
                cv = (((wC - 1) * self.priorhmm.covars
                       + wM * (meandiff)**2
                       + stats['obs**2'] 
                       - 2 * hmm._means * stats['obs']
                       + hmm._means ** 2)
                      / (wC - 1 + stats['nobs']))
                if hmm._cvtype == 'spherical':
                    hmm._covars = cv.mean(axis=1)
                elif hmm._cvtype == 'diag':
                    hmm._covars = cv
            elif hmm._cvtype in ('tied', 'full'):
                hmm._covars[:] = 0
                print hmm._means.shape, hmm._nstates
                c = 2
                print np.outer(hmm._means[c], hmm._means[c])
                for c in xrange(hmm._nstates):
                    print c, hmm.means[c]
                    cv = (((wC - 1) * self.priorhmm.covars[c]
                           + wM * np.outer(meandiff[c], meandiff[c])
                           + stats['obs.T*obs'][c] 
                           - 2 * np.outer(stats['obs'][c], hmm._means[c])
                           + np.outer(hmm._means[c], hmm._means[c]))
                          / (wC - 1 + stats['nobs']))
                    if hmm._cvtype == 'tied':
                        hmm._covars += cv / hmm._nstates
                    elif hmm._cvtype == 'full':
                        hmm._covars[c] = cv
            print 'COVARS (%s)' % self.cvtype, hmm._covars
