import itertools
import unittest

from numpy.testing import *
import numpy as np
import scipy as sp
import scipy.stats

from test_gmm import _generate_random_spd_matrix

import hmm

class TestGaussianHMM(unittest.TestCase):
    cvtypes = ['spherical', 'tied', 'diag', 'full']

    nstates = 10
    ndim = 4
    startprob = np.random.rand(nstates)
    startprob = startprob / startprob.sum()
    transmat = np.random.rand(nstates, nstates)
    transmat /= np.tile(transmat.sum(axis=1)[:,np.newaxis], (1, nstates))
    means = np.random.randint(-20, 20, (nstates, ndim))
    covars = {'spherical': (0.1 + 2 * np.random.rand(nstates))**2,
              'tied': _generate_random_spd_matrix(ndim),
              'diag': (0.1 + 2 * np.random.rand(nstates, ndim))**2,
              'full': np.array([_generate_random_spd_matrix(ndim)
                                for x in xrange(nstates)])}
    def test_bad_cvtype(self):
        for cvtype in self.cvtypes:
            g = hmm._GaussianHMM(20, 1, cvtype)
        self.assertRaises(ValueError, hmm.HMM, 20, 1, 'badcvtype')

    def _test_attributes(self, cvtype):
        g = hmm._GaussianHMM(self.nstates, self.ndim, cvtype)

        self.assertEquals(g.emission_type, 'gaussian')
        
        self.assertEquals(g.nstates, self.nstates)
        self.assertEquals(g.ndim, self.ndim)
        self.assertEquals(g.cvtype, cvtype)

        g.startprob = self.startprob
        assert_array_almost_equal(g.startprob, self.startprob)
        self.assertRaises(ValueError, g.__setattr__, 'startprob',
                          2 * self.startprob)
        self.assertRaises(ValueError, g.__setattr__, 'startprob', [])
        self.assertRaises(ValueError, g.__setattr__, 'startprob',
                          np.zeros((self.nstates - 2, self.ndim)))

        g.transmat = self.transmat
        assert_array_almost_equal(g.transmat, self.transmat)
        self.assertRaises(ValueError, g.__setattr__, 'transmat',
                          2 * self.transmat)
        self.assertRaises(ValueError, g.__setattr__, 'transmat', [])
        self.assertRaises(ValueError, g.__setattr__, 'transmat',
                          np.zeros((self.nstates - 2, self.nstates)))

        g.means = self.means
        assert_array_almost_equal(g.means, self.means)
        self.assertRaises(ValueError, g.__setattr__, 'means', [])
        self.assertRaises(ValueError, g.__setattr__, 'means',
                          np.zeros((self.nstates - 2, self.ndim)))

        g.covars = self.covars[cvtype]
        assert_array_almost_equal(g.covars, self.covars[cvtype])
        self.assertRaises(ValueError, g.__setattr__, 'covars', [])
        self.assertRaises(ValueError, g.__setattr__, 'covars',
                          np.zeros((self.nstates - 2, self.ndim)))

    def test_attributes_spherical(self):
        self._test_attributes('spherical')
    def test_attributes_tied(self):
        self._test_attributes('tied')
    def test_attributes_diag(self):
        self._test_attributes('diag')
    def test_attributes_full(self):
        self._test_attributes('full')

    def _test_eval(self, cvtype):
        g = hmm._GaussianHMM(self.nstates, self.ndim, cvtype)
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        g.means = 20 * self.means
        g.covars = self.covars[cvtype]

        gaussidx = np.repeat(range(self.nstates), 5)
        nobs = len(gaussidx)
        obs = np.random.randn(nobs, self.ndim) + g.means[gaussidx]

        ll, posteriors = g.eval(obs)

        self.assertEqual(len(ll), nobs)
        self.assertEqual(posteriors.shape, (nobs, self.nstates))
        assert_array_almost_equal(posteriors.sum(axis=1), np.ones(nobs))
        assert_array_equal(posteriors.argmax(axis=1), gaussidx)

    def test_eval_spherical(self):
        self._test_eval('spherical')
    def test_eval_tied(self):
        self._test_eval('tied')
    def test_eval_diag(self):
        self._test_eval('diag')
    def test_eval_full(self):
        self._test_eval('full')

    def _test_rvs(self, cvtype, n=1000):
        g = hmm._GaussianHMM(self.nstates, self.ndim, cvtype)
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        g.means = 20 * self.means
        g.covars = np.maximum(self.covars[cvtype], 0.1)
        g.startprob = self.startprob

        samples = g.rvs(n)
        self.assertEquals(samples.shape, (n, self.ndim))

    def test_rvs_spherical(self):
        self._test_rvs('spherical')
    def test_rvs_tied(self):
        self._test_rvs('tied')
    def test_rvs_diag(self):
        self._test_rvs('diag')
    def test_rvs_full(self):
        self._test_rvs('full')

    def _test_train(self, cvtype):
        g = hmm._GaussianHMM(self.nstates, self.ndim, cvtype)
        g.startprob = self.startprob
        g.means = self.means
        g.covars = 20*self.covars[cvtype]

        # Create a training and testing set by sampling from the same
        # distribution.
        train_obs = g.rvs(n=200)
        test_obs = g.rvs(n=20)

        g.init(train_obs, minit='points')
        init_testll = g.eval(test_obs)[0].sum()

        trainll = g.train(train_obs)
        self.assert_(np.all(np.diff(trainll) > -1))

        post_testll = g.eval(test_obs)[0].sum()
        self.assertTrue(post_testll > init_testll)

    def test_train_spherical(self):
        self._test_train('spherical')
    def test_train_tied(self):
        self._test_train('tied')
    def test_train_diag(self):
        self._test_train('diag')
    def test_train_full(self):
        self._test_train('full')
        

if __name__ == '__main__':
    unittest.main()
