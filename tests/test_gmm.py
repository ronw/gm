import itertools
import unittest

from numpy.testing import *
import numpy as np
import scipy as sp
import scipy.stats

import gmm


def _generate_random_spd_matrix(ndim):
    """Return a random symmetric, positive-definite matrix."""
    A = np.random.rand(ndim, ndim)
    U, s, V = np.linalg.svd(np.dot(A.T, A))
    randspd = np.dot(np.dot(U, 1.0+np.diag(np.random.rand(ndim))), V)
    return randspd

class TestLogsum(unittest.TestCase):
    def test_logsum_1D(self):
        A = np.random.rand(10) + 1.0
        Asum = gmm.logsum(A)
        self.assertAlmostEqual(np.exp(Asum), np.sum(np.exp(A)))

    def test_logsum_2D(self):
        A = np.random.rand(10, 4) + 1.0
        Asum = gmm.logsum(A)
        self.assertAlmostEqual(np.exp(Asum), np.sum(np.exp(A)))

    def test_logsum_with_axis_1D(self):
        A = np.random.rand(10) + 1.0
        for axis in range(1):
            Asum = gmm.logsum(A, axis)
            assert_array_almost_equal(np.exp(Asum), np.sum(np.exp(A), axis))

    def test_logsum_with_axis_2D(self):
        A = np.random.rand(10, 4) + 1.0
        for axis in range(2):
            Asum = gmm.logsum(A, axis)
            assert_array_almost_equal(np.exp(Asum), np.sum(np.exp(A), axis))

    def test_logsum_with_axis_3D(self):
        A = np.random.rand(10, 4, 5) + 1.0
        for axis in range(3):
            Asum = gmm.logsum(A, axis)
            assert_array_almost_equal(np.exp(Asum), np.sum(np.exp(A), axis))

class TestSampleGaussian(unittest.TestCase):
    def _test_sample_gaussian_diag(self, ndim, n=10000):
        mu = np.random.randint(10) * np.random.rand(ndim)
        cv = (np.random.rand(ndim) + 1.0)**2

        samples = gmm.sample_gaussian(mu, cv, cvtype='diag', n=n)

        if ndim > 1:
            axis = 1
        else:
            axis = None
        assert_array_almost_equal(samples.mean(axis), mu, decimal=1)
        assert_array_almost_equal(samples.var(axis), cv, decimal=1)

    def test_sample_gaussian_diag_1D(self):
        self._test_sample_gaussian_diag(1)
    def test_sample_gaussian_diag_2D(self):
        self._test_sample_gaussian_diag(2)
    def test_sample_gaussian_diag_10D(self):
        self._test_sample_gaussian_diag(10)

    def _test_sample_gaussian_spherical(self, ndim, n=10000):
        mu = np.random.randint(10) * np.random.rand(ndim)
        cv = (np.random.rand() + 1.0)**2

        samples = gmm.sample_gaussian(mu, cv, cvtype='spherical', n=n)

        if ndim > 1:
            axis = 1
        else:
            axis = None
        assert_array_almost_equal(samples.mean(axis), mu, decimal=1)
        assert_array_almost_equal(samples.var(axis), np.repeat(cv, ndim),
                                  decimal=1)

    def test_sample_gaussian_spherical_1D(self):
        self._test_sample_gaussian_spherical(1)
    def test_sample_gaussian_spherical_2D(self):
        self._test_sample_gaussian_spherical(2)
    def test_sample_gaussian_spherical_10D(self):
        self._test_sample_gaussian_spherical(10)

    def _test_sample_gaussian_full(self, ndim, n=10000):
        mu = np.random.randint(10) * np.random.rand(ndim)
        cv = _generate_random_spd_matrix(ndim)

        samples = gmm.sample_gaussian(mu, cv, cvtype='full', n=n)

        if ndim > 1:
            axis = 1
        else:
            axis = None
        assert_array_almost_equal(samples.mean(axis), mu, decimal=1)
        assert_array_almost_equal(np.cov(samples), cv, decimal=1)

    def test_sample_gaussian_full_1D(self):
        self._test_sample_gaussian_full(1)
    def test_sample_gaussian_full_2D(self):
        self._test_sample_gaussian_full(2)
    def test_sample_gaussian_full_10D(self):
        self._test_sample_gaussian_full(10)


class TestLmvnpdf(unittest.TestCase):
    def _slow_lmvnpdfdiag(self, obs, means, covars):
        lpr = np.empty((len(obs), len(means)))
        stds = np.sqrt(covars) 
        for c, (mu, std) in enumerate(itertools.izip(means, stds)):
            for o, currobs in enumerate(obs):
                lpr[o,c] = np.log(sp.stats.norm.pdf(currobs, mu, std)).sum()
        return lpr

    def _test_lmvnpdfdiag(self, ndim, nmix, nobs=100):
        # test the slow and naive implementation of lmvnpdf and
        # compare it to the vectorized version (gmm.lmvnpdf) to test
        # for correctness
        mu = np.random.randint(10) * np.random.rand(nmix, ndim)
        cv = (np.random.rand(nmix, ndim) + 1.0)**2
        obs = np.random.randint(10) * np.random.rand(nobs, ndim)

        reference = self._slow_lmvnpdfdiag(obs, mu, cv)
        lpr = gmm.lmvnpdf(obs, mu, cv, 'diag')
        assert_array_almost_equal(lpr, reference)

    def test_lmvnpdfdiag_univariate_single_gaussian(self):
        self._test_lmvnpdfdiag(1, 1)
    def test_lmvnpdfdiag_univariate(self):
        self._test_lmvnpdfdiag(1, 10)
    def test_lmvnpdfdiag_single_gaussian(self):
        self._test_lmvnpdfdiag(5, 1)
    def test_lmvnpdfdiag(self):
        self._test_lmvnpdfdiag(5, 10)

    def _test_lmvnpdfspherical(self, ndim, nmix, nobs=100):
        mu = np.random.randint(10) * np.random.rand(nmix, ndim)
        spherecv = np.random.rand(nmix, 1)**2 + 1
        obs = np.random.randint(10) * np.random.rand(nobs, ndim)

        cv = np.tile(spherecv, (ndim, 1))
        reference = self._slow_lmvnpdfdiag(obs, mu, cv)
        lpr = gmm.lmvnpdf(obs, mu, spherecv, 'spherical')
        assert_array_almost_equal(lpr, reference)

    def test_lmvnpdfspherical_univariate_single_gaussian(self):
        self._test_lmvnpdfspherical(1, 1)
    def test_lmvnpdfspherical_univariate(self):
        self._test_lmvnpdfspherical(1, 10)
    def test_lmvnpdfspherical_single_gaussian(self):
        self._test_lmvnpdfspherical(5, 1)
    def test_lmvnpdfspherical(self):
        self._test_lmvnpdfspherical(5, 10)

    def _test_lmvnpdffull_with_diagonal_covariance(self, ndim, nmix, nobs=100):
        mu = np.random.randint(10) * np.random.rand(nmix, ndim)
        cv = (np.random.rand(nmix, ndim) + 1.0)**2
        obs = np.random.randint(10) * np.random.rand(nobs, ndim)

        fullcv = np.array([np.diag(x) for x in cv])

        reference = self._slow_lmvnpdfdiag(obs, mu, cv)
        lpr = gmm.lmvnpdf(obs, mu, fullcv, 'full')
        assert_array_almost_equal(lpr, reference)

    def test_lmvnpdffull_with_diagonal_covariance_univariate_single_gaussian(
        self):
        self._test_lmvnpdffull_with_diagonal_covariance(1, 1)
    def test_lmvnpdffull_with_diagonal_covariance_univariate(self):
        self._test_lmvnpdffull_with_diagonal_covariance(1, 10)
    def test_lmvnpdffull_with_diagonal_covariance_single_gaussian(self):
        self._test_lmvnpdffull_with_diagonal_covariance(5, 1)
    def test_lmvnpdffull_with_diagonal_covariance(self):
        self._test_lmvnpdffull_with_diagonal_covariance(5, 10)

    def _test_lmvnpdftied_with_diagonal_covariance(self, ndim, nmix, nobs=100):
        mu = np.random.randint(10) * np.random.rand(nmix, ndim)
        tiedcv = (np.random.rand(ndim) + 1.0)**2
        obs = np.random.randint(10) * np.random.rand(nobs, ndim)

        cv = np.tile(tiedcv, (nmix, 1))

        reference = self._slow_lmvnpdfdiag(obs, mu, cv)
        lpr = gmm.lmvnpdf(obs, mu, np.diag(tiedcv), 'tied')
        assert_array_almost_equal(lpr, reference)
        
    def test_lmvnpdftied_with_diagonal_covariance_univariate_single_gaussian(
        self):
        self._test_lmvnpdftied_with_diagonal_covariance(1, 1)
    def test_lmvnpdftied_with_diagonal_covariance_univariate(self):
        self._test_lmvnpdftied_with_diagonal_covariance(1, 10)
    def test_lmvnpdftied_with_diagonal_covariance_single_gaussian(self):
        self._test_lmvnpdftied_with_diagonal_covariance(5, 1)
    def test_lmvnpdftied_with_diagonal_covariance(self):
        self._test_lmvnpdftied_with_diagonal_covariance(5, 10)

    def test_lmvnpdftied_consistent_with_lmvnpdffull(self):
        nmix = 4
        ndim = 20
        nobs = 200
        
        mu = np.random.randint(10) * np.random.rand(nmix, ndim)
        tiedcv = _generate_random_spd_matrix(ndim)
        obs = np.random.randint(10) * np.random.rand(nobs, ndim)

        cv = np.tile(tiedcv, (nmix, 1, 1))

        reference = gmm.lmvnpdf(obs, mu, cv, 'full')
        lpr = gmm.lmvnpdf(obs, mu, tiedcv, 'tied')
        assert_array_almost_equal(lpr, reference)


class TestGMM(unittest.TestCase):
    cvtypes = ['spherical', 'tied', 'diag', 'full']

    nmix = 10
    ndim = 4
    weights = np.random.rand(nmix)
    weights = weights / weights.sum()
    means = np.random.randint(-20, 20, (nmix, ndim))
    covars = {'spherical': (0.1 + 2 * np.random.rand(nmix))**2,
              'tied': _generate_random_spd_matrix(ndim),
              'diag': (0.1 + 2 * np.random.rand(nmix, ndim))**2,
              'full': np.array([_generate_random_spd_matrix(ndim)
                                for x in xrange(nmix)])}
    def test_bad_cvtype(self):
        for cvtype in self.cvtypes:
            g = gmm.GMM(20, 1, cvtype)

        self.assertRaises(ValueError, gmm.GMM, 20, 1, 'badcvtype')

    def _test_attributes(self, cvtype):
        g = gmm.GMM(self.nmix, self.ndim, cvtype)
        self.assertEquals(g.nmix, self.nmix)
        self.assertEquals(g.ndim, self.ndim)
        self.assertEquals(g.cvtype, cvtype)

        g.weights = self.weights
        assert_array_almost_equal(g.weights, self.weights)
        self.assertRaises(ValueError, g.__setattr__, 'weights', [])
        self.assertRaises(ValueError, g.__setattr__, 'weights',
                          np.zeros((self.nmix - 2, self.ndim)))

        g.means = self.means
        assert_array_almost_equal(g.means, self.means)
        self.assertRaises(ValueError, g.__setattr__, 'means', [])
        self.assertRaises(ValueError, g.__setattr__, 'means',
                          np.zeros((self.nmix - 2, self.ndim)))

        g.covars = self.covars[cvtype]
        assert_array_almost_equal(g.covars, self.covars[cvtype])
        self.assertRaises(ValueError, g.__setattr__, 'covars', [])
        self.assertRaises(ValueError, g.__setattr__, 'covars',
                          np.zeros((self.nmix - 2, self.ndim)))

    def test_attributes_spherical(self):
        self._test_attributes('spherical')
    def test_attributes_tied(self):
        self._test_attributes('tied')
    def test_attributes_diag(self):
        self._test_attributes('diag')
    def test_attributes_full(self):
        self._test_attributes('full')

    def _test_eval(self, cvtype):
        g = gmm.GMM(self.nmix, self.ndim, cvtype)
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        g.means = 20 * self.means
        g.covars = self.covars[cvtype]

        gaussidx = np.repeat(range(self.nmix), 5)
        nobs = len(gaussidx)
        obs = np.random.randn(nobs, self.ndim) + g.means[gaussidx]

        ll, posteriors = g.eval(obs)

        self.assertEqual(len(ll), nobs)
        self.assertEqual(posteriors.shape, (nobs, self.nmix))
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
        g = gmm.GMM(self.nmix, self.ndim, cvtype)
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        g.means = 20 * self.means
        g.covars = np.maximum(self.covars[cvtype], 0.1)
        g.weights = self.weights

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
        pass

if __name__ == '__main__':
    unittest.main()
