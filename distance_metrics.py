import numpy as np

import gmm
import hmm

def bhattacharyya_divergence(model1, model2, dist_type='jensen', norm=True):
    """Returns an approximation to (or lower bound on if norm is true) the
    Bhattacharyya divergence between gmm1 and gmm2.  The returned
    divergence is related to the Bhattacharyya coefficient
    (i.e. similarity) by:  Bdiv = -log(Bsim)
    
    This funcation can compute various approximations described in:
    J. R. Hershey and P. A. Olsen, "Variational Bhattacharyya Divergence
    for Hidden Markov Models", ICASSP 2008.

    dist_type controls the type of approximation and can be either 'jensen'
    (default) or 'variational'.  If norm is true (default) the
    divergence will be normalized such that Bdiv = -log(1/2) if gmm1 ==
    gmm2 and > -log(1/2) otherwise.

    Only supports diagonal covariances.
    """

    if type(model1) != type(model2):
        raise ValueError, 'model1 and model2 must have the same type.'

    if model1.nstates > model2.nstates:
        model1, model2 = model2, model1

    if isinstance(model1, gmm.GMM):
        funcs = {'jensen': _gmm_compute_jensen_lower_bound,
                 'variational': _gmm_compute_variational_lower_bound}
    elif isinstance(model1, hmm._BaseHMM):
        funcs = {'jensen': _hmm_compute_jensen_lower_bound,
                 'slowjensen': _hmm_compute_jensen_lower_bound_slow,
                 'variational': _hmm_compute_variational_lower_bound}
    try:
        fun = funcs[dist_type]
    except KeyError:
        raise ValueError, 'Distance type ''%s'' not supported.' % dist_type

    Be12 = fun(model1, model2)
    if norm:
        Be11 = fun(model1, model1)
        Be22 = fun(model2, model2)
        Bnormed = Be12 / (2 * np.sqrt(Be11 * Be22))
    else:
        Bnormed = Be12

    return -np.log(Bnormed)

def _compute_pairwise_gaussian_distances(model1, model2):
    means1 = model1.means
    covars1 = model1.covars
    means2 = model2.means
    covars2 = model2.covars

    funcs = {'diag': _compute_pairwise_gaussian_distances_diag,
             'full': _compute_pairwise_gaussian_distances_full}

    if model1.cvtype == model2.cvtype:
        return funcs[model1.cvtype](means1, covars1, means2, covars2)

def _compute_pairwise_gaussian_distances_diag(means1, covars1, means2,
                                              covars2):
    n1 = len(means1)
    n2 = len(means2)
    Db = np.zeros((n1, n2))
    for s1 in xrange(n1):
        mud = means2 - means1[s1]
        cvs = covars2 + covars1[s1]
        Db[s1] = (np.sum(mud**2 / cvs, 1) + 2*np.sum(np.log(cvs / 2), 1)) / 4
    detcvp = (np.tile(np.sum(np.log(covars1), 1), (n2, 1)).T
              + np.tile(np.sum(np.log(covars2), 1), (n1, 1)))
    return Db - detcvp / 4

def _compute_pairwise_gaussian_distances_full(means1, covars1, means2,
                                              covars2):
    n1 = len(means1)
    n2 = len(means2)
    Db = np.zeros((n1, n2))
    for s1 in xrange(n1):
        for s2 in xrange(n2):
            mud = means2[s2] - means1[s1]
            cvs = covars2[s2] + covars1[s1]
            Db[s1,s2] = (np.dot(np.dot(mud.T, np.linalg.inv(cvs)), mud)
                         + 2 * np.log(np.linalg.det(cvs / 2))) / 4
            
    dets1 = np.asarray([np.linalg.det(x) for x in covars1])
    dets2 = np.asarray([np.linalg.det(x) for x in covars2])
    detcvp = np.log(np.outer(dets1, dets2))

    return Db - detcvp / 4

def _gmm_compute_jensen_lower_bound(gmm1, gmm2):
    Db = _compute_pairwise_gaussian_distances(gmm1, gmm2)
    Bjb = np.exp(-Db) * np.outer(gmm1.weights, gmm2.weights)
    return  Bjb.sum()

def _gmm_compute_variational_lower_bound(gmm1, gmm2):
    Db = _compute_pairwise_gaussian_distances(gmm1, gmm2)
    Bvb = np.exp(-Db)**2 * np.outer(gmm1.weights, gmm2.weights)
    return np.sqrt(Bvb.sum())

def vec(a):
    return a.flatten('F')

def _hmm_compute_jensen_lower_bound_slow(hmm1, hmm2):
    # This is way too slow (because it requires *a lot* of memory)
    # unless the hmms contain very few states
    B = np.exp(-_compute_pairwise_gaussian_distances(hmm1, hmm2))
    vecB = vec(B)
    A = (np.kron(hmm1.transmat, hmm2.transmat)
         * np.kron(vecB[:,np.newaxis], np.ones(hmm1.nstates * hmm2.nstates)))

    I = np.eye(hmm1.nstates * hmm2.nstates)
    C = np.linalg.inv(I - A) - I
    vI = np.kron(hmm1.startprob, hmm2.startprob)
    vF = np.kron(np.tile(1.0 / hmm1.nstates, (hmm1.nstates)),
                 np.tile(1.0 / hmm2.nstates, (hmm2.nstates)))

    # Equation 23 from Hershey and Olsen
    Bjb = np.dot((vF * vecB).T, np.dot(C, vI))
    return Bjb

def _hmm_compute_jensen_lower_bound(hmm1, hmm2, thresh=1e-10):
    B = np.exp(-_compute_pairwise_gaussian_distances(hmm1, hmm2))
    Bjb = 0
    Bjbprev = np.Inf
    Btilde = np.outer(hmm1.startprob, hmm2.startprob)
    n = 0
    while np.abs(Bjb - Bjbprev).sum() > thresh:
        #print n, np.abs(Bjb - Bjbprev).sum(), Bjb
        n += 1
        Btilde = np.dot(hmm1.transmat.T, np.dot(B * Btilde, hmm2.transmat))
        Bjbprev = Bjb
        Bjb += Btilde.mean()
    return Bjb

def _hmm_compute_variational_lower_bound(hmm1, hmm2, thresh=1e-10):
    B2 = np.exp(-_compute_pairwise_gaussian_distances(hmm1, hmm2))**2
    Bvb = 0
    Bvbprev = np.Inf
    Btilde = np.outer(hmm1.startprob, hmm2.startprob)
    n = 0
    while np.abs(Bvb - Bvbprev).sum() > thresh:
        #print n, np.abs(Bvb - Bvbprev).sum(), Bvb
        n += 1
        Btilde = np.dot(hmm1.transmat.T, np.dot(B2 * Btilde, hmm2.transmat))
        Bvbprev = Bvb
        Bvb += np.sqrt(Btilde.mean())
    return Bvb
