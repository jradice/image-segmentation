from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import scipy as sp
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from helper_functions import image_to_matrix, matrix_to_image, flatten_image_matrix, unflatten_image_matrix, image_difference

from random import randint
from functools import reduce
def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    x = image_values.reshape(-1,image_values.shape[-1])
    # Initialize inital_means if necessary
    if initial_means is None:
        initial_means = x[np.random.choice(x.shape[0], size=k, replace=False)]
    u = initial_means
    dist = np.empty(x.shape[:1] + (k,))
    prev_r = None
    while True:
        # Compute sum of squares distance
        for j in range(0,k):
            dist[:,j] = np.sum((x[:]-u[j])**2, axis=1)
        # Compute min sum of square distance and r
        min_dist = np.min(dist, axis=1, out=None, keepdims=True)
        r = dist
        r[r == min_dist] = 1
        r[r != 1] = 0
        # Check for convergence
        if (not prev_r is None) and (r == prev_r).all():
            break
        prev_r = np.copy(r)
        # Compute u
        for j in range(0,k):
            if np.sum(r[:,j]) == 0:
                u[j] = [0,0,0]
            else:
                u[j] = np.sum(r[:,j].reshape(r.shape[:1] + (1,))*x[:], axis=0)/np.sum(r[:,j], axis=0)
    # Construct segmented image
    r = r.astype(int)
    updated_image_values = np.empty(image_values.shape)
    for index in np.ndindex(updated_image_values.shape[:2]):
        updated_image_values[index] = np.array(u[np.argmax(r[index[0]*updated_image_values.shape[1] + index[1]])])
    return updated_image_values

def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr+=1
    else:
        conv_ctr =0

    return conv_ctr, conv_ctr > conv_ctr_cap

from random import randint
import math
from scipy.misc import logsumexp
class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = [0]*num_components
        else:
            self.means = means
        self.variances = [0]*num_components
        self.mixing_coefficients = [0]*num_components

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        means = np.array(self.means)
        variances = np.array(self.variances)
        mixing = np.array(self.mixing_coefficients)
        joint_prob = mixing*(1.0/np.sqrt(2.0*variances*math.pi))*np.exp(-1.0*((val - means)**2/(2.0*variances)))
        
        return math.log(np.sum(joint_prob))

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        x = self.image_matrix.flatten()
        self.means = x[np.random.choice(x.shape[0], size=self.num_components, replace=False)].tolist()
        self.variances[:] = [1.0] * self.num_components
        self.mixing_coefficients[:] = [1.0/self.num_components] * self.num_components

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function that returns True if convergence is reached
        """
        # TODO: finish this
        self.initialize_training()
        log_likelihood = self.likelihood()
        x = self.image_matrix.flatten()
        means = np.array(self.means)
        variances = np.array(self.variances)
        mixing = np.array(self.mixing_coefficients)
        count = 0
        convergence = False
        while not convergence:
            # E step
            resp = np.empty((self.num_components, x.shape[0]))
            for k in range(0, self.num_components):
                resp[k] = mixing[k]*(1.0/np.sqrt(2.0*variances[k]*math.pi))*np.exp(-1.0*((x - means[k])**2/(2.0*variances[k])))
            resp = resp/np.sum(resp, axis=0)
            resp = resp.T
            # M step
            means = np.sum(resp*x[:, np.newaxis], axis=0)/np.sum(resp, axis=0)
            variances = np.sum(resp*np.square(x[:, np.newaxis] - means), axis=0)/np.sum(resp, axis=0)
            mixing = np.sum(resp, axis=0)/resp.shape[0]
            self.means = means
            self.variances = variances
            self.mixing_coefficients = mixing
            # Evaluate likelihood and check for convergence
            prev_likelihood = log_likelihood
            log_likelihood = self.likelihood()
            count, convergence = convergence_function(prev_likelihood, 
                                                      log_likelihood, 
                                                      count)

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        x = self.image_matrix
        means = np.array(self.means)
        variances = np.array(self.variances)
        mixing = np.array(self.mixing_coefficients)
        segment = np.empty(x.shape)
        for index in np.ndindex(segment.shape):
            likelihoods = mixing*(1.0/np.sqrt(2.0*variances*math.pi))*np.exp(-1.0*((x[index] - means)**2/(2.0*variances)))
            max_likelihood = np.argmax(likelihoods)
            segment[index] = means[max_likelihood]
        return segment

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N),ln(sum((k=1 to K), mixing_k * N(x_n | mean_k, stdev_k) )))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        x = self.image_matrix.flatten()
        means = np.array(self.means)
        variances = np.array(self.variances)
        mixing = np.array(self.mixing_coefficients)
        joint_probs = np.empty((self.num_components, x.shape[0]))
        for k in range(0, self.num_components):
            joint_probs[k] = mixing[k]*(1.0/np.sqrt(2.0*variances[k]*math.pi))*np.exp(-1.0*((x - means[k])**2/(2.0*variances[k])))
        joint_probs = np.sum(joint_probs, axis=0)
        joint_probs = np.log(joint_probs)
        log_likelihood = np.sum(joint_probs)
        return log_likelihood

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        likelihoods = []
        segments = []
        for i in range(0, iters):
            self.train_model()
            likelihoods.append(self.likelihood())
            segments.append(self.segment())
        segment =  segments[likelihoods.index(max(likelihoods))]
        return segment

class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient initializations too if that works well.]
        """
        # TODO: finish this
        self.variances[:] = [1.0] * self.num_components
        self.mixing_coefficients[:] = [1.0/self.num_components] * self.num_components
        # Intialize means using k-clustering
        k = self.num_components
        x = self.image_matrix.flatten()
        # Initialize means with random k pixels
        self.means = x[np.random.choice(x.shape[0], size=k, replace=False)]
        u = self.means
        dist = np.empty(x.shape + (k,))
        prev_r = None
        while True:
            # Compute sum of squares distance
            for j in range(0,k):
                dist[:,j] = (x - u[j])**2
            # Compute min sum of square distance and r
            min_dist = np.min(dist, axis=1, out=None, keepdims=True)
            r = dist
            r[r == min_dist] = 1
            r[r != 1] = 0
            # Check for convergence
            if (not prev_r is None) and (r == prev_r).all():
                break
            prev_r = np.copy(r)
            # Compute u
            for j in range(0,k):
                if np.sum(r[:,j]) == 0:
                    u[j] = 0.0
                else:
                    u[j] = np.sum(r[:,j]*x[:], axis=0)/np.sum(r[:,j], axis=0)
        self.means = u

def new_convergence_function(previous_variables, new_variables, conv_ctr, conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    diff = np.absolute(new_variables - previous_variables)
    threshold = diff/previous_variables
    
    if threshold[threshold > 0.1].size == 0:
        conv_ctr +=1
    else:
        conv_ctr = 0
    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        self.initialize_training()
        log_likelihood = self.likelihood()
        x = self.image_matrix.flatten()
        means = np.array(self.means)
        variances = np.array(self.variances)
        mixing = np.array(self.mixing_coefficients)
        count = 0
        convergence = False
        while not convergence:
            # E step
            resp = np.empty((self.num_components, x.shape[0]))
            for k in range(0, self.num_components):
                resp[k] = mixing[k]*(1.0/np.sqrt(2.0*variances[k]*math.pi))*np.exp(-1.0*((x - means[k])**2/(2.0*variances[k])))
            resp = resp/np.sum(resp, axis=0)
            resp = resp.T
            # M step
            means = np.sum(resp*x[:, np.newaxis], axis=0)/np.sum(resp, axis=0)
            variances = np.sum(resp*np.square(x[:, np.newaxis] - means), axis=0)/np.sum(resp, axis=0)
            mixing = np.sum(resp, axis=0)/resp.shape[0]
            prev_variables = np.array([
                self.means,
                self.variances,
                self.mixing_coefficients
            ])
            self.means = means
            self.variances = variances
            self.mixing_coefficients = mixing
            new_variables = np.array([
                self.means,
                self.variances,
                self.mixing_coefficients
            ])
            # Check for convergence
            count, convergence = convergence_function(prev_variables, 
                                                      new_variables, 
                                                      count)

def bayes_info_criterion(gmm):
    # TODO: finish this function
    # BIC = -2*L + k*ln(n)
    # L = maximum log likelihood
    # k = number of paramters * 3.0
    # n = sample size
    L = gmm.likelihood()
    k = 3.0*gmm.num_components
    n = gmm.image_matrix.size
    BIC = -2*L + k*math.log(n)
    return BIC

def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel
    """
    # TODO: finish this method
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563, 0.964706]
    ]
    models = []
    BICs = []
    likelihoods = []
    for components in comp_means:
        gmm = GaussianMixtureModel(image_matrix, len(components))
        gmm.initialize_training()
        gmm.means = np.copy(components)
        models.append(gmm)
        BICs.append(bayes_info_criterion(gmm))
        likelihoods.append(gmm.likelihood())
    min_BIC_model = models[BICs.index(min(BICs))]
    max_likelihood_model = models[likelihoods.index(min(likelihoods))]
    return min_BIC_model, max_likelihood_model

def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    min_BIC_model, max_likelihood_model = BIC_likelihood_model_test()
    bic = min_BIC_model.num_components
    likelihood = max_likelihood_model.num_components
    pairs = {
        'BIC' : bic,
        'likelihood' : likelihood 
    }
    return pairs


