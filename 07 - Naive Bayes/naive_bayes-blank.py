import numpy as np

class NaiveBayes(object):
    def __init__(self, distribution="multinomial"):
        self.distribution = distribution

    def train(self, X, y):
        self.classes = np.unique(y)
        self.prior = self.compute_prior(y)
  
        C = len(self.classes) # number of classes
        D = X.shape[1] # number of parameters

        if self.distribution == "multinomial":
            self.prob_x_given_y = np.zeros((D,C))
            for c in range(C):
                ###########################################################################
                # TODO: Calculate for theta using MLE for multinomial                     #
                ###########################################################################
                self.prob_x_given_y[:,c] = None
                ###########################################################################
                #                              END OF YOUR CODE                           #  
                ###########################################################################
            
        elif self.distribution == "gaussian": 
            self.mu = np.zeros((D,C))
            self.sigma = np.zeros((D,C))
            for c in range(C):
                ###########################################################################
                # TODO: Calculate for mean and variance using MLE for Gaussian            #
                ###########################################################################
                self.mu[:,c] = None
                self.sigma[:,c] = None

                ###########################################################################
                #                              END OF YOUR CODE                           #  
                ########################################################################### 
        
    def compute_prior(self, y):
        '''
        Calculate for the prior of each class y based on their counts.
        prior here is just the counts 

        Return:
        prior : an array of shape (C,) which should contain the priors of each class

        '''
        
        num_examples = len(y) # total number of instances

        #############################################################################
        # TODO: calculate for the prior per class                                   #
        #       prior per class is just the counts of per class / total             #
        ############################################################################# 
        prior = None

        for c in range(len(self.classes)):
            # implement

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prior


    def compute_gaussian_probabilitiy(self, X, mean, var):
        coeff = 1.0 / np.sqrt((2.0 * np.pi) * var)
        exponent = np.exp(-0.5*(np.expand_dims(X,-1) - mean)**2 / var)

        return coeff * exponent

    def compute_log_likelihood(self, X):
    	
        if self.distribution == "multinomial":
            #############################################################################
            # TODO: Calculate the log likelihood for multinomial distributions          #
            ############################################################################# 
            return None
            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################
        elif self.distribution == "gaussian":
            #############################################################################
            # TODO: Calculate the log likelihood for Gaussian distributions             #
            ############################################################################# 
            return None
            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################

    
    def predict(self, X):
        '''
        Calculate for the posterior by adding the log-likelihood and the log-prior
        To classify, we do not need to calculate for the evidence (denominator).
        If we wanted to get the actual probabilities, we need to calculate for the evidence.

        Return 
        prediction : An array of shape (len(X),) containing the class index with the highest 
                     posterior probability (array of shape (len(X),))
        '''
        posterior = None
        prediction = None

        return prediction

