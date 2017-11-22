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
                indices = [i for i, j in enumerate(y) if j == c]
                self.prob_x_given_y[:,c] = (np.sum(X[indices],axis=0)+1) / (np.sum(X[indices])+D)
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
                indices = [i for i, j in enumerate(y) if j == c]
                self.mu[:,c] = np.mean(X[indices,:], axis=0)
                self.sigma[:,c] = np.var(X[indices,:], axis=0)

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
        prior = []

        for c in range(len(self.classes)):
            # implement
            prior.append(len([i for i, j in enumerate(y) if j == c])/num_examples)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prior


    def compute_gaussian_probability(self, X, mean, var):
        coeff = 1.0 / np.sqrt((2.0 * np.pi) * var)
        exponent = np.exp(-0.5*(np.expand_dims(X,-1) - mean)**2 / var)

        return coeff * exponent

    def compute_log_likelihood(self, X):
    	
        if self.distribution == "multinomial":
            #############################################################################
            # TODO: Calculate the log likelihood for multinomial distributions          #
            ############################################################################# 
            return np.dot(X,np.log(self.prob_x_given_y))
            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################
        elif self.distribution == "gaussian":
            #############################################################################
            # TODO: Calculate the log likelihood for Gaussian distributions             #
            ############################################################################# 
            loli = np.sum(np.log(self.compute_gaussian_probability(X, self.mu, self.sigma)),axis=1)
            #same shit pero prod: loli = -np.log(np.prod(self.compute_gaussian_probability(X, self.mu, self.sigma),axis=1))
            
            
            return loli
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
        posterior = self.compute_log_likelihood(X[:]) + np.log(self.prior)
        print(self.prior)
        prediction = np.argmax(posterior,axis=1)
        print(prediction.shape)

        return prediction

