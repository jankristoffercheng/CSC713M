import numpy as np

class MultinomialLogisticRegression(object):
    def __init__(self, input_dim, num_classes, std_dev=1e-2):
        self.initialize_weights(input_dim, num_classes, std_dev)

    def initialize_weights(self, input_dim, num_classes, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, C) because each class will have its own set of weights
        b: bias vector; has shape (C,) 
        where C is the number of classes
        
        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        # Check the comment above to know the expected shape of the weights and bias#
        # The weight vector are initialized small random values, while bias gets 0s #
        #############################################################################
        self.params['W'] = np.random.randn(input_dim, num_classes) * std_dev
        #self.params['b'] = np.random.randn(num_classes) - was suggested
        self.params['b'] = np.zeros(num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train logistic regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            # Search up np.random.choice                                            #
            # and numpy array indexing:  https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html#indexing-multi-dimensional-arrays #
            #########################################################################
            indices = np.random.choice(num_train,batch_size)
            X_batch = X[indices,:]
            y_batch = y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            if it % 100 == 0:
                loss_history.append(np.squeeze(loss))

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            self.params['W'] = self.params['W'] + learning_rate * grads['W']
            self.params['b'] = self.params['b'] + learning_rate * grads['b']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def softmax(self,x):
        '''
        Implement the softmax activation function.
        
        Inputs: 
        - x: a numpy array of shape (N,C) denoting the scores of the hypotheses of 
          the classes
        
        Outputs:
        - probs: a numpy array of shape (N, C) containing the probabilities given 
          the scores of each X[i]
          
        Hint: 
        Any np.exp(a) with a large enough a will yield an incredibly large value
        Thankfully, softmax is constant invariant as long as the constant c is added to
        all the scores of that X[i]. 

        Refer to the proof here: 
        https://www.quora.com/Why-is-softmax-invariant-to-constant-offsets-to-the-input
        
        '''
        max = np.reshape(np.amax(x, axis=1), (-1, 1))
        divisor = np.sum(np.exp(x-max), axis=1, keepdims=True)
        probs = np.exp(x-max) / divisor
        return probs


    def cross_entropy(self,probs, labels):
        '''
        Cross entropy for C outcomes. (Can be thought of as a measure of misclassification)

        Inputs:
        - probs: a numpy array of shape (N,C) showing the probabilities of an X[i]
          belonging to any of the C classes; this is the output of self.softmax
        - labels: a numpy array of shape (N,) containing the actual labels (y) of each X[i] 

        Outputs:
        The data loss given the hypotheses (before regularization)
        '''
        
        N = probs.shape[0]
        
        #own version for array
        #array = np.zeros((probs.shape[0], probs.shape[1]))
        #indices = np.arange(probs.shape[0])
        #array[indices,labels] = 1
        
        array = np.eye(probs.shape[1])
        array = array[labels]
        ce = -np.sum(np.multiply(array, np.log(probs))) / N
        return ce
        


    def softmax_cross_entropy_loss(self,x,labels):
        '''
        This is a special function designed to compute the loss (loss), and calculate the 
        gradient the loss (dloss). As the gradient computation also need the individual 
        prob of the actual class (y), both the loss and dloss are computed here with the
        output of self.softmax. 
        
        Note: We can calculate for the gradient separately by calling self.softmax again,
        but note how that will double our computation for the probs (softmax)

        Inputs:
        - x: a numpy array of shape (N,C) denoting the scores of the hypotheses of 
          the classes
        - labels: a numpy array of shape (N,) containing the actual labels (y) of each X[i] 

        Hint: the gradient of the loss only affects probs belonging to the actual label.
        There are two ways how you could get the probs of the actual label:
        #1 Create a (N,C) matrix, where in each row, the actual class will get a 1 and the
           rest will get 0 (hint^2: np.eye can accept an array of which indices the ones 
           will be placed
        #2 A matrix indices can be separated per dimension: following the style of
           matrix[dim_indx_1, dim_indx_2]
           See : http://cs231n.github.io/python-numpy-tutorial/#numpy-array-indexing
           In this case, dim_indx_1 could be an array going through 1..N, and dim_indx_1 
           refer to the columns (label/class) you just need

        Outputs:
        The data loss given the hypotheses (before regularization)
        '''
        
        N = x.shape[0]
        probs = self.softmax(x) # turn scores into probs
        loss = self.cross_entropy(probs, labels)

        dloss = probs.copy()
        #########################################################################
        # TODO: Calculate for the gradients of the loss                         #
        #########################################################################
        
        #own version for array
        #array = np.zeros((probs.shape[0], probs.shape[1]))
        #indices = np.arange(probs.shape[0])
        #array[indices,labels] = 1
        
        array = np.eye(probs.shape[1])
        array = array[labels]
        dloss = array-np.multiply(probs, array)
        
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        
        # the gradient of the loss may already be calculated here
        return loss, dloss

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - reg: Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W, b = self.params['W'], self.params['b']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        score = X.dot(W) + b
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################
        softmax_ce_loss, dloss = self.softmax_cross_entropy_loss(score,y)
        loss = softmax_ce_loss + ((reg/2)* np.sum(np.square(W)))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        dW = np.dot(np.transpose(X), dloss) / N + reg * W

        db = np.mean(dloss, axis=0)
        
        grads['W'] = dW
        grads['b'] = db
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
    
    def predict(self, X, poly_order = 1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - prediction: A sorted numpy array of shape (num_test,num_classes) containing the probabilities for X[i] belonging to each of the classes
        """
        W, b = self.params['W'], self.params['b']
        scores = X.dot(W) + b
        probs = self.softmax(scores)
        prediction = np.argmax(probs, axis=1) # Remember to get the most probable class as the label
        
        return prediction

