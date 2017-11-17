import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_size, num_classes, hidden_activation_fn = 'relu', std_dev=1e-4):
        self.hidden_activation_fn = hidden_activation_fn
        self.initialize_weights(input_dim, hidden_size, num_classes, std_dev)

    def initialize_weights(self, input_dim, hidden_size, num_classes, std_dev):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - hidden_size: (int) The number of neurons H in the hidden layer.
        - num_classes: (int) The number of classes C.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        #       See comment abovet to know the initialization                       #
        #############################################################################
        self.params['W1'] = None
        self.params['b1'] = None
        self.params['W2'] = None
        self.params['b2'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        


    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, beta1=0.9, beta2=0.999,
            optimizer="sgd", epsilon = 1e-8, batch_size=200, verbose=False):
        """
        Train this neural network using the specified optimizer.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - beta1: (float) momentum parameter
        - beta2: (float) RMSprop parameter
        - optimizer: (str) specifies which optimizer to use. Can be sgd, momentum, rmsprop, or adam
        - epsilon: (float) small number added to prevent division by zero.
        - batch_size: (integer) number of training examples to use at each iteration.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train,batch_size)
            X_batch = X[indices,:]
            y_batch = y[indices]
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            if it % 500 == 0:
                learning_rate = learning_rate*0.95
            if it % 100 == 0:
                loss_history.append(np.squeeze(loss))


            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using the specified optimizer. You'll need to use the gradients       #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            if optimizer == "sgd":
                ###########################################################################
                # Update the weights using Stochastic Gradient descent.                   #
                # Note : you now have to update the weights and the bias in the 1st layer #
                #                           and the weights and the bias in the 2nd layer # 
                ###########################################################################
                pass
                ###########################################################################
                #                              END OF YOUR CODE                           #
                ###########################################################################        

            elif optimizer == "momentum":
                if not hasattr(self,'velocity'):
                    self.velocity = {}
                    self.velocity['W1'] = 0
                    self.velocity['b1'] = 0
                    self.velocity['W2'] = 0
                    self.velocity['b2'] = 0

                ############################################################################
                # Update the weights using Stochastic Gradient descent with Momentum       #
                #   Don't forget to update self.velocity to the current iteration          #
                #   Do not assign the bias correction (div by (1-beta)^t) to self.velocity #
                #       This correction is just for processing the current iteration       #
                ############################################################################
                pass
                ############################################################################
                #                              END OF YOUR CODE                            #
                ############################################################################  
            elif optimizer == "rmsprop":
                if not hasattr(self,'rmsgrad'):
                    self.rmsgrad = {}
                    self.rmsgrad['W1'] = 0
                    self.rmsgrad['b1'] = 0
                    self.rmsgrad['W2'] = 0
                    self.rmsgrad['b2'] = 0
                    self.t = 1


                ############################################################################
                # Update the weights using RMSProp. Apply bias correction.                 #
                #   Do not assign the bias correction (div by (1-beta)^T) to self.rmsgrad  #
                #       This correction is just for processing the current iteration       #
                ############################################################################
                pass
                ############################################################################
                #                              END OF YOUR CODE                            #
                ############################################################################  
            
                self.t += 1

            elif optimizer == "adam":
                if not hasattr(self,'rmsgrad'):
                    self.rmsgrad = {}
                    self.rmsgrad['W1'] = 0
                    self.rmsgrad['b1'] = 0
                    self.rmsgrad['W2'] = 0
                    self.rmsgrad['b2'] = 0

                if not hasattr(self,'velocity'):
                    self.velocity = {}
                    self.velocity['W1'] = 0
                    self.velocity['b1'] = 0
                    self.velocity['W2'] = 0
                    self.velocity['b2'] = 0
                    self.t = 1
                
                #########################################################################
                # Update the weights using RMSProp. Apply bias correction on both       #
                # velocity and rmsgrad.                                                 #
                #########################################################################
                pass
                #########################################################################
                #                              END OF YOUR CODE                         #
                ######################################################################### 
                
                self.t += 1
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def softmax(self,x):
        """
        Softmax activation function.

        Inputs:
        - x: A numpy array of shape (N, C) containing the raw scores for each class c.

        Outputs:
        Probability of belonging to each class
        """

        probs =  None
        return probs


    def cross_entropy(self,probs, labels):
        """
        Cross entropy

        Inputs:
        - probs: A numpy array of shape (N, C) containing the probabilities for each class c.
        - labels: A numpy array of shape (N,) containing the ground truth class
        Outputs:
        Cross entropy loss
        """
        cross_entropy = None
        return cross_entropy


    def softmax_cross_entropy_loss(self,x,labels):
        """
        Softmax cross entropy loss. (combines softmax and cross entropy)

        Inputs:
        - x: A numpy array of shape (N, C) containing the raw scores for each class c.
        - labels: A numpy array of shape (N,) containing the ground truth class

        Returns:
        - loss : (float) Softmax cross entropy loss
        - dloss : the gradient of the loss with respect to the input x
        """

        return loss, dloss

    def sigmoid(self,x):
    	return 1 / (1 + np.exp(-x))

    def tanh(self,x):
        return np.tanh(x)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################

        if self.hidden_activation_fn == "relu":
            pass
        elif self.hidden_activation_fn == "sigmoid":
            pass
        elif self.hidden_activation_fn == "tanh":
            pass

        scores = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # If the target labels are not given then we return the raw scores
        if y is None:
          return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Finish the implementations#
        # of the softmax, cross_entropy, and softmax_cross_entropy_loss.            #
        #############################################################################
        loss = None
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W1'] should store     #
        # the gradient on W1, and be a matrix of same size.                          #
        #############################################################################
        
        if self.hidden_activation_fn == "relu":
            pass
        elif self.hidden_activation_fn == "sigmoid":
            pass
        elif self.hidden_activation_fn == "tanh":
            pass
        
        grads['W2'] = None
        grads['b2'] = None
        grads['W1'] = None
        grads['b1'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
    
    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        #############################################################################
        if self.hidden_activation_fn == "relu":
            pass
        elif self.hidden_activation_fn == "sigmoid":
            pass
        elif self.hidden_activation_fn == "tanh":
            pass
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        y_pred = None
        
        return y_pred


        
    def train_step(self,X, y, learning_rate=1e-3, reg=1e-5, batch_size=200):
        """
        Performs one iteration of gradient descent. Only used for the training animation.
        """
        num_train, dim = X.shape
        indices = np.random.choice(num_train,batch_size)
        X_batch = X[indices,:]
        y_batch = y[indices]

        loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

        self.params['W1'] += -learning_rate * grads['W1']
        self.params['b1'] += -learning_rate * grads['b1']
        self.params['W2'] += -learning_rate * grads['W2']
        self.params['b2'] += -learning_rate * grads['b2']
