import numpy as np
import matplotlib.pyplot as plt

class TreeNode():
    def __init__(self, node_type="Leaf", feature=None, threshold=None,
                 value=None, left_branch=None, right_branch=None):
        """
        This class represents a decision node or leaf node in the decision tree.
        A decision node would store the feature index, the threshold value, and 
        a reference to the left and right subtrees. A leaf node only needs to store
        a value.

        Inputs:
        - node_type: (str) Keeps track of the type of note (Decision or Leaf).
        - feature: (int) Index of the feature that is currently being split
        - threshold: (int / float) The threshold value where the split will be made.
        - value: (int / float) Stores the value of a leaf node. 
        - left_branch: (TreeNode) A reference to the left subtree after the split.
        - right_branch: (TreeNode) A reference to the right subtree after the split.
        """
        self.node_type = node_type
        self.feature = feature          
        self.threshold = threshold         
        self.value = value                
        self.left_branch = left_branch      
        self.right_branch = right_branch   
        
class DecisionTree(object):
    def __init__(self, min_samples_split=2, max_depth=np.inf, min_impurity=1e-7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity = min_impurity
    
    def train(self, X, y):
        """
        Builds the decision tree from the training set.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        """

        # Simple check if a feature / dimension is boolean or not
        self.is_bool_feature = np.all(X == X.astype(bool), axis=0)
        # build the tree and remember the root node
        self.root_node = self.create_tree(X, y)

    def create_tree(self,X, y, depth=0):
        """
        Top-down approach for building decision trees (recursive definition).

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        - depth: (int) Keeps track of the current depth of the tree

        Output:
        A reference to the current node.
        """
        N, D = X.shape
        
        #########################################################################
        # TODO: Build the tree recursively.                                     #
        #########################################################################
        if(np.unique(y).shape[0] == 1):
            return TreeNode(self, value=self.compute_leaf(y))
        else:
            max_impurity, best_feature, best_threshold, split_idx = self.choose_best_feature_split(X,y)
            left_branch = self.create_tree(X[split_idx['left_idx']], y[split_idx['left_idx']], depth+1)
            right_branch = self.create_tree(X[split_idx['right_idx']], y[split_idx['right_idx']], depth+1)
            return TreeNode(node_type="Decision", feature=best_feature, threshold=best_threshold, left_branch=left_branch, right_branch=right_branch)
        #########################################################################
        #                              END OF YOUR CODE                         #
        #########################################################################  

        


    def choose_best_feature_split(self, X, y):
        """
        Iterates through all the possible splits and choose the best one according to
        some impurity criteria.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.

        Output:
        - max_impurity: (float) The maximum impurity gain out of all the splits.
        - best_feature: (int) The index of the feature with maximum impurity gain.
        - best_threshold: (int / float) The value of the best split for the feature.
        - split_idx: (dict)
            - "left_idx": stores the indices of the data in the left subtree after the split
            - "right_idx": stores the indices of the data in the right subtree after the split
        """
        N, D = X.shape
        
        max_impurity = 0
        best_feature = None
        best_threshold = None
        split_idx = {}

        #########################################################################
        # TODO: Choose the best feature to split on.                            #
        #########################################################################
        for i in range(D):
            unique = np.unique(X[i])
            for j in range(unique.shape[0]):
                if(self.is_bool_feature[i]):
                    left_idx = [X[:,i] == j]
                    right_idx = [X[:,i] != j]
                else:
                    left_idx = [X[:,i] < j]
                    right_idx = [X[:,i] >= j]
                gain = self.information_gain(X[:,i], X[left_idx,i], X[right_idx,i])
                if(best_feature == None or gain > max_impurity):
                    best_feature = i
                    best_threshold = j
                    max_impurity = gain
                    split_idx['left_idx'] = left_idx
                    split_idx['right_idx'] = right_idx
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return max_impurity, best_feature, best_threshold, split_idx
    
    def traverse_tree(self, X, node):
        """
        Traverse the decision tree to determine the predicted value for a given input.

        Inputs:
        - X: A numpy array of shape (D,) containing training data; The test samples are 
            evaluated one at a time.
        - node: (TreeNode) Current node being evaluated.

        Output:
        Returns the value of the leaf node following the decisions made.
        """

        #########################################################################
        # TODO: Traverse the tree following path based on the decisions that    #
        # made based on the input data.                                         #
        #########################################################################
        if node.node_type == "Leaf":
            return node.value

        X_feature_val = X[node.feature]
        
        if self.is_bool_feature[node.feature]:
            if X_feature_val == node.threshold:
                branch = node.right_branch
            else:
                branch = node.left_branch
        else:
            if X_feature_val >= node.threshold:
                branch = node.right_branch
            else:
                branch = node.left_branch

        return self.traverse_tree(X, branch)
        #########################################################################
        #                              END OF YOUR CODE                         #
        #########################################################################         
    
    def predict(self, X):
        """
        Iterates through each example and traverse the decision tree 
        to get the predicted value.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data

        Output:
        Returns the predicted value following the decisions made in the tree.
        """

        N, D = X.shape
        prediction = []
        
        for i in range(N):
            pred = self.traverse_tree(X[i], node=self.root_node)
            prediction.append(pred)

        return prediction

    
    def compute_leaf(self, y):
        raise NotImplementedError("Should be implemented in the subclass")

    def compute_impurity(self, y_S, y_A, y_B):
        raise NotImplementedError("Should be implemented in the subclass")

    def visualize_tree(self):
        """
        Code for visualizing the tree. This is not part of the exercise but feel free
        to look at the code.
        """
        def traverse_tree_viz(node, parent_pt, center_pt, depth=0):

            if node.node_type == "Decision":
                feature_name = node.feature
                if self.is_bool_feature[node.feature]:
                    text = "F {} == {}".format(feature_name, node.threshold)
                else:
                    text = "F {} >= {}".format(feature_name, node.threshold)
                    
                r = 0.5 - depth * 0.05
                theta = np.pi * 13 / 12 + depth * np.pi / 15
                traverse_tree_viz(node.left_branch, center_pt, (r*np.cos(theta) + center_pt[0], r* np.sin(theta) + center_pt[1]), depth+1)
                theta = - np.pi / 12 - depth * np.pi / 15
                traverse_tree_viz(node.right_branch, center_pt, (r*np.cos(theta) + center_pt[0], r* np.sin(theta) + center_pt[1]), depth+1)

                if parent_pt is None:
                    plotNode(plt_axis, text, center_pt, None, node.node_type)
                else:
                    plotNode(plt_axis, text, center_pt, parent_pt, node.node_type)
                    
            elif node.node_type == "Leaf":
                text = node.value
                plotNode(plt_axis, text, center_pt, parent_pt, node.node_type)
                
        
        def plotNode(axis, text, center_point, parent_point, node_type):
            decNode = dict(boxstyle="round, pad=0.5", fc='0.8')
            leafNode = dict(boxstyle="round, pad=0.5", fc="0.8")
            arrow_args = dict(arrowstyle="<|-,head_length=0.5,head_width=0.5", edgecolor='black',lw=3, facecolor="black")

            if node_type == "Leaf":
                boxstyle = leafNode
            else:
                boxstyle = decNode

            if parent_point is None:
                axis.text(0.5, 1, text, va="center", ha="center", size=15, bbox=decNode)
            else:
                axis.annotate(text, xy=parent_point, xytext=center_point, 
                    va="center", ha="center", bbox=boxstyle, arrowprops=arrow_args, size=15)

        
        
        plt.figure(figsize=(13,15))
        plt_axis = plt.subplot(111, frameon=False)
        traverse_tree_viz(self.root_node, None, (0.5,1))
        







            
                
    
