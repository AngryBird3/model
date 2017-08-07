#!/usr/bin/python

'''
Created on July 22, 2017
@author: Dhara
'''
import numpy as np
class BinaryFeatureFiltering(object):
    """
    Binary Feature Filtering class for Binary Decision Tree.
    Find feature and theta value to split data into next level.
    Assuming that theta are continuous value, and we've decision like left
    branch values are < theta and right branch values are > theta.
    """
    # Minimum error threshold point at which we'll stop splitting binary tree
    MINIMUM_ERROR_THRESHOLD = 7
    # Minimum record count at each leaf node, at which we we'll stop splitting
    # binary tree
    FEW_RECORDS = 10
    def __init__(self):
        """
        """
        pass

    def get_feature_and_theta_with_least_entropy(self, x, y):
        """
        Find the feature, theta_val for which we have the most information gain
        @type x: ndarray (m x n)
        @type y: ndarray (m x 1)
        @returns (int, int/str) => (feature, theta_val)
        """

        # TODO: Move this to README
        # Pick a feature, with theta val, such that, next level gives the most
        # information gain or least entropy
        #
        # https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain
        #
        # Calculating entropy:
        #          n
        # H(X) = - ∑   p(x_i) log ( p(x_i) )
        #        i = 1
        #        where n is # of outcomes (e.g. x[feature] < theta OR x[feature] > theta)
        #
        # Information gain with feature f:
        # Gain = entropy_before_split - entropy_after_split
        # entropy_after_split = (num_of_records_in_left * entroy_left + num_of_records_in_right * entroy_right)/total dataset before split
        # entroy_left = from H(x)
        #
        # In words, select an attribute and for each value check target attribute
        # value ... so p(yj) is the fraction of patterns at Node N are
        # in category yj - one for true in target value and one one for false.
        #
        # https://stackoverflow.com/questions/14363689/calculating-entropy-in-decision-tree-machine-learning
        #
        # So how do we calculate entropy for continuous feature?
        #
        # I can do various things:
        # A) For binary split: find mean; left branch < mean; right branch > mean
        # B) Consider all the values for given feature; do branch < (>) val. find
        #    the val with minimum split error or max gain
        # C) Consider say 5 split; then sort it; pick 5 different value as theta
        # D) We can use variance (instead of our entropy function). Look at various
        #    values for split, choose value which gives minimum variance
        #
        # We'll go with B.

        # precondition
        if x.shape[0] != len(y):
            raise ValueError("FeatureFiltering: Shape mismatch, Input data rows"
                " doesn't match with label rows")

        # For each feature, consider all the values as lambda and compute entroy
        best_feature, least_err_val = None, None
        for col in range(x.shape[1]):
            val, error = self.get_theta_with_least_entropy(x, col, y)
            if best_feature == None or error < least_err_val:
                best_feature, least_err_val = col, error
        return (best_feature, least_err_val)


    def get_theta_with_least_entropy(self, data, feature, label_vector):
        """
        Find the theta_val for which we have the most information gain
        @type data: ndarray (m x n)
        @type feature: int (Range 0..n)
        @type y: ndarray (m x 1)
        @returns (int, int/str) => (feature, theta_val)
        """
        # precondition
        if feature > data.shape[1]:
            raise ValueError("FeatureFiltering: Shape mismatch, feature to\
                consider is out side of range data (features)")

        feature_values = np.unique(data[:,feature])
        # I'll store split error for each theta/value in given feature vector,
        # I'll need it to return theta with least error
        least_entropy_values = {}
        for theta in feature_values:
            left_data, right_data = self.filter_data(np.append(data, label_vector, axis=1), feature, theta)
            left_subtree_label = left_data[:,-1] # Get the last column (label)
            right_subtree_label = right_data[:,-1] # Get the last column (label)

            err_left = self.get_error(left_subtree_label)
            err_right = self.get_error(right_subtree_label)
            split_error = (float(len(left_subtree_label))  * err_left +
                           float(len(right_subtree_label)) * err_right) / float(len(data))
            least_entropy_values[theta] = split_error
        return  min(least_entropy_values, key=lambda k:least_entropy_values[k]), min(least_entropy_values.values())

    def filter_data(self, data, feature, theta_val):
        """
        Filter Data based on feature and theta_val.
        @type data: ndarray
        @type feature: int
        @type theta_val: int
        @returns tuple(ndarray, ndarray)
        """
        # precondition
        if feature > data.shape[1]:
            raise ValueError("FeatureFiltering: Shape mismatch, feature to\
                consider is out side of range data (features)")

        left_data = data[data[:, feature] < theta_val]
        right_data = data[data[:, feature] > theta_val]

        return (left_data, right_data)

    def get_error(self, y):
        """
        Calculate least min error
        """

        # TODO move this to README
        # As we're predicting house price which is continuous feature, I'm calculating
        # error for given node data as: how far they are from mean (looking at variance)
        #
        # 1/2 ∑ (Yi - MU)**2
        # I can remove 1/2; since its constant

        MU = np.average(y)
        # >>> np.sum(list(map(lambda y: (y - mu)**2, a)))
        # 0.5
        # >>> (3 - 3.5)**2 + (4 - 3.5)**2
        # 0.5
        # >>>
        return np.sum(list(map(lambda y: (y - MU)**2, y)))

    def should_we_stop_filtering(self, y):
        '''
        Tells us whether we should stop splitting into more to avoid both over/under
        fitting
        @type y: ndarray (n x 1)
        @returns boolean
        '''
        # If there are few records remaining, stop
        if len(y) < BinaryFeatureFiltering.FEW_RECORDS or \
            self.get_error(y) < BinaryFeatureFiltering.MINIMUM_ERROR_THRESHOLD:
            return True
        return False

    def assign_label(self, y):
        '''
        Assign label for leaf node having 'y' data
        I decided to just do average of what ended up in leaf node 'y'
        @type y: ndarray (n x 1)
        @returns int
        '''
        return np.average(y)
