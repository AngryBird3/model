#!/usr/bin/python

'''
Created on July 12, 2017
@author: Dhara
'''
import BinaryDecisionTree

class DecisionTreeClassifier(object):
    def __init__(self):
        self.binary_decision_tree = None

    def train(self, x, y):
        """
        @type x: ndarray (training data)
        @type y: ndarray (training data label)
        """
        self.binary_decision_tree = create_decision_tree(x, y)

    def create_decision_tree(self, x, y, tree=BinaryDecisionTree()):
        """
        @type tree: BinaryDecisionTree
        @type x: ndarray (training data)
        @type y: ndarray (training data label)
        @return (trained) BinaryDecisionTree
        """

        '''
        TODO : move this to README
                    root
            left            right
           /    \
          /      \

        At root, I've data with SOME entropy (information gain) I want to split it
        into left and right SUBTREE with BEST FEATURE(f) and BEST FEATURE VALUE(theta)
        such that left has all the data for which data[f] <= theta and right has all
        the data for which data[f] >= theta. So my tree structure could look like:
        {FEATURE : {"THETA, <" : left_subtree, "THETA', >": right_subtree}}

        Inorder to find BEST FEATURE(f) and BEST FEATURE VALUE(theta) => Assume we've
        function which returns both f, theta = find_best_feature_theta(x, y)

        This looks recursive, but we need to stop sometime: what could be the criteria?
        Let's make a method which does that: should_we_stop(tree, y)
        '''
        # Terminate condition
        if should_we_stop(tree, y):
            tree.add_label(self.find_label(y))
            return tree

        # Find the feature,theta for splitting into subtrees
        is_discrete, best_feature, theta_val = self.get_feature_and_theta_with_least_entropy(x, y)

        tree.set_monothetic()
        tree.set_best_feature(best_feature)
        tree.add_theta(theta_val)

        # Now, I need to devide current node data into left and right subtrees
        # based on best_feature and theta_val
        left_data, right_data = self.filter_data(np.append(x, y, axis=1), is_discrete, best_feature, theta_val)

        left_subtree_label = left_data[:,-1] # Get the last column (label)
        right_subtree_label = right_data[:,-1] # Get the last column (label)
        left_subtree_data = np.delete(left_data, -1, axis=-1) # Removing last column (label)
        right_subtree_data = np.delete(right_data, -1, axis=-1) # Removing last column (label)

        left_subtree = DecisionTreeClassifier(left_subtree_data, left_subtree_label)
        right_subtree = DecisionTreeClassifier(right_subtree_data, right_subtree_label)

        tree.set_left_subtree(left_subtree)
        tree.set_right_subtree(right_subtree)

        return tree

    def filter_data(self, data, is_discrete, best_feature, theta_val):
        """
        Filter Data based on best_feature and theta_val. It looks for
        monothetic flag to either compare '<' theta_val or '=' theta_val
        @type data: ndarray
        @type is_discrete: Boolean
        @type best_feature: int
        @type theta_val: int
        @returns tuple(ndarray, ndarray)
        """

        '''
        TODO: Move this to README
        filter by data with:
        if is_discrete:
            left_data = data[best_feature] where data[best_feature] < theta_val
            right_data = data[best_feature] where data[best_feature] > theta_val
        else:
            left_data = data[best_feature] where data[best_feature] = theta_val
            right_data = data[best_feature] where data[best_feature] != theta_val
        '''
        if is_discrete:
            left_data = data[data[:, best_feature] < theta_val]
            right_data = data[data[:, best_feature] > theta_val]
        else:
            left_data = data[data[:, best_feature] == theta_val]
            right_data = data[data[:, best_feature] != theta_val]

        return (left_data, right_data)

    def get_feature_and_theta_with_least_entropy(self, x, y):
        """
        Find the feature, theta_val for which we have the most information gain
        If the x[feature] is discrete, it will return Boolean = True
        @type x: ndarray
        @type y: ndarray
        @returns (Boolean, int, int/str) => (is_discrete, feature, theta_val)
        """

        '''
        TODO: Move this to README
        Pick a feature, with theta val, such that, next level gives the most
        information gain or least entropy

        https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain

        Calculating entropy:
                 n
        H(X) = - ∑   p(x_i) log ( p(x_i) )
               i = 1
               where n is # of outcomes (e.g. x[feature] < theta OR x[feature] > theta)

        Information gain with feature f:
        Gain = entropy_before_split - entropy_after_split
        entropy_after_split = (num_of_records_in_left * entroy_left + num_of_records_in_right * entroy_right)/total dataset before split
        entroy_left = from H(x)

        In words, select an attribute and for each value check target attribute
        value ... so p(yj) is the fraction of patterns at Node N are
        in category yj - one for true in target value and one one for false.

        https://stackoverflow.com/questions/14363689/calculating-entropy-in-decision-tree-machine-learning

        So how do we calculate entropy for continuous feature?
        '''


    def
