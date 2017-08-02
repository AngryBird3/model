#!/usr/bin/python

'''
Created on July 12, 2017
@author: Dhara
'''
from BinaryFeatureFiltering import BinaryFeatureFiltering
from BinaryDecisionTree import BinaryDecisionTree
class DecisionTreeClassifier(object):
    """
    Main functions for classifying using (binary) decision tree
    """
    def __init__(self, feature_selector = BinaryFeatureFiltering()):
        self.binary_decision_tree = None
        # Letting user implement different feature selctor to select feature
        # and value for each split; please see FeatureSelector class for more info
        self.feature_selector = feature_selector

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

        # TODO : move this to README
        #             root
        #     left            right
        #    /    \
        #   /      \
        #
        # At root, I've data with SOME entropy (information gain) I want to split it
        # into left and right SUBTREE with BEST FEATURE(f) and BEST FEATURE VALUE(theta)
        # such that left has all the data for which data[f] <= theta and right has all
        # the data for which data[f] >= theta. So my tree structure could look like:
        # {FEATURE : {"THETA, <" : left_subtree, "THETA', >": right_subtree}}
        #
        # Inorder to find BEST FEATURE(f) and BEST FEATURE VALUE(theta) => Assume we've
        # function which returns both f, theta = find_best_feature_theta(x, y)
        #
        # This looks recursive, but we need to stop sometime: what could be the criteria?
        # Let's make a method which does that: should_we_stop(tree, y)


        # Terminate condition
        if feature_selector.should_we_stop_filtering(y):
            tree.add_label(self.find_label(y))
            return tree

        # Find the feature,theta for splitting into subtrees
        best_feature, theta_val = feature_selector.get_feature_and_theta_with_least_entropy(x, y)

        tree.set_best_feature(best_feature)
        tree.add_theta(theta_val)

        # Now, I need to devide current node data into left and right subtrees
        # based on best_feature and theta_val
        left_data, right_data = self.filter_data(np.append(x, y, axis=1), best_feature, theta_val)

        left_subtree_label = left_data[:,-1] # Get the last column (label)
        right_subtree_label = right_data[:,-1] # Get the last column (label)
        left_subtree_data = np.delete(left_data, -1, axis=-1) # Removing last column (label)
        right_subtree_data = np.delete(right_data, -1, axis=-1) # Removing last column (label)

        left_subtree = DecisionTreeClassifier(left_subtree_data, left_subtree_label)
        right_subtree = DecisionTreeClassifier(right_subtree_data, right_subtree_label)

        tree.set_left_subtree(left_subtree)
        tree.set_right_subtree(right_subtree)

        return tree

    def predict(self, x, root="BEGIN"):
        """
        @type x: ndarray (m x n)
        @return int (prediction)
        """
        if root == "BEGIN":
            root = self.binary_decision_tree
        # TODO: Move this to README
        # My trained tree looks like:
        #
        #              ROOT
        #              ____
        #             |    |
        #             |____| (feature=f, theta=theta)
        #
        #
        # LEFT                       RIGHT
        #
        # I'll look at x[f]
        # if x[f] < theta:
        #     predict(x, root.left)
        # else:
        #     predict(x, root.right)
        #
        # Somehting feels odd, about this, may be create_decision_tree and predict
        # can be in BinaryDecisionTree: I think No - cause that's classifier,
        # BinaryDecisionTree is a type
        if root.is_leaf():
            return root.get_label()
        if x[root.get_best_feature()] < root.get_theta():
            return self.predict(x, root.get_left_subtree())
        else:
            return self.predict(x, root.get_right_subtree())
