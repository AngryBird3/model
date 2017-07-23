#!/usr/bin/python

'''
Created on July 12, 2017
@author: Dhara
'''
class BinaryDecisionTree(object):
    """
    As name suggests, its a binary decision tree,
    we could easily extend it to write non-binary
    one as well. We need to add more theta values
    """
    def __init__(self):
        # Boolean to store whether this is a leaf node
        self.leaf = False
        # If its leaf node, we need to assign a label
        self.label = None

        # Next level trees (if any) are split based on this features
        self.best_feature = None
        # Theta for next level, if the attributes are real values
        # then there will be only one atribute. Meaning discrete
        # feature values, we'll have binary decision tree for them
        # as left tree container all the the values <theta, and oppsosite
        # for right subtree.
        # For discrete feature value (e.g. color = Green?), our answers
        # are either equal to or not, or more classes e.g. which color
        # but as our data is continuous, I'm focusing on just that
        self.theta_val = None
        # It stores, whether attribute is discrete or not
        self.discrete = False

        # Well, yeah again, I could use a list to store all
        # child nodes, but for binary tree, just left and right
        # are suffice
        # Left one will contain all the values <theta (polythetic feature values)
        # (or =theta, if len(self.theta) > 1 ==> monothetic feature value)
        # and right one will contain all the values >theta
        self.left_subtree = None
        self.right_subtree = None

    def is_leaf(self):
        """
        Whether this is a leaf node
        """
        return self.leaf

    def set_best_feature(self, feature):
        """
        Add best feature to split for next level, for most information gain or least entropy
        @type feature: int (Its column number in our ndarray)
        """
        self.best_feature = feature

    def get_best_feature(self):
        """
        Get best feature to split for next level, for most information gain or least entropy
        @return int (Its column number in our ndarray)
        """
        return self.best_feature

    def add_theta(self, theta_val):
        """
        Add theta values for going to next level tree
        """
        self.theta= theta_val

    def get_theta(self):
        """
        Get theta values for going to next level tree
        @return int (In our case data is continuous)
        """
        return self.theta[0]

    def set_discrete(self):
        """
        Mark tree's feature as discrete
        """
        self.discrete = True

    def set_left_subtree(self, tree):
        """
        @type tree: BinaryDecisionTree
        """
        self.left_subtree = tree

    def get_left_subtree(self):
        """
        @return tree: BinaryDecisionTree
        """
        return self.left_subtree

    def set_right_subtree(self, tree):
        """
        @type tree: BinaryDecisionTree
        """
        self.right_subtree = tree

    def get_right_subtree(self):
        """
        @return tree: BinaryDecisionTree
        """
        return self.right_subtree

    def add_label(self, label):
        """
        @type label: any primitive type(str, int, double)
        """
        self.leaf = True
        self.label = label

    def get_label(self):
        """
        @return any primitive type(str, int, double)
        """
        return self.label
