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
        # then there will be only one atribute. Meaning polythetic
        # feature values, we'll have binary decision tree for them
        # as left tree container all the the values <theta, and oppsosite
        # for right subtree.
        # For monothetic feature value (e.g. color = Green?), our answers
        # are either equal to or not
        self.theta_val = None
        # It stores, whether attribute is monothetic or not
        self.monothetic = False

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
        """
        self.best_feature = feature

    def add_theta(self, theta_val):
        """
        Add theta values for going to next level tree
        """
        self.theta.append(theta_val)

    def set_monothetic(self):
        """
        Mark tree's feature as monothetic
        """
        self.monothetic = True

    def set_left_subtree(self, tree):
        """
        @type tree: BinaryDecisionTree
        """
        self.left_subtree = tree

    def set_right_subtree(self, tree):
        """
        @type tree: BinaryDecisionTree
        """
        self.right_subtree = tree
        
    def add_label(self, label):
        """
        @type label: any primitive type(str, int, double)
        """
        self.leaf = True
        self.label = label
