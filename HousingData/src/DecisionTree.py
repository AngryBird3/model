#!/usr/bin/python

'''
Created on July 12, 2017
@author: Dhara
'''
class DecisionTree(object):
    def __init__(self):
        pass

    def train(self, x, y):
        """
        @type x: ndarray (training data)
        @type y: ndarray (training data label)
        """
        pass

    def create_decision_tree(self, tree={}, x, y):
        """
        @type tree: dictionary (Keys as tuple (feature value, split value), Value as tree)
        @type x: ndarray (training data)
        @type y: ndarray (training data label)
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
        the data for which data[f] >= theta

        Inorder to find BEST FEATURE(f) and BEST FEATURE VALUE(theta) => Assume we've
        function which returns both f, theta = find_best_feature_theta(x, y)
        '''
