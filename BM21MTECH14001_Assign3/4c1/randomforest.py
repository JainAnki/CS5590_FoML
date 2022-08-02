
from __future__ import division
import numpy as np
from scipy.stats import mode
from utilities import shuffle_in_unison
from decisiontree import DecisionTreeClassifier



class RandomForestClassifier(object):
    """ A random forest classifier.

    A random forest is a collection of decision trees that vote on a
    classification decision. Each tree is trained with a subset of the data and
    features.
    """

    def __init__(self, n_estimators=20, max_features=np.sqrt, max_depth=25,
        min_samples_split=20, bootstrap=0.9):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            max_features: Controls the number of features to randomly consider
                at each split.
            max_depth: The maximum number of levels that the tree can grow
                downwards before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
            bootstrap: The fraction of randomly choosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []
        self.out_of_bag_models = {}
        self.X_oob = 0
        self.y_oob = 0

    def fit(self, X, y):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples*self.bootstrap)
        print(self.max_features)
        for i in range(self.n_estimators):
            print(i)
            shuffle_in_unison(X, y)
            X_subset = X[:n_sub_samples]
            y_subset = y[:n_sub_samples]
            self.X_oob = X
            self.y_oob = y
            tree = DecisionTreeClassifier(self.max_features, self.max_depth,
                                            self.min_samples_split)
            tree.fit(X_subset, y_subset)   
            print(tree)
            # track which data point this model wasn't trained on
            for i in (set(range(n_samples)) - set(range(len(X_subset)))):
                      self.out_of_bag_models.setdefault(i, []).append(tree)
            print(self.out_of_bag_models)
            self.forest.append(tree)

    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)

        return mode(predictions)[0][0]


    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/n_samples
        return accuracy
    
    def oob_score(self):
        # out of bag error; validation
        size_s = len(self.out_of_bag_models.items())
        bag_error_sum = sum([self.g_bar_error(models, self.X_oob[i], self.y_oob[i]) 
                             for i, models in self.out_of_bag_models.items()])
        out_of_bag_error = bag_error_sum / size_s
        return out_of_bag_error
    
    def g_bar_error(self,models, x, y):
        y_hat = []
        for model in models:
            print(model)
            y_hat.append(model.predict(np.reshape(x, (1, x.shape[0])))[0])
        return int(mode(y_hat, axis=None)[0][0] != y)
        

