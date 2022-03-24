import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from abc import ABC, abstractmethod
import logging


class Node(ABC):
    @abstractmethod
    def predict(self):
        pass

class Impurity(ABC):
    @abstractmethod
    def calculate(self):
        pass

class Gini(Impurity):
    def calculate(self, dataset):
        #JOAN cal vigilar aixo, pero millor on es fa la crida
        if dataset.num_samples==0:
            return 0.0
        pc = dataset.distribution()
        return 1 - np.sum(pc ** 2)
class Entropy(Impurity):
    def calculate(self, dataset):
        #JOAN cal vigilar aixo, pero millor on es fa la crida
        if dataset.num_samples==0:
            return 0.0
        pc = dataset.distribution()
        pc = pc[pc>0]
        suma = np.sum(pc*np.log(pc))
        return -suma

class RandomForestClassifier(Node):
    #JOAN l'ordre dels parametres esta malament
    #def __init__(self, num_trees, min_size, max_depth, ratio_samples, ho teniem així
    #             num_random_features, criterion, optimization):
    def __init__(self, max_depth, min_size, ratio_samples,
        num_trees, num_random_features, criterion, optimization):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        if criterion == "gini":
            self.impurity=Gini()
        elif criterion == "entropy":
                self.impurity=Entropy()
        else:
            assert False, "Non existing method"
        if optimization=="extra-trees":
            self.optimization = optimization
        else:
            self.optimization = None

    def fit(self, X, y):
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = Dataset(X, y)
        self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset):
        self.decision_trees = []
        logging.warning("MAKE DECISION TREES")
        for i in range(self.num_trees):
            logging.warning("number of trees:", self.num_trees)
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            logging.warning("We have created a subset")
            tree = self._make_node(subset, 1)  # the root of the decision tree
            logging.warning("We have created a tree")
            self.decision_trees.append(tree)
            logging.warning("We add the new tree")

    def _make_node(self, dataset, depth):
        logging.warning("MAKE_NODE")
        if depth == self.max_depth or dataset.num_samples <= self.min_size or len(np.unique(dataset.y)) == 1:
            logging.warning("We are creating a leaf")
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
            logging.warning("Leaf created")
        else:
            logging.warning("We are creating a parent")
            node = self._make_parent_or_leaf(dataset, depth)
            logging.warning("Parent created")
        return node

    def _make_leaf(self, dataset):
        # label = most frequent class in dataset
        return Leaf(dataset.most_frequent_label())


    def _make_parent_or_leaf(self, dataset, depth):
        # select a random subset of features, to make trees more diverse
        logging.warning("MAKE PARENT OR LEAF")
        idx_features = np.random.choice(range(dataset.num_features),
                                        self.num_random_features, replace=False)
        logging.warning("Random features created")
        best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)
        logging.warning("We have called best_split")
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            logging.warning("Datasets are 0")
            # this is a special case : dataset has samples of at least two
            # classes but the best split is moving all samples to the left or right
            # dataset and none to the other, so we make a leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            logging.warning("Datasets are NOT 0")
            node = Parent(best_feature_index, best_threshold)
            logging.warning("Create parent")
            node.left_child = self._make_node(left_dataset, depth + 1)
            logging.warning("Create left child")
            node.right_child = self._make_node(right_dataset, depth + 1)
            logging.warning("Create right child")
            return node

    def _best_split(self, idx_features, dataset):
        logging.warning("BEST_SPLIT")
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        if self.optimization=="extra-trees":
            logging.warning("Should not be here, extra-trees")
            pass
        else:
            for idx in idx_features:
                values = np.unique(dataset.X[:, idx])
                for val in values:
                    left_dataset, right_dataset = dataset.split(idx, val)
                    cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                    if cost < minimum_cost:
                        best_feature_index, best_threshold, minimum_cost, \
                        best_split = idx, val, cost, [left_dataset, right_dataset]
            logging.warning("Best split was found")
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset):
        # the J(k,v) equation in the slides, using Gini
        cost = (left_dataset.num_samples * self.impurity.calculate(left_dataset)
                + right_dataset.num_samples * self.impurity.calculate(
            right_dataset)) / \
               (left_dataset.num_samples + right_dataset.num_samples)
        #JOAN atencio : + right_dataset.num_samples + self.impurity.calculate ho teniem així
        return cost

    def predict(self, X):
        ypred = []
        for x in X:
            predictions = [root.predict(x) for root in self.decision_trees]
            # majority voting
            ypred.append(max(set(predictions), key=predictions.count))
        return np.array(ypred)


class Leaf(Node):
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label


class Parent(Node):
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = None
        self.right_child = None

    def predict(self, x):
        if x[self.feature_index] < self.threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)


class Dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]

    def distribution(self):
        pc = np.bincount(self.y)
        p = pc / np.sum(pc)
        return p

    def random_sampling(self, ratio_samples):
        #JOAN cal fer cast a int
        idx=np.random.choice(range(self.num_samples), int(ratio_samples * self.num_samples), replace=True)
        return Dataset(self.X[idx], self.y[idx])

    def most_frequent_label(self):
        distribution = self.distribution()
        return np.argmax(distribution)

    def split(self, idx, val):
        #JOAN intercanvio comparacions per estar d'acord amb predict de Parent
        rightIDX = self.X[:,idx] >= val
        leftIDX = self.X[:,idx] < val

        right_dataset = Dataset(self.X[rightIDX], self.y[rightIDX])
        left_dataset = Dataset(self.X[leftIDX], self.y[leftIDX])

        #JOAN
        #return right_dataset, left_dataset
        return left_dataset, right_dataset


def load_sonar():
    df = pd.read_csv('sonar.all-data.csv',header=None)
    logging.warning("File has loaded correctly")
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    logging.warning("X and y created")
    return X, y

def load_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data, iris.target

logging.getLogger().setLevel(logging.ERROR)

#JOAN
test = 'iris'
criterion = 'entropy'  # 'gini' or 'entropy'
if test=='iris':
    X,y = load_iris()
    print('IRIS') # millor logging
elif test=='sonar':
    X, y = load_sonar()
    print('SONAR') # millor logging
else:
    assert False

logging.warning("We have saved the data")
idx_rocks = y==0
idx_mines = y==1
plt.close('all')
plt.figure(), plt.plot(X[idx_rocks].T,'b'), plt.title('all samples of class rock')
plt.figure(), plt.plot(X[idx_mines].T,'r'), plt.title('all samples of class mine')
ratio_train, ratio_test = 0.7, 0.3
# 70% train, 30% test
num_samples, num_features = X.shape
logging.warning(num_samples)
# 150, 4
idx = np.random.permutation(range(num_samples))
# shuffle  {0,1, ... 149} because samples come sorted by class!
num_samples_train = int(num_samples*ratio_train)

num_samples_test = int(num_samples*ratio_test)
idx_train = idx[:num_samples_train]
idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]
max_depth = 10      # maximum number of levels of a decision tree
min_size_split = 5  # if less, do not split a node
ratio_samples = 0.7 # sampling with replacement
num_trees = 20     # number of decision trees
num_random_features = int(np.sqrt(num_features))
                    # number of features to consider at
                    # each node when looking for the best split

logging.warning("We have chosen:",  criterion)
rf = RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                            num_trees, num_random_features, criterion, "hola")
logging.warning("Random Forest created")
# train = make the decision trees
rf.fit(X_train, y_train)
logging.warning("We have created fit")
# classification
ypred = rf.predict(X_test)
logging.warning("Y predictions")
# compute accuracy
num_samples_test = len(y_test)
num_correct_predictions = np.sum(ypred == y_test)
accuracy = num_correct_predictions/float(num_samples_test)
print('accuracy {} %'.format(100*np.round(accuracy,decimals=2)))


