from pprint import pprint
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
import functions
from joblib import Parallel, delayed


def train_reg(reg, clazz, X, X_val, two_class_y, two_class_y_val):
    print 'Training clazz', clazz, 'with C=', reg
    model = LogisticRegression('l1', False, C=reg)
    model.fit(X, two_class_y)
    precision = functions.precision(model.predict_proba(X_val), two_class_y_val)
    return model, precision

def to_two_class(clazz, y):
    y = y.copy()
    y[y!=clazz] = -1
    y[y==clazz] = 0
    y[y==-1] = 1
    return y


def train_single_lr(X, X_val, clazz, reg_grid, y, y_val):
    two_class_y = to_two_class(clazz, y)
    two_class_y_val = to_two_class(clazz, y_val)
    if len(set(two_class_y)) < 2:
        raise Exception
    if len(set(two_class_y_val)) < 2:
        raise Exception
    max_precision = 0
    max_reg = -1
    models = []
    precisions = []

    #        models_precisions = Parallel(n_jobs=1, pre_dispatch=4, verbose=11)(
    #            delayed(train_reg)(reg, clazz, X, X_val, two_class_y, two_class_y_val) for reg in reg_grid)

    models_precisions = [train_reg(reg, clazz, X, X_val, two_class_y, two_class_y_val) for reg in reg_grid]

    models_precisions = zip(*models_precisions)
    models, precisions = models_precisions
    maxindex = np.argmax(precisions)
    max_reg = reg_grid[maxindex]
    max_model = models[maxindex]
    return max_model, max_reg



class MultiLogistic():

    def __init__(self, penalty = 'l1'):
        self.penalty = penalty
        self.models = {}
        self.reg_choices = {}




    def train_cross_validation(self, X, X_val, y, y_val, reg_grid):

        classes = sorted(set(y))
        models_regs = Parallel(n_jobs=-1, verbose=11)(
            delayed(train_single_lr)(X, X_val, clazz, reg_grid, y, y_val) for clazz in classes)

        clazz = 0
        for max_model, max_reg in models_regs:
            self.models[clazz] = max_model
            self.reg_choices[clazz] = max_reg
            clazz+=1

        print 'reg_choices:'
        pprint(self.reg_choices)
        print 'end reg_choices'

    def train_after_cv(self, X, y):
        for clazz in sorted(set(y)):
            two_class_y = to_two_class(clazz, y)
            model = self.models[clazz]
            model.fit(X, two_class_y)


    def predict_proba(self, X):
        proba = []
        for clazz in sorted(self.models.keys()):
            model = self.models[clazz]
            proba.append(model.predict_proba(X)[:,0])
        return np.vstack(proba).T


if __name__ == '__main__':
    print 'testing only'
    model = MultiLogistic()

    from sklearn import datasets
    from sklearn.cross_validation import StratifiedShuffleSplit

    digits = datasets.load_digits()
#    x, y = digits.data[:200], digits.target[:200]
    x, y = digits.data, digits.target
    x_train, x_val, y_train, y_val = None,None,None,None


    print y.shape
    skf = StratifiedShuffleSplit(y,test_size= 0.3, n_iterations=2)
    for train_index, test_index in skf:
        print 'in skf'
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
        print x_train.shape
        print x_val.shape

#    model.train_cross_validation(x_train, x_val, y_train, y_val, [1,2,50,x_train.shape[0]+1])

    from sklearn.svm import SVC
    model = SVC(C=1000000, kernel='poly', degree=3, probability=True)
    model.fit(x_train, y_train)
    proba_train = model.predict_proba(x_train)
    print 'train: ', functions.precision(proba_train, y_train)
    proba_val = model.predict_proba(x_val)
    print 'validate:', functions.precision(proba_val, y_val)


