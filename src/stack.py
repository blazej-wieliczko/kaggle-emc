import functions
import numpy as np

def print_accuracy(predicted, y):
    print 'logloss: ', functions.logloss(predicted, y)
    print 'precision: ', functions.precision(predicted, y)

y= []


def load_x(path, skip=False):
    return np.genfromtxt(path, delimiter=',', skip_header=skip)


x1 = load_x('../submission/submission-binary_validate-17.47.08')
print x1.shape
x3 = load_x('../submission/submission_validate-16.29.12-numbered', skip=True)
print x3.shape

x1 = x1[:, 1:]
x3 = x3[:, 1:]

X = np.hstack((x1,x3))

pickled_labels = '../../../data/train_labels.pkl'

from sklearn.externals import joblib
y = joblib.load(pickled_labels)

y_validate = y[-x1.shape[0]:]
print y_validate.shape


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(dual=False, fit_intercept=True)


from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_validate, test_size=0.7, random_state=0)

classifier.fit(X, y_validate)
predicted = classifier.predict_proba(X)

print 'Blended'
print_accuracy(predicted, y_validate)
print 'x1'
print_accuracy(x1, y_validate)
print 'x3'
print_accuracy(x3, y_validate)



