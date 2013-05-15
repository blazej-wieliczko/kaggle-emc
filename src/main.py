from genericpath import exists
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm.sparse.classes import NuSVC
from sklearn.utils import shuffle
from scipy import sparse
from numpy import var
from numpy import hstack
import sys

from sklearn.decomposition.nmf import NMF
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA

import emcio
import functions
import numpy as np
from sklearn.externals import joblib

import os
import scipy as sp
from optparse import OptionParser
# settings
from multilogistic import MultiLogistic

home_dir_var = os.getenv("HOME")

class Main:

    def __init__(self):
        self.is_trim_data = None
        self.trimmed_size = None
        self.val_percentage = None
        self.do_submission = None
        self.use_cached_models = None
        self.shuffle_data = None
        self.do_translate_labels = None
        self.use_cached_data = None
        self.train_data_pkl = None
        self.test_data_pkl = None
        self.pickled_labels = None
        self.model_file = None
        self.submission_file = None
        self.train_data_csv = None
        self.train_labels_csv = None
        self.test_data_csv = None
        self.penalty = None
        self.regularisation = None
        self.classifier = None
        self.nu = None
        self.is_preprocess_data = None
        self.sublinear_tf = None
        self.reg_grid = None


    #        use this number for validation split 0.741522
        parser = OptionParser()
        parser.add_option("--is_trim_data"         , dest="is_trim_data"        , default = False                                                      , action = "store_true")
        parser.add_option("--trimmed_size"         , dest="trimmed_size"        , default = 100                                                        , type = "int")
        parser.add_option("--val_percentage"       , dest="val_percentage"      , default = 1.0                                                        , type = "float")
        parser.add_option("--dont_do_submission"   , dest="do_submission"       , default = True                                                       , action = "store_false")
        parser.add_option("--use_cached_models"    , dest="use_cached_models"   , default = False                                                      , action = "store_true")
        parser.add_option("--shuffle_data"         , dest="shuffle_data"        , default = False                                                      , action = "store_true")
        parser.add_option("--do_translate_labels"  , dest="do_translate_labels" , default = False                                                      , action = "store_true")
        parser.add_option("--dont_use_cached_data" , dest="use_cached_data"     , default = True                                                       , action = "store_false")
        parser.add_option("--train_data_pkl"       , dest="train_data_pkl"      , default = home_dir_var + '/Dropbox/kaggle/EMC/data/train_data.pkl'   , type = "str")
        parser.add_option("--test_data_pkl"        , dest="test_data_pkl"       , default = home_dir_var + '/Dropbox/kaggle/EMC/data/test_data.pkl'    , type = "str")
        parser.add_option("--pickled_labels"       , dest="pickled_labels"      , default = home_dir_var + '/Dropbox/kaggle/EMC/data/train_labels.pkl' , type = "str")
        parser.add_option("--model_file"           , dest="model_file"          , default = '../models/logistic_regression.pkl'                        , type = "str")
        parser.add_option("--submission_file"      , dest="submission_file"     , default = '../submission/submission'                                 , type = "str")
        parser.add_option("--train_data_csv"       , dest="train_data_csv"      , default = home_dir_var + '/Dropbox/kaggle/EMC/data/train_data.csv'   , type = "str")
        parser.add_option("--train_labels_csv"     , dest="train_labels_csv"    , default = home_dir_var + '/Dropbox/kaggle/EMC/data/train_labels.csv' , type = "str")
        parser.add_option("--test_data_csv"        , dest="test_data_csv"       , default = home_dir_var + '/Dropbox/kaggle/EMC/data/test_data.csv'    , type = "str")
        parser.add_option("--regularisation"       , dest="regularisation"      , default = 'l2')
        parser.add_option("--penalty"              , dest="penalty"             , default = 0.0                                                        , type = 'float')
        parser.add_option("--classifier"           , dest="classifier"          , default = 'lr'                                                        , type = 'str')
        parser.add_option("--nu"                   , dest="nu"                  , default = '0.01'                                                      , type = 'float')
        parser.add_option("--preprocess"           , dest="is_preprocess_data"  , default = False                                                      , action = "store_true")
        parser.add_option("--sublinear_tf"         , dest="sublinear_tf"        , default = True                                                      , action = "store_true")

        (options, args) = parser.parse_args()
        self.__dict__ = options.__dict__
        self.cache_final_data = True
        self.reg_grid = [6.0]
        self.binary = True
        print 'regularisation:', self.regularisation,',',
        print 'penalty:', self.penalty,',',
        print 'sublinear_tf:', self.sublinear_tf,',',

    #----------------------

    def translate_labels(self):
        if self.do_translate_labels:
            unique_labels = np.unique(self.y)
            label_to_id = dict(zip(unique_labels, range(len(unique_labels))))
            # new labels are 0-based
            self.y = np.array([label_to_id[i] for i in self.y])


    def valsize(self, x):
        return int(x.shape[0] * self.val_percentage)


    def trim_data(self):
        if self.is_trim_data:
#            print 'Trimmed size:', self.trimmed_size
            if self.shuffle_data:
#                print 'shuffling'
                self.x, self.y = shuffle(self.x, self.y)
            self.x = self.x[0:self.trimmed_size]
            self.y = self.y[0:self.trimmed_size]
            self.translate_labels()

    def kmeans(self):
        pass

    def preprocess_data(self):
        if self.is_preprocess_data:
            self.compute_pca()
            
    def create_classifier(self):
        if self.classifier == 'lr':
#            print 'Using Logistic Regression'
            return LogisticRegression(penalty=self.regularisation, C=self.penalty, tol=0.05, dual=False, fit_intercept=True)
        elif self.classifier =='rf':
            return RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=2)
        elif self.classifier == 'nu':
            return NuSVC(kernel='linear', probability=True, nu=self.nu)
        elif self.classifier == 'nb':
            return MultinomialNB()
        elif self.classifier == 'knn':
            print 'using knn'
            return KNeighborsClassifier(n_neighbors=10)
        elif self.classifier == 'multilogistic':
            return MultiLogistic()
        else: return None


    def load_data(self):

        print 'noncached', ',',
        self.y = None

        if exists(self.train_data_pkl)\
           and exists(self.test_data_pkl)\
           and exists(self.pickled_labels)\
           and self.use_cached_data:
#               print 'Loading data from cache'
               self.x = joblib.load(self.train_data_pkl)
               self.y = joblib.load(self.pickled_labels)
               self.x_test = joblib.load(self.test_data_pkl)
        else:
#            print 'Not using data cache or no cache available, loading data from csv'
            self.x = emcio.ReadData(self.train_data_csv)
            self.y = emcio.ReadLabels(self.train_labels_csv)
            if self.do_submission:
                self.x_test = emcio.ReadData(self.test_data_csv)
                joblib.dump(self.x_test, self.test_data_pkl)
            joblib.dump(self.x, self.train_data_pkl)
            joblib.dump(self.y, self.pickled_labels)

        self.trim_data()
        #print 'Size x:', self.x.shape

        split = self.valsize(self.x)
        self.x_train = self.x[0:split]
        self.y_train = self.y[0:split]
        if self.do_validation():
            self.x_validate = self.x[split:]
            self.y_validate = self.y[split:]
            assert self.x_validate.shape[0] + self.x_train.shape[0] == self.x.shape[0]
        print 'TFIDF Transformation'
        self.transform_tfidf()

        self.preprocess_data()
        if self.cache_final_data:
            joblib.dump(self.x_train, "../data/x_train_final")
            if self.do_validation():
                joblib.dump(self.x_validate, "../data/x_validate_final")
            joblib.dump(self.x_test, "../data/x_test_final")

    def load_cached_data(self):
        print 'cached', ',',
        if self.cache_final_data and exists("../data/x_train_final"):
            self.x_train = joblib.load("../data/x_train_final")
            self.x_validate = joblib.load("../data/x_validate_final")
            self.x_test = joblib.load("../data/x_test_final")
            split = self.x_train.shape[0]
            self.y = joblib.load(self.pickled_labels)
            self.y_train = self.y[0:split]
            self.y_validate = self.y[split:]

        else:
            self.load_data()

    def predict(self):
        print 'Predicting'

        if self.classifier == 'knn':
            train_score = self.model.score(self.x_train, self.y_train)
            print 'KNN train score:', train_score

            if self.do_validation():
                val_score = self.model.score(self.x_validate, self.y_validate)
                print 'KNN val score: ', val_score
        else:
            import pprint

            self.predict_train = self.model.predict_proba(self.x_train)
            print 'Training Score logloss: ', functions.logloss(self.predict_train, self.y_train), ',',
            print 'Training Score precision: ', functions.precision(self.predict_train, self.y_train), ',',

            if self.do_validation():
                self.predict_validate = self.model.predict_proba(self.x_validate)
                precision = functions.logloss(self.predict_validate, self.y_validate)
                print 'Validation Score: ', precision, ',',
                precision = functions.precision(self.predict_validate, self.y_validate)
                print 'Validation Score precision: ', precision, ',',


    def eval_testset(self):
        missed_rowids = []
        if self.do_submission:
            predict_test = self.model.predict_proba(self.x_test)
            emcio.write_submission(self.submission_file, predict_test)
        if self.do_validation():
            predict_validation = self.model.predict_proba(self.x_validate)
            emcio.write_submission(self.submission_file + '_validate', predict_validation)

            missed_rowids = functions.precision_with_missed(predict_validation, self.y_validate)
        return missed_rowids

    def do_validation(self):
        return self.val_percentage < 1.0

    def get_extra_features(self, m):
        if True:
            return
        m_mean = m.sum(1)
        m_mean = m_mean.reshape(m_mean.shape[0], 1)

        extra_features = m_mean
        return extra_features

    def nonzeroify(self, x):
        pass

    def transform_tfidf(self):
        print 'Transformation'
        sys.stdout.flush()
        transformer = None
        if self.binary:
            class BinaryTransformer():
                def transform(self, x):
                    print 'transforming'
                    sys.stdout.flush()
                    x.data[:] = 1
                    return x
            transformer = BinaryTransformer()
        else:
            transformer = TfidfTransformer(sublinear_tf = self.sublinear_tf).fit(self.x_train)
        extra_features = self.get_extra_features(self.x_train)
        #self.x_train = sparse.hstack((self.x_train, transformer.transform(self.x_train)))
        self.x_train = transformer.transform(self.x_train)

        #self.nonzeroify(self.x_train)
        # self.x_train = sparse.hstack([self.x_train, extra_features]).tocsr()

        if self.do_submission:
            extra_features = self.get_extra_features(self.x_test)
            #self.x_test = sparse.hstack((self.x_test, transformer.transform(self.x_test)))
            self.x_test = transformer.transform(self.x_test)

            #self.nonzeroify(self.x_test)
            # self.x_test = sparse.hstack([self.x_test, extra_features]).tocsr()

        if self.do_validation():
            extra_features = self.get_extra_features(self.x_validate)
            #self.x_validate = sparse.hstack((self.x_validate, transformer.transform(self.x_validate)))
            self.x_validate = transformer.transform(self.x_validate)
            #self.nonzeroify(self.x_validate)
            # self.x_validate = sparse.hstack([self.x_validate, extra_features]).tocsr()
        print 'after tfidf'
        sys.stdout.flush()


    def train_model(self):
        # load model
        print 'loading model'
        sys.stdout.flush()
        if self.use_cached_models and exists(self.model_file):
#            print 'Loading model from file'
            self.model = joblib.load(self.model_file)
        else:
#            print 'Training model of size', self.x_train.shape
            self.model = self.create_classifier()
#            print 'Created model, training'

            if self.classifier == 'multilogistic':
                self.model.train_cross_validation(self.x_train, self.x_validate, self.y_train, self.y_validate,
                    self.reg_grid)
            else:
                print 'shit - ', self.classifier
                sys.stdout.flush()
                self.model.fit(self.x_train, self.y_train)
            #joblib.dump(self.model, self.model_file)

    def compute_pca(self):
#        print 'We have ', self.x.shape[1], 'features. Reducing dimensionality.'
        pca_count = 200
        pca = RandomizedPCA(pca_count, copy = False, whiten=True)
        pca.fit(self.x_train)
        self.x_train = pca.transform(self.x_train)
        if self.do_submission:
            self.x_test = pca.transform(self.x_test)

        if self.do_validation():
            self.x_validate = pca.transform(self.x_validate)

if __name__ == '__main__':
   m1 = Main()
   print 'created main'
   sys.stdout.flush()
   m1.load_cached_data()
   m1.train_model()
   m1.predict()
   m1.eval_testset()
   print 'done'
