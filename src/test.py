import scipy
import numpy as np

__author__ = 'bwieliczko'

import emcio
import functions
import main

print emcio.__name__
print functions.__name__
print main.__name__
print 'tested!'

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
    help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
    action="store_false", dest="verbose", default=True,
    help="don't print status messages to stdout")

(options, args) = parser.parse_args()

def add_length():

    x = scipy.sparse.csc_matrix([[1,2,3],[4,5,6],[7,8,9]])

    s = x.sum(axis=1)
#    s = scipy.sparse.csc_matrix(s)

    print s.shape
    print x.shape
    x = scipy.sparse.hstack((x, s))

    print x.shape




if __name__ == "__main__":
    add_length()




