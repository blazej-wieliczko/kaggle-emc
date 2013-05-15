# Read data stored as a CSR matrix and labels
from numpy.core.shape_base import vstack, hstack
from numpy import mean
from numpy import var
from scipy import sparse
import numpy
import time
import functions

def ReadData( filePath ):
    """ USAGE:
        reads a CSR sparse matrix from file and converts it
        to a Matrix library object in a CSR format
    
        PARAMETERS:
        filePath - full/relative path to the data file
    """
    
    # open file for reading
    inFile = open(filePath,"r")

    # read matrix shape
    matrixShape = numpy.fromstring(inFile.readline(),dtype = 'int',sep = ',');

    # read matrix data, indices and indptr
    data = numpy.fromstring(inFile.readline(),dtype = 'float',sep = ',');
    indices = numpy.fromstring(inFile.readline(),dtype = 'int',sep = ',');
    indptr = numpy.fromstring(inFile.readline(),dtype = 'int',sep = ',');

    # close file
    inFile.close()

    m = sparse.csr_matrix((data,indices,indptr),shape = (matrixShape[0],matrixShape[1]))
    #m = sparse.hstack(m,ones((m.shape[0]), 2))
    return m

def ReadLabels( filePath ):
    """ USAGE:
        reads a list of labels into memory
    
        PARAMETERS:
        filePath - full/relative path to the data file
    
        RETURN:
        list of labels
     """

    # read data from file
    data = numpy.loadtxt(open(filePath,"r"), dtype='int', delimiter = ",")

    return data


def time_stamp():
    return time.strftime('%X').replace(':',".")

def write_submission(filename, h):
    print 'Saving submission'
    timed_filename = filename + "-" + time_stamp()

    ids = numpy.array(range(h.shape[0]), ndmin = 2).T
    h = hstack((ids, h))
    numpy.savetxt(fname = timed_filename, X = h, fmt='%.15e', delimiter=",")
    functions.add_lineno_and_header(timed_filename)
