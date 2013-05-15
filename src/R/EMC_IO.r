# Read data stored as a CSR matrix and labels
library(Matrix)

EMC_ReadData <- function( filePath ){
	# USAGE:
    # reads a CSR sparse matrix from file and converts it
	# to a Matrix library object in a CSR format
    #
    # PARAMETERS:
    # filePath - full/relative path to the data file

	f<-file(filePath,'r')
	shape<-as.integer(strsplit(readLines(f,1),",")[[1]])
	x<-as.numeric(strsplit(readLines(f,1),",")[[1]])
	j<-as.integer(strsplit(readLines(f,1),",")[[1]])
	p<-as.integer(strsplit(readLines(f,1),",")[[1]])

	matOut<-sparseMatrix(x=x, j=j+1, p=p, dims=c(shape[1],shape[2]))

	return(matOut)
}

EMC_ReadLabels <- function( filePath ){
    # USAGE:
    # reads a list of labels into memory
    #
    # PARAMETERS:
    # filePath - full/relative path to the data file
    #
    # RETURN:
    # list of labels

    return( read.csv( file=filePath, head=FALSE, sep="," ) )
}
