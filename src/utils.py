# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:19:39 2017

@author: Roman_
"""

import cPickle
import cv2
import numpy as np
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

cifar10Path = '../cifar-10-batches-py'
modelsPath = '../models'
pcaBlueFileName = 'pcaBlue'
pcaGreenFileName = 'pcaGreen'
pcaRedFileName = 'pcaRed'
classifierFileName = 'featureVectorClassifier'

# this is the number of components for all PCAs
pcaNComponents = 20

# the number of total classes to classify - used to create the confusion matrix
totalClasses = 10

def unpickle(file):
    """ 
    Unpickle file in given path and return its content 
    """
    return cPickle.load(open(file, 'rb'))

def dumpToFile(data, path):
    """ 
    Pickle the given data and save it to the given path
    """
    # if the folder doesn't exist yet, we create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # cPickle.HIGHEST_PROTOCOL means cPickle uses the best compression it can
    cPickle.dump(data, open(path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def flatArraysToImages(arr):
    """ 
    Format a list of flat arrays as images of shape (32 X 32 X 3)
    The input should be a list of 1D arrays each having 3072 byte elements,
    where the first 1024 bytes represent the red channel, the second represent 
    the green channel and the last represents the blue channel of the image
    """
    return map(lambda x: np.stack(x, axis=2), np.reshape(arr, (len(arr), 3, 32, 32)))


def getHogDescriptor():
    """
    Create and return a histogram of oriented gradients descriptor
    """
    return cv2.HOGDescriptor(_winSize=(16,16),_blockSize=(16,16),_blockStride=(1,1),
                             _cellSize=(8,8),_nbins=9,_derivAperture=1,_winSigma=3.5,
                             _histogramNormType=0,_L2HysThreshold=2.0000000000000001e-01,
                             _gammaCorrection=True,_nlevels=2,_signedGradient=True)
    
def getFeatureVectors(images, imagesRed,
                      imagesGreen, imagesBlue):
    """
    Create a feature vector for each image and return the vectors as a list
    
    Input: list of images, and 3 lists of color representations of all images
    Output: a list of feature vectors where vectors[i] corresponds to images[i]
    
    The feature vector is a concatenation of the image's HOG transform, and the
    three color representations of the given image
    """
    vectors = []
    hogDescriptor = getHogDescriptor()
    for i in range(len(images)):
        # transform the image to gray
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        
        # compute and flatten the HOG of the image
        currentHog = np.ravel(hogDescriptor.compute(gray))
        concat = np.concatenate((currentHog,
                                 imagesRed[i],
                                 imagesBlue[i], 
                                 imagesGreen[i]))
        vectors.append(concat)
        if i % 5000 == 0:
            print 'created feature vector for ' + str(i) + '/' + str(len(images)) + ' images'
    print 'created feature vector for ' + str(len(images)) + '/' + str(len(images)) + ' images'
    return vectors
    
def getClassifier():
    """
    Get the classifier to use in order to classify the images' feature vectors
    """
    clf1 = QuadraticDiscriminantAnalysis()
    clf2 = LinearDiscriminantAnalysis()    
    return VotingClassifier(estimators=[('qda', clf1), ('lda', clf2)],
                                        weights=[0.4,0.6],
                                        voting='soft')


def getConfusionMatrix(predictedLabels, actualLabels):
    """
    Get the confusion matrix of the given predicted labels vs. the actual labels
    """
    matrix = [[0 for x in range(totalClasses)] for y in range(totalClasses)]
    for i in range(len(predictedLabels)):
        matrix[predictedLabels[i]][actualLabels[i]] += 1
    return matrix