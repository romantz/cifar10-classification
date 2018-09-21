# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:18:11 2017

@author: Roman_
"""

import utils
import numpy as np
import time
from sklearn.decomposition import PCA

currentTime = time.time()

print 'Loading train images...'
dictionary = utils.unpickle(utils.cifar10Path + "/data_batch_1")
trainImagesArr =  dictionary.get('data')
trainLabels = dictionary.get('labels')
for i in range(2, 6):
    dictionary = utils.unpickle(utils.cifar10Path + "/data_batch_" + str(i))
    trainImagesArr = np.concatenate((trainImagesArr, dictionary.get('data')))
    trainLabels = np.concatenate((trainLabels, dictionary.get('labels')))

# load the train images themselves from the flat arrays 
trainImages = utils.flatArraysToImages(trainImagesArr)
print 'Finished loading train images'

print 'Running PCA on all 3 channels, this will take a few moments...'
pcaRed = PCA(n_components=utils.pcaNComponents,svd_solver='arpack')
pcaGreen = PCA(n_components=utils.pcaNComponents,svd_solver='arpack')
pcaBlue = PCA(n_components=utils.pcaNComponents,svd_solver='arpack')

# fit and transform pca for each channel of each image independently
transformedTrainImagesRed = pcaRed.fit_transform(trainImagesArr[:,0:1024], trainLabels)
transformedTrainImagesGreen = pcaGreen.fit_transform(trainImagesArr[:,1024:2048], trainLabels)
transformedTrainImagesBlue = pcaBlue.fit_transform(trainImagesArr[:,2048:3072], trainLabels)

# save the three fitted pca models to files
utils.dumpToFile(pcaRed, utils.modelsPath + '/' + utils.pcaRedFileName)
utils.dumpToFile(pcaGreen, utils.modelsPath + '/' + utils.pcaGreenFileName)
utils.dumpToFile(pcaBlue, utils.modelsPath + '/' + utils.pcaBlueFileName)
print 'Finished running PCA on all 3 channels'

print 'Creating feature vector for each image...'
trainFeatureVectors = utils.getFeatureVectors(trainImages, 
                                         transformedTrainImagesRed, 
                                         transformedTrainImagesGreen, 
                                         transformedTrainImagesBlue)
print 'Finished creating feature vector for each image'

print 'Fitting classifier...'
classifier = utils.getClassifier()

classifier.fit(trainFeatureVectors, trainLabels)

# save the fitted classifier to file
utils.dumpToFile(classifier, utils.modelsPath + '/' + utils.classifierFileName)
print 'Finished fitting classifier'

print 'Took ' + str(time.time() - currentTime) + ' seconds to train'





