# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:04:05 2017

@author: Roman_
"""

import utils
import time

currentTime = time.time()

print 'Loading test images...'
dictionary = utils.unpickle(utils.cifar10Path + "/test_batch")
testImagesArr =  dictionary.get('data')
testLabels = dictionary.get('labels')

# load the test images themselves from the flat arrays 
testImages = utils.flatArraysToImages(testImagesArr)
print 'Finished loading test images'

print 'Running separate PCAs on all 3 channels...'
# load the 3 trained pca models from files
pcaRed = utils.unpickle(utils.modelsPath + '/' + utils.pcaRedFileName)
pcaGreen = utils.unpickle(utils.modelsPath + '/' + utils.pcaGreenFileName)
pcaBlue = utils.unpickle(utils.modelsPath + '/' + utils.pcaBlueFileName)

# transform the test images' channels using the models
transformedTestImagesRed = pcaRed.transform(testImagesArr[:,0:1024])
transformedTestImagesGreen = pcaGreen.transform(testImagesArr[:,1024:2048])
transformedTestImagesBlue = pcaBlue.transform(testImagesArr[:,2048:3072])

print 'Finished running PCAs'

print 'Creating feature vector for each image...'
testFeatureVectors = utils.getFeatureVectors(testImages, 
                                         transformedTestImagesRed, 
                                         transformedTestImagesGreen, 
                                         transformedTestImagesBlue)
print 'Finished creating feature vector for each image'

print 'Classifying...'

classifier = utils.unpickle(utils.modelsPath + '/' + utils.classifierFileName)

# get predicted labels for each image using the classifier
predictedLabels = classifier.predict(testFeatureVectors)

# count the number of correct labels. This could be done using classifier.score,
# but we need the predicted labels themselves in order to create the confusion matrix
correct = 0
for i in range(len(predictedLabels)):
    if predictedLabels[i] == testLabels[i]:
        correct += 1

# the accuracy is the number of correct labels / number of total images
score = correct / float(len(predictedLabels))

print 'Finished classifying'

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
           'frog', 'horse', 'ship', 'truck']

print 'Took ' + str(time.time() - currentTime) + ' seconds to classify'
print 'Accuracy: ' + str(score * 100) + '%'
print 'Error rate: ' + str(100 - score * 100) + '%'

print 'Confusion matrix is: '

# print the confusion matrix as a table
print ' '.join(map(lambda x: '%10s' % x, classes))
confusionMatrix = utils.getConfusionMatrix(predictedLabels, testLabels)
for i in range(len(confusionMatrix)):
    print ' '.join(map(lambda x: '%10s' % x, confusionMatrix[i])) + ('%12s' % classes[i])

