# cifar10-classification

Popular image classification problem. My solution uses only classic machine learning methods (no deep learning).
I use a combination of two classifiers (QuadraticDiscriminantAnalysis and LinearDiscriminantAnalysis) to achieve the best results.
Each image is translated to a feature vector which consists of a PCA vector for each color component and a hog descriptor for the entire image.
The accuracy of this model is around 68%. The training time is less than 30s and the weight of the resulting model is around 12mb.