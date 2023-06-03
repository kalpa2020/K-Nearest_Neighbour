"""
    Classification Assignment

    Reads a file and saves the data in arrays. Then test several versions of the
    K-Nearest Neighbour algorithm using the saved data.
"""

import numpy as np
import csv
from matplotlib import pyplot as plt

## Reads a file and process the data

filename = 'OnlineNewsPopularity.csv'

# Reads a file of data and saves the labels (targets) in a list
rawData = np.genfromtxt(filename, delimiter=",", names=True)
labels = rawData.dtype.names

# Reads a file of data
rawData = np.genfromtxt(filename, delimiter=",", skip_header=1)

# Saves the data of the classification column in a list
labels = rawData[:,60] >= 1400

# Saves the data of the file in a list
data = rawData[:,1:60]

## Splits the data into training and testing data sets

from sklearn.model_selection import train_test_split

trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)

## KNN implementation

# Classifier function which predicts the label of a new item using the training data set
def classifier(newItem, dataSet, dataSetLabels, k):
    global predictedClassifierList

    distance = dataSet - newItem
    square = np.square(distance)
    sum = np.sum(square, axis=1)
    squareRoot = np.sqrt(sum)
    sortedArray = np.argsort(squareRoot)
    closestItems = dataSetLabels[sortedArray][:k]

    predictedClassifierList = {
        "True": 0,
        "False": 0
    }

    for element in closestItems:
        if (element == True):
            predictedClassifierList["True"] += 1
        else:
            predictedClassifierList["False"] += 1

    if (predictedClassifierList["True"] > predictedClassifierList["False"]):
        predictedClassifier = "True"
    else:
        predictedClassifier = "False"

    return predictedClassifier

kValue = 26
runs = 3
predictedCorrectlyList = []

# Runs the algorithm run times to get the overall accuracy
for run in range(0, runs, 1):
    predictedLabels = []

    for row in testingData:
        # Adds the predicted label of the new item to the predictedLabels list
        predictedLabels.append(classifier(row, trainingData, trainingLabels, kValue))

    predictedCorrectly = 0
    count = 0

    # Calculate how many labels were predicted correctly by comparing them to the testing labels
    for label in predictedLabels:
        if (label == str(testingLabels[count])):
            predictedCorrectly += 1
        count += 1

    # Adds how many labels were correctly predicted on each run to the predictedCorrectlyList,
    # in order to calculate the overall accuracy later
    predictedCorrectlyList.append(predictedCorrectly)

    print()
    print("======== MY IMPLEMENTATION ============")
    print("Correct: ", predictedCorrectly)
    print("Incorrect: ", (len(predictedLabels) - predictedCorrectly))
    print("Accuracy: ", ((predictedCorrectly / (len(testingLabels))) * 100),"%")

totalCorrectlyPredicted = 0

for i in predictedCorrectlyList:
    totalCorrectlyPredicted += i

print()
print("Overall Accuracy: ", ((totalCorrectlyPredicted / ((len(testingLabels)) * runs)) * 100), "%")

#-----------------------Uses SKLearn's implementation---------------------------

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

# Training the classifier
clf = clf.fit(trainingData, trainingLabels)

# Ignore future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Classify and measure accuracy
from sklearn.metrics import accuracy_score, f1_score

print()
print("======== SKLEARN\'S IMPLEMENTATION =============")
prediction = clf.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print("Correct: ", correct)
print("Incorrect: ", len(prediction) - correct)
print("Accuracy: ", accuracy_score(testingLabels, prediction) * 100,"%")

## SKLearn’s KNN implementation

print()
print("======== SKLEARN\'S KNN IMPLEMENTATION =============")

# All parameters are set to default values
c0 = KNeighborsClassifier()
c0 = c0.fit(trainingData, trainingLabels)
prediction = c0.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("All Default Parameters")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# N_Neighbors parameter is set to 1
c1 = KNeighborsClassifier(n_neighbors=1)
c1 = c1.fit(trainingData, trainingLabels)
prediction = c1.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("N_Neighbors = 1")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# N_Neighbors parameter is set to 14
c1 = KNeighborsClassifier(n_neighbors=14)
c1 = c1.fit(trainingData, trainingLabels)
prediction = c1.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("N_Neighbors = 14")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# Weights parameter is set to distance
c2 = KNeighborsClassifier(weights='distance')
c2 = c2.fit(trainingData, trainingLabels)
prediction = c2.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("Weights = distance")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# N_Neighbors parameter is set to 1 and Weights parameter is set to distance
c2 = KNeighborsClassifier(n_neighbors=1, weights='distance')
c2 = c2.fit(trainingData, trainingLabels)
prediction = c2.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("N_Neighbors = 1, Weights = distance")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# N_Neighbors parameter is set to 14 and Weights parameter is set to distance
c2 = KNeighborsClassifier(n_neighbors=14, weights='distance')
c2 = c2.fit(trainingData, trainingLabels)
prediction = c2.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("N_Neighbors = 14, Weights = distance")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# P parameter is set to 1
c3 = KNeighborsClassifier(p=1)
c3 = c3.fit(trainingData, trainingLabels)
prediction = c3.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("P = 1")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")

# N_Neighbors parameter is set to 1 and P parameter is set to 1
c3 = KNeighborsClassifier(n_neighbors=1, p=1)
c3 = c3.fit(trainingData, trainingLabels)
prediction = c3.predict(testingData)
correct = (prediction == testingLabels).sum()     # Number of Trues
print()
print("N_Neighbors = 1, P = 1")
print("Correct:",correct)
print("Incorrect:",len(prediction)-correct)
print("Accuracy:",accuracy_score(testingLabels, prediction)*100,"%")
print("F1:",f1_score(testingLabels, prediction, average="macro")*100,"%")