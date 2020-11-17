#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:39 2020

@author: pkrobinette
Description: Comparing multiple machine learning classifiers for accuracy and digit
recognition.
"""

# import files 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


np.random.seed(11) # Set the seed

# ____________________________________________________________________________
# Paths of the Training and Test data.

PATH2 = '/Users/probinette4/Desktop/PrestonTest.npz'

info = np.load(PATH2)
PrestonData = pd.DataFrame({'Name': info['Name'],'Vmax': info['Vmax'], 'Vmin': info['Vmin'],
                     'Vrange': info['Vrange'], 'Vavg': info['Vavg'],
                     'Fmax': info['Fmax']})


# The features used in the regression model.
testFeature = ['Vmax', 'Vmin', 'Fmax']
X = PrestonData[testFeature]
Y = PrestonData['Name']


# Possible Changes to data
#normalized_X = preprocessing.normalize(X) # Data Normalization
standardized_X = preprocessing.scale(X) # Data Standardization


# Split the data into a training and test set 80/20 split.
xtrain, xtest, ytrain, ytest = train_test_split(standardized_X, Y, test_size = 0.20)

#_____________________________________________________________________________
# Logistic Regresssion
print('LOGISTIC REGRESSION:')

# Make a model using the logistic regression algorithm
log_reg = LogisticRegression(multi_class='multinomial',
                                 solver = 'lbfgs', C=1e10,
                                 max_iter = 9000, dual = False)
log_reg.fit(xtrain,ytrain)

# Logistic Predictions
y_pred = log_reg.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
logR_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(logR_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the Logisitc Regression Classifier")

# Confusion Matrix
plt.figure(2)
logR_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(logR_cm, annot=True)
plt.title('Confusion Matrix of the Logistic Regression Classifier')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#______________________________________________________________________________
# Suppoprt Vector Machine
print('SVM:')

# Make a model using logistic regression algorithm
#support = svm.SVC(kernel='poly')  #polynomial kernal, DO NOT USE ON PC
support = svm.SVC(kernel = 'rbf', gamma = 'auto')   # Gaussian kernel
#support = svm.SVC(kernel = 'linear) # Linear kernel
support.fit(xtrain,ytrain)

# Linear Predictions
y_pred = support.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
svm_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(svm_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the SVM Classifier")

# Confusion Matrix
plt.figure(2)
svm_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(svm_cm, annot=True)
plt.title('Confusion Matrix of the SVM Classifier')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#_____________________________________________________________________________
# Decision Tree Classifier
print('DECISION TREE CLASSIFIER:')
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=12, splitter = 'best')
tree_clf.fit(xtrain,ytrain)

# Plot the Decision Tree
#plt.figure(1)
#plt.clf()
#tree.plot_tree(tree_clf.fit(xtrain,ytrain),fontsize=6)
#plt.axis('tight')
#plot = tree.export_graphviz(tree_clf, out_file=None)
#graph = graphviz.Source(plot)
#graph.render("gestures")

# Predictions
y_pred = tree_clf.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
dcr_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(dcr_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the Decision Tree Classifier")

# Confusion Matrix
plt.figure(2)
dcr_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(dcr_cm, annot=True)
plt.title('Confusion Matrix of the Decision Tree Classifier')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#_____________________________________________________________________________
# Random Forest Classifier
print('RANDOM FOREST CLASSIFIER:')

rnd_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(xtrain,ytrain)

# Predictions
y_pred = rnd_clf.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
rfc_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(rfc_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the Random Forest Classifier")

# Confusion Matrix
plt.figure(2)
rfc_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(rfc_cm, annot=True)
plt.title('Confusion Matrix of the Random Forest Classifier')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#_____________________________________________________________________________
# K Nearest Neighbor
print('K NEIGHBOR CLASSIFIER:')

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(xtrain, ytrain)

# Predictions
y_pred = neigh.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
kn_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(kn_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the KNN Classifier")

# Confusion Matrix
plt.figure(2)
kn_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(kn_cm, annot=True)
plt.title('Confusion Matrix of the KNN Classifier')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# ____________________________________________________________________________
# KMeans Cluster (Unsupervised)
print("KMEANS CLUSTERING:")

KMFeatures = ['Vrange', 'Fmax']
X = PrestonData[KMFeatures]
Y = PrestonData['Name']

standardized_X = preprocessing.scale(X) # Data Standardization
#normalized_X = preprocessing.normalize(X) # Data Normalization


# Split the data into a training and test set 80/20 split.
xtrain, xtest, ytrain, ytest = train_test_split(standardized_X, Y, test_size = 0.20)

kmeans = KMeans(n_clusters=7)
kmeans = kmeans.fit(xtrain)

# Predicting
y_pred = kmeans.predict(xtest)

# Getting the cluster centers
C = kmeans.cluster_centers_

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
#print("Precision:",metrics.precision_score(ytest, y_pred, average=None))
#print("Recall:", metrics.recall_score(ytest, y_pred, average=None))
#print("F1 Score:", metrics.f1_score(ytest,y_pred, average=None))

# Classification Report
plt.figure(1)
km_cr = classification_report(ytest, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(km_cr).iloc[:-1, :-3].T, annot=True, cmap = "YlGnBu")
plt.title("Classification Report of the KMeans Cluster")

# Confusion Matrix
plt.figure(2)
km_cm = confusion_matrix(ytest, y_pred)
sns.heatmap(km_cm, annot=True)
plt.title('Confusion Matrix of the Kmeans Cluster')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.figure(3)
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain)
plt.colorbar()
#plt.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)
plt.xlabel("Vamp")
plt.ylabel("Fmax")
plt.title("Voltage Amplitude vs. Max Frequency")
plt.show()
