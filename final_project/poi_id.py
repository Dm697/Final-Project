#!/usr/bin/python
import os
import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',\
                 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',\
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',\
                 'long_term_incentive', 'from_poi_to_this_person']# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
def checkNaN(value):
    if value == 'NaN' or value == 'nan':
        return True
for key, value in data_dict.iteritems():
    if data_dict[key]['poi'] == False:
        if checkNaN(data_dict[key]['salary']) or checkNaN(data_dict[key]['bonus']) or checkNaN(data_dict[key]['total_stock_value']):
            data_dict[key]['compensation_ratio'] = 0
        else:
            data_dict[key]['compensation_ratio'] = float(data_dict[key]['total_stock_value']) / \
                                                   ((float(data_dict[key]['salary']) + float(data_dict[key]['bonus']) + float(data_dict[key]['total_stock_value'])))
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

clf_NB = GaussianNB()
clf_DT = tree.DecisionTreeClassifier(random_state = 71)
kmeans = KMeans(n_clusters = 2, random_state=0, n_init = 50)
clf_RF = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', min_samples_split = 2, min_samples_leaf = 4,
                                random_state = 0)
clf_KNN = KNeighborsClassifier(n_neighbors = 1, algorithm = 'ball_tree', weights = 'uniform', leaf_size = 1)
cv = StratifiedShuffleSplit(n_splits = 100)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

"""Scaling Features"""
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.fit_transform(features_test)

"""PCA"""
#set up PCA to explain pre-selected % of variance (perc_var)
perc_var = .99
pca = PCA(n_components = perc_var)
pca_transform = pca.fit_transform(rescaled_features_train)
print 'PCA explained variance ratio:', pca.explained_variance_ratio_

"""SelectKBest"""
#set up SelectKBest to find out which features have the best feature scores
selection = SelectKBest()
selection.fit_transform(rescaled_features_train, labels_train)
print 'SKB features best parameters:', selection.scores_

# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

#Build pipeline to tune PCA/ SKB parameters, scale features, and implement decision tree
pipeline = Pipeline([('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ("features", combined_features),
                     ("dt", clf_DT)])

#Parameters to adjust in GridSearchCV for PCA and SKB
param_grid = dict(features__pca__n_components = [None, 1, 2, 3, 4, 5, 6],
                  features__univ_select__k = [1, 2, 3, 4, 5, 6])

#Create GridSearch object
gs = GridSearchCV(pipeline, param_grid = param_grid, verbose = 10, scoring = 'f1', cv = cv)

#Fit to data and determine best estimator
gs.fit(features_train, labels_train)
print 'best feature parameters:', (gs.best_estimator_)

#Adjust SelectKBest to reflect best parameter tune of k = 3
selection = SelectKBest(k = 3)

#Create a new pipeline that only includes univariate features
pipeline = Pipeline([('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ("features", selection),
                     ("dt", clf_DT)])

#Parameters to adjust in GridSearchCV for decision tree
param_grid = dict(dt__criterion = ['gini', 'entropy'],
                  dt__min_samples_split = [2, 3, 4, 5, 5, 6, 7])

#Create new GridSearch object that includes new parameters
gs = GridSearchCV(pipeline, param_grid = param_grid, verbose = 10, scoring = 'f1', cv = cv)

#Fit to data and determine best parameters for DT
gs.fit(features_train, labels_train)
print 'best decision tree parameters:', (gs.best_estimator_)

#Create classifier for output
clf_DT = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 3)
pipeline = Pipeline([("features", selection), ("dt", clf_DT)])
clf = pipeline.fit(features_train, labels_train)


### Task 6: Dump classifier, dataset, and features_list so it can
### best tested using tester.py
dump_classifier_and_data(clf, my_dataset, features_list)
