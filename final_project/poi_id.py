#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi', 'fraction_stock_exercised'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Task 3: Create new feature(s)

### new feature is: fraction_stock_exercised

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_stock_exercised=dict_to_list("exercised_stock_options","total_stock_value")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_stock_exercised"]=fraction_stock_exercised[count]
    count +=1

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

scaler = MinMaxScaler()
skb = SelectKBest(f_classif)
pipeline = Pipeline(steps=[('scaling', scaler),("kbest", skb), ("DTC", DecisionTreeClassifier(random_state = 42))])

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


parameters = {'kbest__k': [1,2,3]}
cv = StratifiedShuffleSplit(labels, 20, random_state = 42)
gs = GridSearchCV(pipeline, parameters, cv = cv)
gs.fit(features, labels)

# The optimal model selected by GridSearchCV:
clf = gs.best_estimator_

# Access the feature importances

# create a new list that contains the features selected by SelectKBest
# in the optimal model selected by GridSearchCV
features_selected=[features_list[i+1] for i in clf.named_steps['kbest'].get_support(indices=True)]

# The step in the pipeline for the Decision Tree Classifier is called 'DTC'
# that step contains the feature importances
importances = clf.named_steps['DTC'].feature_importances_

import numpy as np
indices = np.argsort(importances)[::-1]

# Use features_selected, the features selected by SelectKBest, and not features_list
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

clf, dataset, feature_list = load_classifier_and_data()
test_classifier(clf, dataset, feature_list)