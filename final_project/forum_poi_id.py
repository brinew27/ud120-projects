import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

features_list = ['poi', 'bonus', 'shared_receipt_with_poi', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler()
kbest = SelectKBest(f_classif)
clf_DTC = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('scaling', scaler), ("kbest", kbest), ("DTC", clf_DTC)])
parameters = {'kbest__k': [1, 2, 3, 4, 5, 6]}
cv = StratifiedShuffleSplit(labels, 20, random_state=42)
gs = GridSearchCV(pipeline, parameters, cv=cv)
gs.fit(features, labels)

# The optimal model selected by GridSearchCV:
clf = gs.best_estimator_

print ' '
# use test_classifier to evaluate the model
# selected by GridSearchCV
print "Tester Classification report:"
test_classifier(clf, my_dataset, features_list)
print ' '

# Access the feature importances

# The step in the pipeline is called 'DTC':
importances = clf.named_steps['DTC'].feature_importances_

# The rest of the code is as you wrote it earlier:
import numpy as np

indices = np.argsort(importances)[::-1]

print 'Feature Ranking: '
for i in range(3):
    print "feature no. {}: {} ({})".format(i + 1, features_list[indices[i] + 1], importances[indices[i]])