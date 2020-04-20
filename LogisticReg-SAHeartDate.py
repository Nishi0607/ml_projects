# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:45:57 2020

@author: NK
"""
#1. Logistic regression modeling for SAHeart data set

import pandas as pd

data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data')

#Encode the categorical columns
encoded_data = pd.get_dummies(data, columns=['famhist'], drop_first=True)

from sklearn.model_selection import train_test_split

features = list(encoded_data.columns)
features.remove('row.names')
features.remove('chd')

X = encoded_data[features]
y = encoded_data.chd

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=40)

import statsmodels.api as sm

logit_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()

logit_model.summary2()

significant_vars = ['tobacco', 'ldl', 'typea', 'age', 'famhist_Present']

X_significant = encoded_data[significant_vars]

X_train, X_test, y_train,y_test = train_test_split(X_significant, y, test_size=0.2, random_state=40)

logit_model_2 = sm.Logit(y_train, X_train).fit()

logit_model_2.summary2()

#Get the predicted probabilities
y_pred = logit_model_2.predict(X_test)

#tobacco, ldl, age and famhist_Present have +ve coefficients, hence they affect the probability of chd positively
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, drop_intermediate=False)

#Optimal cut-off probability using Youden's index
tpr_fpr_df = pd.DataFrame({'tpr':tpr, 'fpr':fpr, 'thresholds':threshold})
tpr_fpr_df.head()
tpr_fpr_df['diff'] = tpr_fpr_df['tpr'] - tpr_fpr_df['fpr']
tpr_fpr_df.sort_values('diff', ascending=False)[0:5]

#Probability cut-off of 0.448 gives the maximum tpr-fpr diff
y_pred_df = pd.DataFrame({'actual':y_test, 'predicted_prob':y_pred})
y_pred_df['pred'] = y_pred_df['predicted_prob'].map(lambda x: 1 if x > 0.448 else 0)
y_pred_df.head()

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

def get_conf_matrix(actual, pred):
    conf_mat = confusion_matrix(actual, pred, [1,0])
    sn.heatmap(conf_mat, annot=True, fmt="0.2f", xticklabels=["chd", "no chd"], yticklabels=["chd", "no chd"])
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()
    return conf_mat
    
conf_mat = get_conf_matrix(y_pred_df['actual'], y_pred_df['pred'])

from sklearn.metrics import classification_report

print(classification_report(y_pred_df['actual'], y_pred_df['pred']))

#Optimal cut-off probability using cost-based approach
#Cost(FN) = 5 * Cost(FP)

def getTotalCost(actual, pred, cost_fps, cost_fns):
    #Get the confusion matrix
    conf = confusion_matrix(actual, pred, [1,0])
    conf_mat = np.array(conf)
    return conf_mat[0,1] * cost_fns + conf_mat[1,0] * cost_fps

#Capture the cost against different probability cutoff values
cost_df = pd.DataFrame(columns = ['prob', 'cost'])
probs = range(10, 50)    

idx = 0
for p in probs:
    cost = getTotalCost(y_pred_df['actual'], y_pred_df.predicted_prob.map(lambda x: 1 if x > (p/100) else 0), 1, 5)
    cost_df.loc[idx] = [(p/100), cost]
    idx+=1
    
cost_df.sort_values('cost', ascending=True)[0:10]
#Lowest cost achieved for prob cut-off of 0.14

y_pred_df['pred_new'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.14 else 0)

#Find the precision/recall of new classification model
print(classification_report(y_pred_df['actual'],y_pred_df['pred_new']))

#2. Implement decision tree for the same dataset
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data')

features = list(data.columns)
features.remove('row.names')
features.remove('chd')

X = data[features]
y = data['chd']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=40)

dtClassifier = DecisionTreeClassifier(criterion='gini', max_depth=4)
dtClassifier.fit(X_train, y_train)

#Check model performance on test data
y.value_counts()

y_pred = dtClassifier.predict(X_test)
from sklearn import metrics
metrics.roc_auc_score(y_test, y_pred)

#GridSearchCV to find optimal max_depth with Gini index as splitting criteria
from sklearn.model_selection import GridSearchCV

params_grid = {'criterion':['gini'], 'max_depth':[3,4,5,6,7,8,9,10]}

classifier = DecisionTreeClassifier()
clf = GridSearchCV(estimator=classifier, param_grid=params_grid, cv=10, scoring='roc_auc')
clf.fit(X_train, y_train)
clf.best_params_
clf.best_score_
#Optimal max_depth = 4

dtClassifierOpt = DecisionTreeClassifier(max_depth=4, criterion='gini')
dtClassifierOpt.fit(X_train, y_train)

#Displaying the decision tree (Meed GraphViz software installed on machine)
from sklearn.tree import export_graphviz
import pydotplus as pdot
from IPython.display import Image

#Export the tree into an odt file
export_graphviz(dtClassifierOpt, out_file="chd_tree.odt", feature_names=X_train.columns, filled=True)
chd_tree_graph = pdot.graphviz.graph_from_dot_file('chd_tree.odt')
chd_tree_graph.write_jpg('chd_tree.png')
#Render the png file
Image(filename='chd_tree.png')
