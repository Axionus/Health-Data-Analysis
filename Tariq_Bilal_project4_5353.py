import pandas as pd
import numpy as np
import scipy as sp
import plotly.express as px

from pandas.api.types import is_numeric_dtype, is_object_dtype
from pprint import pprint as pp

import sklearn
from sklearn.impute import SimpleImputer  # use Imputer if sklearn v0.19
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import util_5353

# Problem A [0 points]
def read_data(dataset_id):
  data = None
  # Begin CODE

  if dataset_id == 'breast_cancer':
    file = open('wdbc.data')
    data = pd.read_csv(file, header=None).drop(columns=[0])
  
  elif dataset_id == 'hyperthyroidism':
    datafile = 'allhyper.data'
    testfile = 'allhyper.test'
    with open(datafile) as train, open(testfile) as test:
      traindata = pd.read_csv(train, header=None, na_values='?')
      testdata = pd.read_csv(test, header=None, na_values='?')
      alldata = pd.concat( [traindata, testdata] )
      data = alldata

  elif dataset_id == 'cervical_cancer':
    file = open('risk_factors_cervical_cancer.csv')
    data = pd.read_csv(file, na_values='?').drop( columns=['Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology'] )


  elif dataset_id == 'liver_cancer':
    file = open('Indian Liver Patient Dataset (ILPD).csv')
    data = pd.read_csv(file, header=None)
  #print(data)
  # End CODE
  return data

# Problem B [0 points]
def dimensions(dataset_id, dataset):
  dim = None
  # Begin CODE
  r, c = dataset.shape
  dim=(r,c-1)
  # End CODE
  return dim

# Problem C [0 points]
def feature_values(dataset_id, dataset):
  fvalues = []
  # Begin CODE
  for feature in dataset:

    if dataset_id == 'breast_cancer' and feature == 1:
      continue
    elif dataset_id == 'hyperthyroidism' and feature == 29:
      continue
    elif dataset_id == 'cervical_cancer' and feature == 'Biopsy':
      continue
    elif dataset_id == 'liver_cancer' and feature == 10:
      continue

    col = dataset[feature]

    if is_numeric_dtype( col.dtype ):
      if col.isna().all():
        fvalues.append( set() )
      else:
        fvalues.append( ( float( col.min() ), float( col.mean() ), float( col.max() ) ) )
    else:
      fvalues.append( set( col.astype(str) ) - set( [str(np.nan)] ) )
  
  #print(fvalues)
  # End CODE
  return fvalues

# Problem D [0 points]
def outcome_values(dataset_id, dataset):
  values = set()
  # Begin CODE
  outcome_col = { 'breast_cancer': 0, 'hyperthyroidism': 29, 'cervical_cancer': -1, 'liver_cancer': -1 }

  col = dataset.iloc[::,outcome_col[dataset_id]]

  if dataset_id == 'hyperthyroidism':
    col = col.str.replace(r'\.\|.*', '', )
    dataset[outcome_col[dataset_id]] = col

  #print(dataset[outcome_col[dataset_id]])

  values = set( col.astype(str) )
  #print(values)
  # End CODE
  return values

# Problem E [0 points]
def outcomes(dataset_id, instances):
  outcomes = []
  # Begin CODE

  outcome_col = { 'breast_cancer': 0, 'hyperthyroidism': -1, 'cervical_cancer': -1, 'liver_cancer': -1 }
  y = np.array(instances)
  y = y.T[outcome_col[dataset_id]]
  #print(y.dtype)

  if y.dtype == np.float64:
    y = y.astype(int)

  outcomes = y.astype(str).tolist()
  #print(outcomes)
  # End CODE
  return outcomes

# Problem 1 [10 points]
def data_split(dataset_id, dataset, percent_train):
  split = None
  # Begin CODE

  train, test = train_test_split(dataset.values.tolist(), test_size=1-percent_train, shuffle=False)
  #print(len(train), len(test))

  split = ( train, test )
  # End CODE
  return split

# Problem 2 [10 points]
def baseline(dataset_id, dataset):
  baseline = None
  # Begin CODE
  data = np.array(dataset)
  baseline = str( sp.stats.mode(outcomes(dataset_id, dataset))[0][0] )
  #print(type(baseline))

  # End CODE
  return baseline

def cleanX(dataset_id, train, test):
  outcome_col = { 'breast_cancer': 0, 'hyperthyroidism': 29, 'cervical_cancer': 28, 'liver_cancer': 10 }

  X_train = pd.DataFrame(train).drop(columns=outcome_col[dataset_id])
  X_test = pd.DataFrame(test).drop(columns=outcome_col[dataset_id])

  for col in X_train:
    values = set(X_train[col])
    if 't' in values or 'f' in values:
      X_train[col] = X_train[col].replace('t', 1).replace('f', 0)
      X_train[col] = X_train[col].astype(np.float64)
    if 'M' in values or 'F' in values:
      X_train[col] = X_train[col].replace('M', 1).replace('F', 0)
      X_train[col] = X_train[col].astype(np.float64)
    if 'Male' in values or 'Female' in values:
      X_train[col] = X_train[col].replace('Male', 1).replace('Female', 0)
      X_train[col] = X_train[col].astype(np.float64)

  for col in X_test:
    values = set(X_test[col])
    if 't' in values or 'f' in values:
      X_test[col] = X_test[col].replace('t', 1).replace('f', 0)
      X_test[col] = X_test[col].astype(np.float64)
    if 'M' in values or 'F' in values:
      X_test[col] = X_test[col].replace('M', 1).replace('F', 0)
      X_test[col] = X_test[col].astype(np.float64)
    if 'Male' in values or 'Female' in values:
      X_test[col] = X_test[col].replace('Male', 1).replace('Female', 0)
      X_test[col] = X_test[col].astype(np.float64)

  if dataset_id == 'hyperthyroidism':
    X_train = X_train.drop(columns=[26,27])
    X_test = X_test.drop(columns=[26,27])
    X_train = pd.get_dummies(X_train, columns=[28])
    X_test = pd.get_dummies(X_test, columns=[28])

  for col in X_test:
    if X_test[col].dtype == object:
      print(X_test[col])

  imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
  imputer.fit(X_train)
  X_train = imputer.transform(X_train)
  X_test = imputer.transform(X_test)

  return (X_train, X_test)

# Problem 3 [15 points]
def decision_tree(dataset_id, train, test):
  predictions = []
  # Begin CODE

  y_train = outcomes(dataset_id, train)

  X_train, X_test = cleanX(dataset_id, train, test)

  model = DecisionTreeClassifier(max_depth=5, random_state=1)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test).tolist()

  # End CODE
  return predictions

# Problem 4 [15 points]
def knn(dataset_id, train, test):
  predictions = []
  # Begin CODE

  y_train = outcomes(dataset_id, train)

  X_train, X_test = cleanX(dataset_id, train, test)

  model =  KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test).tolist()
  #print(predictions)
  # End CODE
  return predictions

# Problem 5 [15 points]
def naive_bayes(dataset_id, train, test):
  predictions = []
  # Begin CODE

  y_train = outcomes(dataset_id, train)

  X_train, X_test = cleanX(dataset_id, train, test)

  model =  GaussianNB()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test).tolist()
  #print(predictions)
  # End CODE
  return predictions

# Problem 6 [15 points]
def svm(dataset_id, train, test):
  predictions = []
  # Begin CODE
  y_train = outcomes(dataset_id, train)

  X_train, X_test = cleanX(dataset_id, train, test)

  model = SVC( C=1, kernel='rbf', gamma=2, random_state=1)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test).tolist()
  #print(predictions)

  # End CODE
  return predictions

# Problem 7 [10 points]
def evaluate(dataset_id, gold, predictions):
  evaluation = {}
  # Begin CODE

  evaluation['accuracy'] = float( sklearn.metrics.accuracy_score(gold, predictions) )
  for label in set(gold)|set(predictions):
    precision, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(gold, predictions, labels=[label], average=None)
    f1 = sklearn.metrics.f1_score(gold, predictions, labels=[label], average=None)
    #print(precision, recall)
    evaluation[label] = {}
    evaluation[label]['precision'] = float( precision )
    evaluation[label]['recall'] = float( recall )
    evaluation[label]['f1'] = float( f1 )
  
  #pp(evaluation, indent=4, compact=True)

  # End CODE
  return evaluation

# Problem 8 [10 points]
def learning_curve(dataset_id, train_sets, test, class_func):
  accuracies = []
  data_lengths = []
  # Begin CODE
  y_test = outcomes(dataset_id, test)
  for train_set in train_sets:
    predictions = class_func(dataset_id, train_set, test)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    accuracies.append( float(accuracy) )
    data_lengths.append(len(train_set))
  #print(accuracies)
  # End CODE
  return accuracies

# Problem 9 [10 points extra]
def visualize(dataset_id, train_sets, test, class_func):
  # Begin CODE
  accuracies = learning_curve(dataset_id, train_sets, test, class_func)
  data_lengths = []
  for train_set in train_sets:
    data_lengths.append(len(train_set))
    
  fig = px.line(x=data_lengths, y=accuracies, labels={'x':'Number of data points in training set', 'y':'Accuracy'}, \
    title=str.title( dataset_id.replace('_', ' ') + ' prediction accuracy using ' + class_func.__name__.replace('_', ' ') ) )
  fig.show()
  # End CODE
  pass

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':
  datasets = ['breast_cancer',\
              'hyperthyroidism',\
              'cervical_cancer',\
              'liver_cancer']
  dims =    {'breast_cancer':(569, 30),
             'hyperthyroidism':(3772, 29),
             'cervical_cancer':(858, 28),
             'liver_cancer':(583,10)}
  targets = {'breast_cancer':set(['B', 'M']),
             'hyperthyroidism':set(['goitre', 'secondary toxic', 'negative', 'T3 toxic', 'hyperthyroid']),
             'cervical_cancer':set(['0', '1']),
             'liver_cancer':set(['1', '2'])}

  for dataset_id in datasets:
    print('::  DATASET: %s ::' % dataset_id)
    print('::: Problem 0-A :::')
    data = read_data(dataset_id)
    util_5353.assert_not_none(data, '0-A')

    print('::: Problem 0-B :::')
    b_ret = dimensions(dataset_id, data)
    util_5353.assert_tuple(b_ret, 2, '0-B')
    util_5353.assert_int(b_ret[0], '0-B')
    util_5353.assert_int(b_ret[1], '0-B')
    util_5353.assert_int_eq(dims[dataset_id][0], b_ret[0], '0-B')
    util_5353.assert_int_eq(dims[dataset_id][1], b_ret[1], '0-B')

    print('::: Problem 0-C :::')
    c_ret = feature_values(dataset_id, data)
    util_5353.assert_list(c_ret, dims[dataset_id][1], '0-C')
    for i in range(len(c_ret)):
      if type(c_ret[i]) == set:
        for item in c_ret[i]:
          util_5353.assert_str(item, '0-C')
      else:
        util_5353.assert_tuple(c_ret[i], 3, '0-C')
        util_5353.assert_float(c_ret[i][0], '0-C')
        util_5353.assert_float(c_ret[i][1], '0-C')
        util_5353.assert_float(c_ret[i][2], '0-C')
    if dataset_id == 'breast_cancer':
      util_5353.assert_float_range((6.980, 6.982), c_ret[0][0], '0-C')
      util_5353.assert_float_range((14.12, 14.13), c_ret[0][1], '0-C')
      util_5353.assert_float_range((28.10, 28.12), c_ret[0][2], '0-C')
      util_5353.assert_float_range((143.4, 143.6), c_ret[3][0], '0-C')
      util_5353.assert_float_range((654.8, 654.9), c_ret[3][1], '0-C')
      util_5353.assert_float_range((2500., 2502.), c_ret[3][2], '0-C')
    
    print('::: Problem 0-D :::')
    d_ret = outcome_values(dataset_id, data)
    util_5353.assert_set(d_ret, '0-D', valid_values=targets[dataset_id])
    
    print('::: Problem 1 :::')
    one_ret = data_split(dataset_id, data, 0.6)
    util_5353.assert_tuple(one_ret, 2, '1')
    util_5353.assert_list(one_ret[0], None, '1')
    util_5353.assert_list(one_ret[1], None, '1')
    if dataset_id == 'breast_cancer':
      util_5353.assert_list(one_ret[0], 341, '1')
    if dataset_id == 'cervical_cancer':
      util_5353.assert_list(one_ret[0], 514, '1')
    train = one_ret[0]
    test  = one_ret[1]
    
    print('::: Problem 0-E :::')
    train_out = outcomes(dataset_id, train)
    test_out  = outcomes(dataset_id, test)
    util_5353.assert_list(train_out, len(train), '0-E', valid_values=targets[dataset_id])
    util_5353.assert_list(test_out,  len(test),  '0-E', valid_values=targets[dataset_id])
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('M', train_out[0], '0-E')
      util_5353.assert_str_eq('B', test_out[-1], '0-E')
    
    print('::: Problem 2 :::')
    two_ret = baseline(dataset_id, data)
    util_5353.assert_str(two_ret, '2')
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('B', two_ret, '2')
    
    print('::: Problem 3 :::')
    three_ret = decision_tree(dataset_id, train, test)
    util_5353.assert_list(three_ret, len(test), '3')

    print('::: Problem 4 :::')
    four_ret = knn(dataset_id, train, test)
    util_5353.assert_list(four_ret, len(test), '4')

    print('::: Problem 5 :::')
    five_ret = naive_bayes(dataset_id, train, test)
    util_5353.assert_list(five_ret, len(test), '5')

    print('::: Problem 6 :::')
    six_ret = svm(dataset_id, train, test)
    util_5353.assert_list(six_ret, len(test), '6')

    print('::: Problem 7 :::')
    seven_ret_dt = evaluate(dataset_id, test_out, three_ret)
    seven_ret_kn = evaluate(dataset_id, test_out, four_ret)
    seven_ret_nb = evaluate(dataset_id, test_out, five_ret)
    seven_ret_sv = evaluate(dataset_id, test_out, six_ret)

    for seven_ret in [seven_ret_dt, seven_ret_kn, seven_ret_nb, seven_ret_sv]:
      util_5353.assert_dict(seven_ret, '7')
      util_5353.assert_dict_key(seven_ret, 'accuracy', '7')
      util_5353.assert_float(seven_ret['accuracy'], '7')
      util_5353.assert_float_range((0.0, 1.0), seven_ret['accuracy'], '7')
      for target in targets[dataset_id]:
        util_5353.assert_dict_key(seven_ret, target, '7')
        util_5353.assert_dict(seven_ret[target], '7')
        util_5353.assert_dict_key(seven_ret[target], 'precision', '7')
        util_5353.assert_dict_key(seven_ret[target], 'recall', '7')
        util_5353.assert_dict_key(seven_ret[target], 'f1', '7')
        util_5353.assert_float(seven_ret[target]['precision'], '7')
        util_5353.assert_float(seven_ret[target]['recall'], '7')
        util_5353.assert_float(seven_ret[target]['f1'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['precision'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['recall'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['f1'], '7')

    print('::: Problem 8 :::')
    train_sets = []
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      train_sets.append(train[:int(percent*len(train))])
    eight_ret_dt = learning_curve(dataset_id, train_sets, test, decision_tree)
    eight_ret_kn = learning_curve(dataset_id, train_sets, test, knn)
    eight_ret_nb = learning_curve(dataset_id, train_sets, test, naive_bayes)
    eight_ret_sv = learning_curve(dataset_id, train_sets, test, svm)
    for eight_ret in [eight_ret_dt, eight_ret_kn, eight_ret_nb, eight_ret_sv]:
      util_5353.assert_list(eight_ret, len(train_sets), '8')
      for i in range(len(eight_ret)):
        util_5353.assert_float(eight_ret[i], '8')
        util_5353.assert_float_range((0.0, 1.0), eight_ret[i], '8')

  print('~~~ All Tests Pass ~~~')



