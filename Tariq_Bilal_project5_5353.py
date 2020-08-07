import gzip
import math
import re
import numpy as np
from scipy import sparse
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import metrics

from collections import Counter

import util_5353

# Problem A [0 points]
def read_data(filenames):
  data = None
  # Begin CODE
  data = {}
  for i,filename in enumerate(filenames):
    with gzip.open(filename, 'r') as file:
      text = str( file.read(), 'utf-8' )
      articles = text.split('\n\n')
      for article in articles:
        cleanarticle = []
        for line in article.split('\n'):
          if '-' in line[0:5]:
            cleanarticle.append(line.strip())
          else:
            cleanarticle[-1] += f' {line.strip()}'
        cleanarticle = '\n'.join(cleanarticle)
        pmid = get_tag(cleanarticle, 'PMID').strip()
        articletokens = tokenize(get_tag(cleanarticle, 'TI')) + tokenize(get_tag(cleanarticle, 'AB'))
        data[pmid] = (cleanarticle, articletokens)
  # End CODE
  return data

# Problem B [0 points]
tokenizer = re.compile('\w+|[^\s\w]+')
def tokenize(text):
  return tokenizer.findall(text.lower())

# Problem C [0 points]
def pmids(data):
  pmids = []
  # Begin CODE
  pmids = list(data.keys())
  # End CODE
  return pmids

# Extracts text of the first occurrence of a 4-letter tag given the document text
def get_tag(text, tag):
  if len(tag) > 4:
    return ''

  pattern = f'^{tag} {{{4-len(tag)}}}- '+ r'([^\n]+)$'
  #print(pattern)
  tagtext = re.search(pattern, text, flags=re.MULTILINE).group(1)
  #print(tagtext)
  return tagtext

# Extracts text of all occurences of a tag given the document text
def get_all_tags(text, tag):
  if len(tag) > 4:
    return ''
  #print(text)
  pattern = f'^{tag} {{{4-len(tag)}}}- '+ r'([^\n]+)$'
  #print(pattern)
  tagtexts = re.findall(pattern, text, flags=re.MULTILINE)
  #print(tagtexts)
  return tagtexts

# Problem 1 [10 points]
def unigrams(data, pmid):
  unigrams = {}
  # Begin CODE
  article = data[pmid]
  tokens = article[1]
  for token in set(tokens):
    unigrams[token] = 1.0
  #print(len(unigrams))
  # End CODE
  return unigrams

# Problem 2 [10 points]
def tfidf(data, pmid):
  tfidf = {}
  # Begin CODE

  article = data[pmid]
  tokens = article[1]
  counts = Counter(tokens)
  for token in set(tokens):
    tf = counts[token]
    N = len(data)
    F = 0
    for pmid in data:
      doctokens = data[pmid][1]
      if token in doctokens:
        F += 1
    tfidf[token] = tf * math.log(N/F)

  #print(tfidf)

  # End CODE
  return tfidf

# Problem 3 [10 points]
def mesh(data, pmid):
  mesh = []
  # Begin CODE
  article = data[pmid]
  mesh_text = get_all_tags(article[0], 'MH')
  terms = map(lambda x: x.split('/')[0].strip('*'), mesh_text)
  mesh += terms
  # End CODE
  return mesh

def get_features(feature_dicts, sorted_features):
  X = sparse.dok_matrix((len(feature_dicts), len(sorted_features)), dtype=np.float64)
  for i,feature_dict in enumerate(feature_dicts):
    for feature in feature_dict:
      if feature in sorted_features:
        j = sorted_features.index(feature)
        X[i,j] = feature_dict[feature]

  return X

def prep_data(data, train, test, meshterm, func):
  train_dicts = [ func(data, pmid) for pmid in train ]
  test_dicts = [ func(data, pmid) for pmid in test ]

  train_tokens = set()
  for train_dict in train_dicts:
    train_tokens |= train_dict.keys()
  
  train_tokens = sorted(train_tokens)

  X_train = get_features(train_dicts, train_tokens)
  y_train = [meshterm in mesh(data, pmid) for pmid in train]
  #print(any(y_train))
  X_test = get_features(test_dicts, train_tokens)
  return (X_train, y_train, X_test, train_tokens)

# Problem 4 [10 points]
def svm_predict_unigram(data, train, test, mesh):
  predictions = {m:[] for m in mesh}
  # Begin CODE
  for term in mesh:
    X_train, y_train, X_test, tokens = prep_data(data, train, test, term, unigrams)
    print(y_train)
    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    for prediction,pmid in zip(y_test, test):
      #print(prediction, pmid)
      if prediction:
        predictions[term].append(pmid)
  #print(predictions)
  # End CODE
  return predictions

# Problem 5 [10 points]
def svm_predict_tfidf(data, train, test, mesh):
  predictions = {m:[] for m in mesh}
  # Begin CODE
  for term in mesh:
    X_train, y_train, X_test, tokens = prep_data(data, train, test, term, tfidf)
    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    for prediction,pmid in zip(y_test, test):
      #print(prediction, pmid)
      if prediction:
        predictions[term].append(pmid)
  # End CODE
  return predictions

# Problem 6 [10 points]
def kmeans(data, k):
  clusters = {}
  # Begin CODE
  all_unigrams = []
  for pmid in data:
    all_unigrams.append( unigrams(data, pmid) )

  tokens = set()
  for train_dict in all_unigrams:
    tokens |= train_dict.keys()
  
  all_tokens = sorted(tokens)

  X = get_features(all_unigrams, all_tokens)

  clustering = KMeans(n_clusters=k, init='random', random_state=0)
  clustering.fit(X)
  for label,pmid in zip(clustering.labels_, data):
    clusters[pmid] = label
  
  # End CODE
  return clusters

# Problem 7 [10 points]
def svm_predict_cluster(data, train, test, mesh, k):
  predictions = {m:[] for m in mesh}
  # Begin CODE
  for term in mesh:
    cluster_labels = kmeans(data, k)

    X_train = [ cluster_labels[pmid] for pmid in train ]
    y_train = [term in mesh(data, pmid) for pmid in train]

    X_test = [ cluster_labels[pmid] for pmid in test ]
    
    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    for prediction,pmid in zip(y_test, test):
      #print(prediction, pmid)
      if prediction:
        predictions[term].append(pmid)
  # End CODE
  return predictions

# Problem 8 [10 points]
def svm_predict_cluster_unigrams(data, train, test, mesh, k):
  predictions = {m:[] for m in mesh}
  # Begin CODE
  for term in mesh:
    cluster_labels = kmeans(data, k)
    X_train, y_train, X_test, tokens = prep_data(data, train, test, term, unigrams)

    X_train_clusters = [ cluster_labels[pmid] for pmid in train ]
    X_test_clusters = [ cluster_labels[pmid] for pmid in test ]

    X_train.resize( (len(train), len(tokens)+1) )
    X_test.resize( (len(test), len(tokens)+1) )

    j = len(tokens)
    for i,cluster in enumerate(X_train_clusters):
      X_train[i,j] = cluster

    for i,cluster in enumerate(X_test_clusters):
      X_test[i,j] = cluster

    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    for prediction,pmid in zip(y_test, test):
      #print(prediction, pmid)
      if prediction:
        predictions[term].append(pmid)
  # End CODE
  return predictions

# Problem 9 [20 points]
def evaluate(data, test, mesh_predict):
  evaluation = {}
  # Begin CODE
  evaluation['accuracy'] = metrics.accuracy_score(test, mesh_predict)
  evaluation['precision'] = metrics.precision_score(test, mesh_predict)
  evaluation['recall'] = metrics.recall_score(test, mesh_predict)
  evaluation['f1'] = metrics.f1_score(test, mesh_predict)
  print(evaluation)
  # End CODE
  return evaluation

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  # Comment out some file names to speed up the development process, but
  # ultimately you want to uncomment the filenames so you ensure that your code
  # works will all files.  The assertions below assume that medline.0.txt.gz is
  # in the list.
  file_list = []
  file_list.append('medline.0.txt.gz')
  file_list.append('medline.1.txt.gz')
  file_list.append('medline.2.txt.gz')
  file_list.append('medline.3.txt.gz')
  file_list.append('medline.4.txt.gz')
  file_list.append('medline.5.txt.gz')
  file_list.append('medline.6.txt.gz')
  file_list.append('medline.7.txt.gz')
  file_list.append('medline.8.txt.gz')
  file_list.append('medline.9.txt.gz')
  
  pmid_list = ['22999938', '23010078', '23018989']

  print('::: Problem A :::')
  data = read_data(file_list)

  print('::: Problem C :::')
  pmids = pmids(data)
  for pmid in pmid_list:
    if pmid not in pmids:
      util_5353.die('C', 'Assertions assume PMID is present: %s', pmid)

  tts = int(len(pmids) * 0.8)
  train = pmids[:tts]
  test = pmids[tts:]

  print('::: Problem 1 :::')
  one_ret = unigrams(data, pmid_list[0])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(99, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['metastasis'], '1')
  one_ret = unigrams(data, pmid_list[1])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(95, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['destruction'], '1')
  one_ret = unigrams(data, pmid_list[2])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(133, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['concurrent'], '1')

  print('::: Problem 2 :::')
  two_ret = tfidf(data, pmid_list[0])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(99, len(two_ret), '2')
  util_5353.assert_float_range((1.5, 3.0), two_ret['metastasis'], '2')
  two_ret = tfidf(data, pmid_list[1])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(95, len(two_ret), '2')
  util_5353.assert_float_range((10.0, 20.0), two_ret['destruction'], '2')
  two_ret = tfidf(data, pmid_list[2])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(133, len(two_ret), '2')
  util_5353.assert_float_range((7.0, 10.0), two_ret['concurrent'], '2')

  print('::: Problem 3 :::')
  three_ret = mesh(data, pmid_list[0])
  GOLD = ['Animals', 'Breast Neoplasms', 'DNA Methylation', 'DNA, Neoplasm', 'DNA-Binding Proteins', 'Dioxygenases', 'Down-Regulation', 'Female', 'Gene Expression Regulation, Neoplastic', 'Humans', 'Male', 'Mice', 'Mice, Inbred BALB C', 'Mice, Nude', 'Mixed Function Oxygenases', 'Neoplasm Invasiveness', 'Prostatic Neoplasms', 'Proto-Oncogene Proteins', 'Tissue Inhibitor of Metalloproteinase-2', 'Tissue Inhibitor of Metalloproteinase-3', 'Tumor Suppressor Proteins']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)
  three_ret = mesh(data, pmid_list[1])
  GOLD = ['Animals', 'Contrast Media', 'Gene Knockdown Techniques', 'Genetic Therapy', 'Mice', 'Mice, Inbred C3H', 'Microbubbles', 'Neoplasms, Squamous Cell', 'RNA, Small Interfering', 'Receptor, Epidermal Growth Factor', 'Sonication', 'Transfection', 'Ultrasonics', 'Ultrasonography']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)
  three_ret = mesh(data, pmid_list[2])
  GOLD = ['Adult', 'Aged', 'Chemoradiotherapy', 'Diffusion Magnetic Resonance Imaging', 'Female', 'Humans', 'Medical Oncology', 'Middle Aged', 'Reproducibility of Results', 'Time Factors', 'Treatment Outcome', 'Tumor Burden', 'Uterine Cervical Neoplasms']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)

  print('::: Problem 4 :::')
  mesh_list = ['Humans', 'Female', 'Male', 'Animals', 'Treatment Outcome',
               'Neoplasms', 'Prognosis', 'Risk Factors', 'Breast Neoplasms', 'Lung Neoplasms']
  mesh_set = set()
  for pmid in pmids:
    mesh_set.update(mesh(data, pmid))
  for m in mesh_list:
    if m not in mesh_set:
      util_5353.die('4', 'Assertions assume MeSH term is present: %s', m)
  four_ret = svm_predict_unigram(data, train, test, mesh_list)
  util_5353.assert_dict(four_ret, '4')
  for m in mesh_list:
    util_5353.assert_dict_key(four_ret, m, '4')
    util_5353.assert_list(four_ret[m], None, '4', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(four_ret[m]), '4')
  util_5353.assert_int_range((len(test)/2, len(test)), len(four_ret['Humans']), '4')

  print('::: Problem 5 :::')
  five_ret = svm_predict_tfidf(data, train, test, mesh_list)
  util_5353.assert_dict(five_ret, '5')
  for m in mesh_list:
    util_5353.assert_dict_key(five_ret, m, '5')
    util_5353.assert_list(five_ret[m], None, '5', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(five_ret[m]), '5')
  util_5353.assert_int_range((len(test)/2, len(test)), len(five_ret['Humans']), '5')

  print('::: Problem 6 :::')
  K = 10
  six_ret = kmeans(data, K)
  util_5353.assert_dict(six_ret, '6')
  util_5353.assert_int_eq(len(pmids), len(six_ret), '6')
  for pmid in pmids:
    util_5353.assert_dict_key(six_ret, pmid, '6')
    util_5353.assert_int_range((0, K-1), six_ret[pmid], '6')

  print('::: Problem 7 :::')
  seven_ret = svm_predict_cluster(data, train, test, mesh_list, K)
  util_5353.assert_dict(seven_ret, '7')
  for m in mesh_list:
    util_5353.assert_dict_key(seven_ret, m, '7')
    util_5353.assert_list(seven_ret[m], None, '7', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(seven_ret[m]), '7')
  util_5353.assert_int_range((len(test)/2, len(test)), len(seven_ret['Humans']), '7')

  print('::: Problem 8 :::')
  eight_ret = svm_predict_cluster_unigrams(data, train, test, mesh_list, K)
  util_5353.assert_dict(eight_ret, '8')
  for m in mesh_list:
    util_5353.assert_dict_key(eight_ret, m, '8')
    util_5353.assert_list(eight_ret[m], None, '8', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(eight_ret[m]), '8')
  util_5353.assert_int_range((len(test)/2, len(test)), len(eight_ret['Humans']), '8')

  print(':: Problem 9 ::')
  nine_ret4 = evaluate(data, test, four_ret)
  nine_ret5 = evaluate(data, test, five_ret)
  nine_ret7 = evaluate(data, test, seven_ret)
  nine_ret8 = evaluate(data, test, eight_ret)
  for nine_ret in [nine_ret4, nine_ret5, nine_ret7, nine_ret8]:
    util_5353.assert_dict(nine_ret, '9')
    for m in mesh_list:
      util_5353.assert_dict_key(nine_ret, m, '9')
      util_5353.assert_dict(nine_ret[m], '9')
      for k in ['accuracy', 'precision', 'recall', 'f1']:
        util_5353.assert_dict_key(nine_ret[m], k, '9')
        util_5353.assert_float(nine_ret[m][k], '9')
        util_5353.assert_float_range((0.0, 1.0), nine_ret[m][k], '9')

  print('~~~ All Tests Pass ~~~')
