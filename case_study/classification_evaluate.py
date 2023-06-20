# -*- coding: utf-8 -*-


# import modules & set up logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef

'''
input_dir = '../data/laptop/laptop_online.txt'
classfication_score = '../data/laptop/laptop_test_score.xlsx'
'''
input_dir = '../data/restaurant/restaurant_online.txt'
classfication_score = '../data/restaurant/restaurant_test_score.xlsx'


# load_data
raw_reviews = []
with open(input_dir,encoding='utf-8') as fcorpus:
	raw_reviews = fcorpus.read().splitlines()
reviews = [content.split('\t')[0] for content in raw_reviews]
labels = [int(content.split('\t')[1]) for content in raw_reviews]


# preprocess data
stemmer = PorterStemmer()
stopset = stopwords.words('english') + ['will', 'also', 'said']
del_str = string.punctuation + string.digits
replace = str.maketrans(del_str,' '*len(del_str))
docs = []
for i in range(len(reviews)):  
    doc = reviews[i]
    # strip punctuations and digits
    doc = doc.translate(replace) 
    doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars
    doc = nltk.word_tokenize(doc)
    doc = [w.lower() for w in doc]
    doc = [w for w in doc if w not in stopset]
    doc = [stemmer.stem(w) for w in doc]
    docs.append(' '.join(doc))

# 5-Fold cross validation for naive bayes classifier
kf = KFold(n_splits=5)
nb_auc = []
nb_F1 =[]
nb_MCC = []
for train_id, test_id in kf.split(docs):
    train_docs = np.array(docs)[train_id]
    train_labels = np.array(labels)[train_id]
    test_docs = np.array(docs)[test_id]
    test_labels = np.array(labels)[test_id]
    # fit a naive bayes classifier
    nb = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    nb.fit(train_docs, train_labels)
    test_pred = nb.predict(test_docs)
    # prediction metrics
    nb_score = nb.predict_proba(test_docs)[:, 1]
    nb_auc.append(roc_auc_score(test_labels, nb_score))
    precision, recall, thresholds = precision_recall_curve(test_labels, nb_score)    
    nb_F1.append(max(2*precision*recall/(precision+recall)))
    MCC = []
    for t in thresholds:
        y_pred = (nb_score > t).astype(int) 
        MCC.append(matthews_corrcoef(test_labels,y_pred))
    nb_MCC.append(max(MCC))

nb_auc = np.mean(nb_auc) 
nb_F1 = np.mean(nb_F1)    
nb_MCC = np.mean(nb_MCC)       
print("nb_AUC: ",nb_auc,"nb_F1: ",nb_F1,"nb_MCC: ",nb_MCC)
    

# load classification scores of test set by different models
scores = pd.read_excel(io = classfication_score)
proposed_score = 1-scores["proposed"]
rCRP_score = 1-scores["rCRP"]
GSDPMM_score = 1-scores["GSDPMM"]
ARTM_score = 1-scores["ARTM"]
nCRP_score = 1-scores["nCRP"]
hARTM_score = 1-scores["hARTM"]




# NB
fpr, tpr, threshold = metrics.roc_curve(test_labels, nb_score)
line0, = plt.plot(fpr, tpr, color = 'olive', lw = 2,ls = "dashdot")

# proposed
proposed_auc = roc_auc_score(labels,proposed_score)
fpr, tpr, threshold = metrics.roc_curve(labels, proposed_score)
line1, = plt.plot(fpr, tpr, color = 'orange', lw = 2,ls="-")
'''
precision, recall, thresholds = precision_recall_curve(labels, proposed_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (proposed_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("proposed_AUC: ",proposed_auc, "proposed_F1: ", F1, "proposed_MCC: ", max(MCC))
'''
# rCRP
rCRP_auc = roc_auc_score(labels,rCRP_score)
fpr, tpr, threshold = metrics.roc_curve(labels, rCRP_score)
line2, = plt.plot(fpr, tpr, color = 'firebrick', lw = 2,ls="dotted")
'''
precision, recall, thresholds = precision_recall_curve(labels, rCRP_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (rCRP_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("rCRP_AUC: ",rCRP_auc, "rCRP_F1: ", F1, "rCRP_MCC: ", max(MCC))
'''
# nCRP
nCRP_auc = roc_auc_score(labels,nCRP_score)
fpr, tpr, threshold = metrics.roc_curve(labels, nCRP_score)
line3, = plt.plot(fpr, tpr, color = 'green', lw = 2,ls="dashed")
'''
precision, recall, thresholds = precision_recall_curve(labels, nCRP_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (nCRP_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("nCRP_AUC: ",nCRP_auc, "nCRP_F1: ", F1, "nCRP_MCC: ", max(MCC))
'''
# GSDPMM
GSDPMM_auc = roc_auc_score(labels,GSDPMM_score)
fpr, tpr, threshold = metrics.roc_curve(labels, GSDPMM_score)
line4, = plt.plot(fpr, tpr, color = 'blue', lw = 2,ls=":")
'''
precision, recall, thresholds = precision_recall_curve(labels, GSDPMM_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (GSDPMM_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("GSDPMM_AUC: ",GSDPMM_auc, "GSDPMM_F1: ", F1, "GSDPMM_MCC: ", max(MCC))
'''
# ARTM
ARTM_auc = roc_auc_score(labels,ARTM_score)
fpr, tpr, threshold = metrics.roc_curve(labels, ARTM_score)
line5, = plt.plot(fpr, tpr, color = 'grey', lw = 2,ls="dashdot")
'''
precision, recall, thresholds = precision_recall_curve(labels, ARTM_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (ARTM_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("ARTM_AUC: ",ARTM_auc, "ARTM_F1: ", F1, "ARTM_MCC: ", max(MCC))
'''
# hARTM
hARTM_auc = roc_auc_score(labels,hARTM_score)
fpr, tpr, threshold = metrics.roc_curve(labels, hARTM_score)
line6, = plt.plot(fpr, tpr, color = 'blueviolet', lw = 2,ls="dashed")
'''
precision, recall, thresholds = precision_recall_curve(labels, hARTM_score)
F1 = max(2*precision*recall/(precision+recall))
MCC = []
for t in thresholds:
    y_pred = (hARTM_score > t).astype(int)
    MCC.append(matthews_corrcoef(labels,y_pred))
print("hARTM_AUC: ",hARTM_auc, "hARTM_F1: ", F1, "hARTM_MCC: ", max(MCC))
'''


plt.plot([0,1],[0,1], color = 'black', lw =1, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(handles=[line0,line1,line2,line3,line4,line5,line6], labels = ['NB (AUC = %0.3f)' % nb_auc, 'Proposed (AUC = %0.3f)' % proposed_auc,  \
            'rCRP (AUC = %0.3f)' % rCRP_auc, 'nCRP (AUC = %0.3f)' % nCRP_auc, 'GSDPMM (AUC = %0.3f)' % GSDPMM_auc, 'ARTM (AUC = %0.3f)' % ARTM_auc, 'hARTM (AUC = %0.3f)' % hARTM_auc], loc="lower right")
plt.tight_layout()
plt.show()


