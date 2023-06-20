# -*- coding: utf-8 -*-

import uuid
import artm
import xlwt
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.corpus import stopwords
import string
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
training data are provided by:
- Yelp Open Dataset: https://www.yelp.com/dataset
- Amazon review dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
Testing data with annotation are provided by:
- Laptop-ACOS and Restaurant-ACOS: https://github.com/NUSTM/ACOS/tree/main/data
'''

train_dir = '../data/laptop/laptop_offline.txt'
test_dir = '../data/laptop/laptop_online.txt'
'''
train_dir = '../data/restaurant/restaurant_offline.txt'
test_dir = '../data/restaurant/restaurant_online.txt'
'''


# load training data
with open(train_dir,encoding='utf-8') as fcorpus:
	reviews = fcorpus.read().splitlines()

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
    if len(doc)>0:
        docs.append(doc)

dct = Dictionary(docs)      
dct.filter_extremes(no_below=50)
dct.compactify() 
corpus = []
# pack into corpus
for i in range(len(docs)):
    words = [word for word in docs[i] if word in dct.token2id.keys()]
    if len(words)>0 and len(words)<=50:
        corpus.append(words)

# write training data in Bag-Of-Words with UCI format
c_train = [dct.doc2bow(_) for _ in corpus]
total_pos = 0
for d in c_train:
    total_pos += len(d)
write_file = open('./experiment/ARTM/docword.laptop.txt', 'w')
write_file.write(str(len(c_train))+"\n")
write_file.write(str(len(dct))+"\n")
write_file.write(str(total_pos)+"\n")
for i in range(len(c_train)):
    for pair in c_train[i]:
        w = pair[0]+1
        count = pair[1]
        write_file.write(str(i+1)+" "+str(w)+" "+str(count)+"\n")
write_file.close()

write_file = open('./experiment/ARTM/vocab.laptop.txt', 'w')
for i in range(len(dct)):
    write_file.write(dct[i]+"\n")
write_file.close()

# training batch
batch_vectorizer = artm.BatchVectorizer(data_path='./experiment/ARTM/',data_format='bow_uci',collection_name='laptop',target_folder='./experiment/ARTM/laptop_batches')
vocab = batch_vectorizer.dictionary
vocab = vocab._master.get_dictionary(vocab._name)
background_density = [vocab.token_value[i] for i in range(len(vocab.token))]
background_density = np.array(background_density)


# Model training
'''
## ARTM training
num_t = 90
artm_model = artm.ARTM(num_topics=num_t, dictionary=batch_vectorizer.dictionary, cache_theta=False, seed = 1)
artm_model.scores.add(artm.PerplexityScore(name='perplexity_score',dictionary=batch_vectorizer.dictionary))
# regularizers
smooth_topics = ['topic_'+str(i) for i in range(0,10)]
specified_topics = ['topic_'+str(i) for i in range(10,num_t)]
artm_model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer', topic_names=specified_topics, tau=-0.1))
artm_model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_phi_regularizer', topic_names=smooth_topics, tau=0.1))
artm_model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer', topic_names=specified_topics, tau=-0.1))
artm_model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='smooth_theta_regularizer', topic_names=smooth_topics, tau=0.1))

artm_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)
print(artm_model.score_tracker['perplexity_score'].value)

phi = artm_model.phi_
specialized_score = np.zeros(num_t)
for i in range(num_t):  
    phi_t = phi.iloc[:,i].values
    specialized_score[i] = 1-(cosine_similarity(phi_t.reshape(1,-1),background_density.reshape(1,-1))[0][0])
print("avg_specialized_score: ",np.mean(specialized_score))
'''

## hARTM training
hier_model = artm.hARTM(dictionary=batch_vectorizer.dictionary, seed = 1)
# first level
level0 = hier_model.add_level(num_topics=1) 
level0.initialize(dictionary=batch_vectorizer.dictionary)
level0.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
# second level
level1 = hier_model.add_level(num_topics=10, parent_level_weight=1)
level1.initialize(dictionary=batch_vectorizer.dictionary)
level1.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)
# third level
level2 = hier_model.add_level(num_topics=80, parent_level_weight=1)
level2.initialize(dictionary=batch_vectorizer.dictionary)
level2.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer2', tau=-0.1))
level2.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer2', tau=-0.1))
level2.regularizers.add(artm.HierarchySparsingThetaRegularizer(name='sparse_hier_regularizer2',tau=1.0))
level2.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)

hier_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)
phi0 = level0.phi_
phi1 = level1.phi_
phi2 = level2.phi_
phi = hier_model.get_phi()
specialized_score = np.zeros(phi.shape[1])
for i in range(phi.shape[1]):  
    phi_t = phi.iloc[:,i].values
    specialized_score[i] = 1-(cosine_similarity(phi_t.reshape(1,-1),background_density.reshape(1,-1))[0][0])
print("avg_specialized_score: ",np.mean(specialized_score))



# load test data
reviews = []
with open(test_dir,encoding='utf-8') as fcorpus:
	raw_reviews = fcorpus.read().splitlines()
reviews = [content.split('\t')[0] for content in raw_reviews]
labels = [int(content.split('\t')[1]) for content in raw_reviews]

test_docs = []
for i in range(len(reviews)):  
    doc = reviews[i]
    # strip punctuations and digits
    doc = doc.translate(replace) 
    doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars
    doc = nltk.word_tokenize(doc)
    doc = [w.lower() for w in doc]
    doc = [w for w in doc if w not in stopset]
    doc = [stemmer.stem(w) for w in doc]  
    doc = [w for w in doc if w in dct.token2id.keys()]  
    test_docs.append(doc)

V = [vocab.token[i] for i in range(len(vocab.token))]
batch = artm.messages.Batch()
batch.id = str(uuid.uuid4())
dictionary = {}

# first step: fill the general batch vocabulary
for i, token in enumerate(V):
    batch.token.append(token)
    dictionary[token] = i

# second step: fill the items
for doc in test_docs:
    item = batch.item.add()

    local_dict = {}
    for token in doc:
        if not token in local_dict:
            local_dict[token] = 0
        local_dict[token] += 1
    for k, v in local_dict.items():
        item.token_id.append(dictionary[k])
        item.token_weight.append(v)

# testing batch
with open('./experiment/ARTM/laptop_test_batches/laptop_test.batch', 'wb') as fout:
    fout.write(batch.SerializeToString())
batch_vectorizer_test = artm.BatchVectorizer(data_path='./experiment/ARTM/laptop_test_batches/',data_format='batches',batches=["laptop_test.batch"])



# Model testing
'''
## ARTM testing
theta_test = artm_model.transform(batch_vectorizer=batch_vectorizer_test)
'''
## hARTM testing
theta_test = hier_model.transform(batch_vectorizer=batch_vectorizer_test)

# compute score
scores = []
for i in range(theta_test.shape[1]):
    score = 0
    for j in range(theta_test.shape[0]):
        score += theta_test.iloc[j,i]*specialized_score[j]
    scores.append(score)
auc_value = 1-roc_auc_score(labels,scores)
print(auc_value)



# save scores for later classification    
f = xlwt.Workbook('encoding = utf-8') 
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) 
sheet1.write(0,0,'hARTM') 
for i in range(len(scores)):
    sheet1.write(i+1,0,scores[i]) 
f.save('laptop_test_score_hARTM.xls')




