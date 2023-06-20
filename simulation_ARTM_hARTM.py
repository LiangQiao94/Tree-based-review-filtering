# -*- coding: utf-8 -*-

import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import artm
import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, roc_auc_score
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



length = 10
alpha = 0.1
train_dir = '../data/synthetic_corpus/'+str(length)+'_'+str(alpha)+'.txt'
test_dir = '../data/synthetic_corpus/'+str(length)+'_'+str(alpha)+'_test.txt'


def hier_likelihood(theta_test,specialized_score,test_docs,phi):
    n = 0
    LL = 0
    for i in range(theta_test.shape[1]):
        doc = test_docs[i]
        n += len(doc)
        p_t = np.zeros(theta_test.shape[0])
        for w in doc:
            p_w = 0        
            for j in range(theta_test.shape[0]):
                p_t[j] = theta_test.iloc[j,i] * phi.iloc[:,j].loc[w]
            p_t = p_t / np.sum(p_t)
            for j in range(theta_test.shape[0]):
                p_w +=  p_t[j] * phi.iloc[:,j].loc[w]
            LL += np.log(p_w)
    return LL/n

def intensity_compute(phi,intensity):
    for t in range(13):
        intensity[t,0,1] = phi.iloc[:,t].loc['1']
        for i in range(1,5):
            for j in range(3):
                v = str(j*4+i+1)
                intensity[t,i,j] = phi.iloc[:,t].loc[v]
    
	



#1 load training data
reviews = []
with open(train_dir,encoding='utf-8') as fcorpus:
	raw_reviews = fcorpus.read().splitlines()
reviews = [content.split('\t')[0] for content in raw_reviews]
docs = []
for doc in reviews:
    doc = nltk.word_tokenize(doc)
    if len(doc)>0:
        docs.append(doc)
dct = Dictionary(docs) 

# write training data in Bag-Of-Words with UCI format
c_train = [dct.doc2bow(_) for _ in docs]
total_pos = 0
for d in c_train:
    total_pos += len(d)
write_file = open('./ARTM/docword.'+str(length)+'_'+str(alpha)+'.txt', 'w')
write_file.write(str(len(c_train))+"\n")
write_file.write(str(len(dct))+"\n")
write_file.write(str(total_pos)+"\n")
for i in range(len(c_train)):
    for pair in c_train[i]:
        w = pair[0]+1
        count = pair[1]
        write_file.write(str(i+1)+" "+str(w)+" "+str(count)+"\n")
write_file.close()
write_file = open('./ARTM/vocab.'+str(length)+'_'+str(alpha)+'.txt', 'w')
for i in range(len(dct)):
    write_file.write(dct[i]+"\n")
write_file.close()

# generate training batch
batch_vectorizer = artm.BatchVectorizer(data_path='./ARTM/',data_format='bow_uci',collection_name=str(length)+'_'+str(alpha),target_folder='./ARTM/'+str(length)+'_'+str(alpha)+'_batches')
vocab = batch_vectorizer.dictionary
vocab = vocab._master.get_dictionary(vocab._name)
background_density = [vocab.token_value[i] for i in range(len(vocab.token))]
background_density = np.array(background_density)




#2 load test data
reviews = []
with open(test_dir,encoding='utf-8') as fcorpus:
	raw_reviews = fcorpus.read().splitlines()
reviews = [content.split('\t')[0] for content in raw_reviews]
labels1 = [content.split('\t')[1] for content in raw_reviews]
labels2 = [content.split('\t')[2] for content in raw_reviews]
test_docs = []
for i in range(len(reviews)):  
    doc = reviews[i]
    doc = nltk.word_tokenize(doc) 
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

# generate testing batch
with open('./ARTM/test_batches/'+str(length)+'_'+str(alpha)+'_test.batch', 'wb') as fout:
    fout.write(batch.SerializeToString())
batch_vectorizer_test = artm.BatchVectorizer(data_path='./ARTM/test_batches/',data_format='batches',batches=[str(length)+'_'+str(alpha)+'_test.batch'])




#3 model validation
auc1 = []
auc2 = []
L = []
replicate = 10
np.random.seed(10)
for iteration in range(replicate):
    seed0 = np.random.randint(100)
    
    # ARTM validation
    ## ARTM training
    num_t = 13
    artm_model = artm.ARTM(num_topics=num_t, dictionary=batch_vectorizer.dictionary, cache_theta=False, seed=seed0)
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score',dictionary=batch_vectorizer.dictionary))
    # regularizers
    smooth_topics = ['topic_'+str(i) for i in range(0,4)]
    specified_topics = ['topic_'+str(i) for i in range(4,num_t)]
    artm_model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer', topic_names=specified_topics, tau=-0.5))
    artm_model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_phi_regularizer', topic_names=smooth_topics, tau=0.5))
    artm_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    phi = artm_model.phi_
    specialized_score = np.zeros(num_t)
    for i in range(num_t):  
        phi_t = phi.iloc[:,i].values
        specialized_score[i] = 1-(cosine_similarity(phi_t.reshape(1,-1),background_density.reshape(1,-1))[0][0])
    
    ## ARTM testing
    theta_test = artm_model.transform(batch_vectorizer=batch_vectorizer_test)
    perplexity = artm_model.get_score('perplexity_score').value
    likelihood = -1 * np.log(perplexity)
    '''
    
    # hARTM validation
    ## hARTM training
    hier_model = artm.hARTM(dictionary=batch_vectorizer.dictionary,seed=seed0)
    #hier_model.scores.add(artm.PerplexityScore(name='perplexity_score',dictionary=batch_vectorizer.dictionary))
    # first level
    level0 = hier_model.add_level(num_topics=1) 
    level0.initialize(dictionary=batch_vectorizer.dictionary)
    level0.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    # second level
    level1 = hier_model.add_level(num_topics=3, parent_level_weight=1)
    level1.initialize(dictionary=batch_vectorizer.dictionary)
    level1.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer', tau=-0.1))
    level1.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer', tau=-0.1))
    level1.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)
    # third level
    level2 = hier_model.add_level(num_topics=9, parent_level_weight=1)
    level2.initialize(dictionary=batch_vectorizer.dictionary)
    level2.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer2', tau=-0.1))
    level2.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer2', tau=-0.1))
    level2.regularizers.add(artm.HierarchySparsingThetaRegularizer(name='sparse_hier_regularizer',tau=10.0))
    level2.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=20)
    #hier_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    phi0 = level0.phi_
    phi1 = level1.phi_
    phi2 = level2.phi_
    phi = hier_model.get_phi()
    specialized_score = np.zeros(phi.shape[1])
    for i in range(phi.shape[1]):  
        phi_t = phi.iloc[:,i].values
        specialized_score[i] = 1-(cosine_similarity(phi_t.reshape(1,-1),background_density.reshape(1,-1))[0][0])
    print("avg_specialized_score: ",np.mean(specialized_score))
    psi = level2.get_psi()
    
    ## hARTM testing
    theta_test = hier_model.transform(batch_vectorizer=batch_vectorizer_test)
    likelihood = hier_likelihood(theta_test, specialized_score, test_docs, phi)
    '''
    
    # compute score
    scores = []
    for i in range(theta_test.shape[1]):
        score = 0
        for j in range(theta_test.shape[0]):
            score += theta_test.iloc[j,i]*specialized_score[j]
        scores.append(score)
    
    L.append(likelihood)
    auc1.append(1-roc_auc_score(labels1,scores))
    auc2.append(1-roc_auc_score(labels2,scores))
    

auc1 = np.array(auc1)
auc2 = np.array(auc2)
L = np.array(L)
print(np.mean(auc1),np.std(auc1)/np.sqrt(replicate))
print(np.mean(auc2),np.std(auc2)/np.sqrt(replicate))
print(np.mean(L),np.std(L)/np.sqrt(replicate))



# visualize results - heatmap for topic distribution over vocabulary 
# only applicable to hierarchical model
'''
mask = np.array([False]*15)
mask = mask.reshape(5, 3)
mask[0,0] = True
mask[0,2] = True

fig, ax0 = plt.subplots(5, 3,figsize=(10,10))
intensity = np.zeros((13,5,3))
intensity_compute(phi, intensity)
sns.heatmap(intensity[0,:,:]/np.sum(intensity[0,:,:]),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[0,1],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
ax0[0,1].set_title('Topic 1')


#psi = hier_model[0].get_parent_psi()
for i in range(3):
    topic = 1+i
    sns.heatmap(intensity[topic,:,:]/np.sum(intensity[topic,:,:]),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[1,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)    
    ax0[1,i].set_title('Topic 1-'+str(i+1))
    idx = np.argsort(psi.iloc[:,i].values)[::-1]
    for j in range(3):      
        topic = idx[j]+4
        sns.heatmap(intensity[topic,:,:]/np.sum(intensity[topic,:,:]),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[j+2,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
        ax0[j+2,i].set_title('Topic 1-'+str(i+1)+"-"+str(j+1))
'''


