from __future__ import unicode_literals, print_function, division
import string
import numpy as np
import gensim, logging, random, math, os, bisect
from sklearn.metrics.pairwise import cosine_similarity
import _pickle as cPickle
import gzip
import nltk
import xlwt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc, roc_auc_score

'''
training (offline) data are provided by:
- Yelp Open Dataset: https://www.yelp.com/dataset
- Amazon review dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
Testing (online) data with annotation are provided by:
- Laptop-ACOS and Restaurant-ACOS: https://github.com/NUSTM/ACOS/tree/main/data
'''

exp_dir = './experiment/'
'''
train_dir = '../data/laptop/laptop_offline.txt'
test_dir = '../data/laptop/laptop_online.txt'
'''
train_dir = '../data/restaurant/restaurant_offline.txt'
test_dir = '../data/restaurant/restaurant_online.txt'



class Topic:
	def __init__(self, num_word, beta):
		self.m_z = 0
		self.phi = np.ones((num_word)) * beta
		self.phi_sum = beta * num_word
		self.specialized_score = 0


class GSDPMM:
    def __init__(self, K, alpha, beta):
        self.K=K
        self.alpha=alpha
        self.beta=beta

        
    def load_data(self,all_docs):
        
        self.dictionary = gensim.corpora.Dictionary(all_docs)      
        self.dictionary.filter_extremes(no_below=50)
        self.dictionary.compactify() 
        self.num_word = len(self.dictionary)
        self.back_density = np.zeros(self.num_word)
        logging.info("vocabulary: "+str(self.num_word))
        self.bow = []

        logging.info('reading data')	
        for doc in all_docs:
            words = [word for word in doc if word in self.dictionary.token2id.keys()]
            if len(words)>0 and len(words)<=50:
                self.bow.append(self.dictionary.doc2bow(words))

        self.clusters = []
        for i in range(self.K):
            t = Topic(self.num_word, self.beta)
            self.clusters.append(t)
        
        # initialize
        self.z_c = {}     
        for d in range(len(self.bow)):
            cluster = self.clusters[int(np.floor(self.K*np.random.uniform()))]
            self.z_c[d] = cluster
            cluster.m_z += 1
            for w in self.bow[d]:
                cluster.phi[w[0]] += w[1]
                cluster.phi_sum += w[1]
                self.back_density[w[0]] += w[1]
        

        
    def gibbs_sampling(self, iterNum):
        
        for i in range(iterNum):
            logp = 0
            for d in range(len(self.bow)):
                cluster = self.z_c[d]
                cluster.m_z -= 1
                for w in self.bow[d]:
                    cluster.phi[w[0]] -= w[1]
                    cluster.phi_sum -= w[1]
                if cluster.m_z == 0:
                    self.clusters.remove(cluster)
                
                cluster=self.sample_cluster(d)
                self.z_c[d] = cluster
                cluster.m_z += 1
                for w in self.bow[d]:
                    cluster.phi[w[0]] += w[1]
                    cluster.phi_sum += w[1]
                    logp += w[1]*np.log(cluster.phi[w[0]]/cluster.phi_sum)
            logging.info('iteration: ' + str(i+1) + ' clusters: '+ str(len(self.clusters)) +' likelihood: ' + str(logp))
                
    def compute_topic_score(self):
        for cluster in self.clusters:
            cluster.specialized_score = 1-cosine_similarity(cluster.phi.reshape(1,-1),self.back_density.reshape(1,-1))[0][0]
    		

    def sample_cluster(self, d):
        roulette = np.zeros((len(self.clusters) + 1))
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            roulette[i] =  cluster.m_z
            count = 0
            for word in self.bow[d]:
                wordNo = word[0]
                wordfreq = word[1]
                for j in range(wordfreq):
                    roulette[i] *= ((cluster.phi[wordNo] + j) * self.num_word / (cluster.phi_sum + count) )
                    count += 1
        
        count = 0
        roulette[-1] = self.alpha
        for word in self.bow[d]:
            wordNo = word[0]
            wordfreq = word[1]
            for j in range(wordfreq):
                roulette[-1] *= ((self.beta + j) * self.num_word / (self.num_word*self.beta + count) )
                count += 1
               
        cluster_index = self.choose(roulette)
        if cluster_index == len(self.clusters):
            # create new cluster
            cluster = Topic(self.num_word, self.beta)
            self.clusters.append(cluster)
            return cluster
        else:
            return self.clusters[cluster_index]

    def online_cluster_document(self, doc):

        bow = self.dictionary.doc2bow(doc)
        roulette = np.zeros(len(self.clusters))

        # sample a cluster
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            roulette[i] =  cluster.m_z
            count = 0
            for word in bow:
                wordNo = word[0]
                wordfreq = word[1]
                for j in range(wordfreq):
                    roulette[i] *= ((cluster.phi[wordNo] + j) * self.num_word / (cluster.phi_sum + count) )
                    count += 1
        p_cluster = roulette/np.sum(roulette)
        cluster_index = self.choose(roulette)
        cluster = self.clusters[cluster_index]
			
        # compute similarity scores
        score = 0
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            score += p_cluster[i] * cluster.specialized_score
        return score


    def choose(self, roulette):
        total = sum(roulette)
        arrow = total * random.random()
        return bisect.bisect(np.cumsum(roulette), arrow)                



def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)                

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object




if __name__ == "__main__":
    
    random.seed(0)
    
    
    # load_training_data
    with open(train_dir,encoding='utf-8') as fcorpus:
	    reviews = fcorpus.read().splitlines()

    stemmer = PorterStemmer()
    stopset = stopwords.words('english') + ['will', 'also', 'said']
    del_str = string.punctuation + string.digits
    replace = str.maketrans(del_str,' '*len(del_str))
    docs = [] 
    for doc in reviews:  
        # strip punctuations and digits
        doc = doc.translate(replace) 
        doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars
        doc = nltk.word_tokenize(doc)
        doc = [w.lower() for w in doc]
        doc = [w for w in doc if w not in stopset]
        doc = [stemmer.stem(w) for w in doc]
        if len(doc)>0:
            docs.append(doc)


    # model setting
    alpha = 1.0
    beta = 0.1
    K = 20
    iterNum = 50

    # initialize
    gsdpmm = GSDPMM(K,alpha,beta)
    gsdpmm.load_data(docs)
    # model training
    gsdpmm.gibbs_sampling(iterNum)
    gsdpmm.compute_topic_score()
    save_zipped_pickle(gsdpmm,'./experiment/restaurant_DPMM/gsdpmm.p')
        
    
    '''
    # or load existing model
    with gzip.open('./experiment/restaurant_DPMM/gsdpmm.p', 'rb') as f:
        gsdpmm = cPickle.load(f)
    '''
    
    # load test set 
    raw_reviews = []
    with open(test_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()
    reviews = [content.split('\t')[0] for content in raw_reviews]   
    labels = [int(content.split('\t')[1]) for content in raw_reviews]

        
    stemmer = PorterStemmer()
    stopset = stopwords.words('english') + ['will', 'also', 'said']
    del_str = string.punctuation + string.digits
    replace = str.maketrans(del_str,' '*len(del_str))
    avg_scores = np.zeros(len(reviews))
    for i in range(len(reviews)):  
        # strip punctuations and digits
        doc = reviews[i].translate(replace) 
        doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars
        doc = nltk.word_tokenize(doc)
        doc = [w.lower() for w in doc]
        doc = [w for w in doc if w not in stopset]
        doc = [stemmer.stem(w) for w in doc]        
        # online estimate
        avg_scores[i] = gsdpmm.online_cluster_document(doc)   
        logging.info('process doc:' + str(i))
    
    # ROC_AUC
    auc_score = 1-roc_auc_score(labels,avg_scores)   
    print(auc_score)   
    
       
    # save avg_scores for later classification    
    f = xlwt.Workbook('encoding = utf-8') 
    sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) 
    sheet1.write(0,0,'GSDPMM') 
    for i in range(len(avg_scores)):
        sheet1.write(i+1,0,avg_scores[i]) 
    f.save('restaurant_test_score_GSDPMM.xls')
    



