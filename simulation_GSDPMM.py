
from __future__ import unicode_literals, print_function, division
import numpy as np
import gensim, logging, random, bisect
from sklearn.metrics.pairwise import cosine_similarity
import _pickle as cPickle
import gzip
import nltk
from sklearn.metrics import roc_curve, auc, roc_auc_score


length = 10
alpha = 0.1
train_dir = '../data/synthetic_corpus/'+str(length)+'_'+str(alpha)+'.txt'
test_dir = '../data/synthetic_corpus/'+str(length)+'_'+str(alpha)+'_test.txt'





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
        self.num_word = len(self.dictionary)
        self.back_density = np.zeros(self.num_word)
        logging.info("vocabulary: "+str(self.num_word))
        self.bow = []

        logging.info('reading data')	
        for doc in all_docs:
            self.bow.append(self.dictionary.doc2bow(doc))

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
            #logging.info('iteration: ' + str(i+1) + ' clusters: '+ str(len(self.clusters)) +' likelihood: ' + str(logp))
                
    def compute_topic_score(self):
        for cluster in self.clusters:
            cluster.specialized_score = 1-cosine_similarity(cluster.phi.reshape(1,-1),self.back_density.reshape(1,-1))[0][0]
    		

    def sample_cluster(self, d):
        roulette = np.zeros((len(self.clusters)))
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
        cluster_index = self.choose(roulette)
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
			
        # compute specialization scores
        score = 0
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            score += p_cluster[i] * cluster.specialized_score
        # compute likelihood
        logp = 0
        for word in bow:
            wordNo = word[0]
            wordfreq = word[1]
            p_w = 0
            for i in range(len(self.clusters)):
                cluster = self.clusters[i]
                p_w += p_cluster[i] * cluster.phi[wordNo]/cluster.phi_sum
            logp += wordfreq * np.log(p_w)
        
        return score,logp


    def choose(self, roulette):
        total = sum(roulette)
        arrow = total * random.random()
        return bisect.bisect(np.cumsum(roulette), arrow)                

    def save_zipped_pickle(self, filename, protocol=-1):
        obj = self
        with gzip.open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol)                

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object




if __name__ == "__main__":
    
    random.seed(0)
    
    # load training data
    reviews = []
    with open(train_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()
        
    reviews = [content.split('\t')[0] for content in raw_reviews]
       
    docs = []
    for doc in reviews:  
        doc = nltk.word_tokenize(doc)
        if len(doc)>0:
            docs.append(doc)
    
    # load test data
    with open(test_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()

    reviews = [content.split('\t')[0] for content in raw_reviews]
    labels1 = [content.split('\t')[1] for content in raw_reviews]
    labels2 = [content.split('\t')[2] for content in raw_reviews]
    N = 0
    test_docs = []
    for i in range(len(reviews)): 
        doc = nltk.word_tokenize(reviews[i])
        N += len(doc) 
        test_docs.append(doc)


    # model validation
    alpha = 1.0
    beta = 0.1
    K = 13
    iterNum = 20

    replicate = 10 
    logp = np.zeros((len(test_docs),replicate))
    avg_scores = np.zeros((len(test_docs),replicate))

    for j in range(replicate):
        # model training
        gsdpmm = GSDPMM(K,alpha,beta)
        gsdpmm.load_data(docs)
        gsdpmm.gibbs_sampling(iterNum)
        gsdpmm.compute_topic_score()
        # model testing
        for i in range(len(test_docs)):       
            avg_scores[i,j],logp[i,j] = gsdpmm.online_cluster_document(test_docs[i])  

   
    # ROC_AUC
    avg_roc_scores1 = np.zeros(replicate)

    for j in range(replicate):
        avg_roc_scores1[j] = 1-roc_auc_score(labels1,avg_scores[:,j])
    # AUC1   
    print(np.mean(avg_roc_scores1),np.std(avg_roc_scores1)/np.sqrt(replicate))
    
    avg_roc_scores2 = np.zeros(replicate)
    for j in range(replicate):
        avg_roc_scores2[j] = 1-roc_auc_score(labels2,avg_scores[:,j])
    # AUC2    
    print(np.mean(avg_roc_scores2),np.std(avg_roc_scores2)/np.sqrt(replicate))
    
    
    logp = np.sum(logp,axis=0)/N
    # averge held-out likelihood
    print(np.mean(logp),np.std(logp)/np.sqrt(replicate))
     








