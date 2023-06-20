'''
The code is based on the rCRP implementation by Joon Hee Kim (joonheekim@gmail.com)
'''

import seaborn as sns
import logging, random, math, os, bisect
import numpy as np
from copy import copy
import _pickle as cPickle
import gzip
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


exp_dir = './experiment/'
V = 13 # vocabulary size

class Table:
	def __init__(self):
		self.words = []
		self.topic = None

class Topic:
	def __init__(self, num_word, beta, level):
		self.big_table_count = 0
		self.small_table_count = 0
		self.children = []
		self.parent = None
		self.cumu_phi = np.ones((num_word)) * beta
		self.phi = np.ones((num_word)) * beta
		self.cumu_phi_sum = beta * num_word
		self.level = level
		self.intensity = np.zeros((5,3))
		self.cumu_intensity = np.zeros((5,3))

class Document:
	def __init__(self, docID, words):
		self.docID = docID
		self.words = words
		self.word2table = [None] * len(words)
		self.tables = []


class Model:
	def __init__(self, alpha, beta, param, iteration, depth_limit):
		self.alpha = alpha 	
		self.beta = beta # topic smoothing parameter
		self.param = param
		self.iteration = iteration
		self.depth_limit = depth_limit

		


	def load_data(self,all_docs):         
		self.num_word = V
		self.corpus = []
		length = 0
		num_doc = len(all_docs)

		logging.info('reading data')	
		for doc in all_docs:
			words = [int(word)-1 for word in doc ]
			doc2id = Document(None, words)
			self.corpus.append(doc2id)
			length += len(words)
		logging.info('average document length:' + str(length / float(num_doc)))
        
        # initialize tree structure
		self.topics = {}	# {level: topics}
		root_topic = Topic(self.num_word, self.beta, 0)
		root_topic.big_table_count += 10
		root_topic.small_table_count += 10
		for i in range(self.depth_limit):
			self.topics[i] = []
		self.topics[0].append(root_topic)
		self.root_topic = root_topic
        
		for level in range(1,self.depth_limit):
			for parent_topic in self.topics[level-1]:
				# generate 3 direct children
				for j in range(3):
					topic = Topic(self.num_word, self.beta, level)
					topic.parent = parent_topic
					parent_topic.children.append(topic)
					self.topics[level].append(topic)
					topic.small_table_count += 10
					topic.big_table_count += 10
					p_topic = parent_topic
					while (p_topic != None):
						p_topic.big_table_count += 10
						p_topic = p_topic.parent
                        


	def run(self):
		for i in range(self.iteration):
			logp = 0
			logging.info('iteration: ' + str(i) + '\t processing: ' + str(len(self.corpus)) + ' documents')
			for document in self.corpus:
				self.process_document(document)
				for j in range(len(document.words)):
					word = document.words[j]
					logp += np.log(document.word2table[j].topic.phi[word]/np.sum(document.word2table[j].topic.phi))
			self.print_count()
			#self.print_state(i)
			logging.info('likelihood: ' + ' ' + str(logp))
		self.intensity_compute()

		
	def top_words(self, vector, n):
		vector = copy(vector)
		result = ''
		for i in range(n):
			argmax = np.argmax(vector)
			value = vector[argmax]
			vector[argmax] = -1
			result += str(argmax+1)
			result += '\t'
			result += ("%.3f"%value)
			result += '\t'
		return result
    

	def choose(self, roulette):
		total = sum(roulette)
		arrow = total * random.random()
		return bisect.bisect(np.cumsum(roulette), arrow)
		
	def print_topic(self, topic):
		string = ''
		if (topic.level == 1):
			string += '\n\n'
		if (topic.level == 2):
			string += '\n'
		string += ('level: ' + str(topic.level) + '\t')
		string += (str(topic.big_table_count) + '\t')
		string += (str(topic.small_table_count) + '\t')
		string += (self.top_words(topic.cumu_phi, 10) + '\n')
		return string

	def print_topics_recursively(self, topic, _string):
		string = self.print_topic(topic)
		for i in range(len(topic.children)):
			child = topic.children[i]
			string += self.print_topics_recursively(child, string)
		return string

	def print_state(self, i):
		logging.info('printing state\t' + self.param)
		if not os.path.isdir(exp_dir + self.param):
			os.mkdir(exp_dir + self.param)
		write_file = open(exp_dir + self.param + '/' + str(i) + '.txt', 'w')
		write_file.write(self.print_topics_recursively(self.root_topic, ''))
		write_file.close()

	def print_count(self):
		num_topics = np.zeros((self.depth_limit))
		num_tables = np.zeros((self.depth_limit))
		for i in range(self.depth_limit):
			num_topics[i] = len(self.topics[i])
			for topic in self.topics[i]:
				num_tables[i] += topic.small_table_count
		logging.info('num_topics: ' + str(num_topics))
		logging.info('num_tables: ' + str(num_tables))
		logging.info('num_average_tables 1: ' + str(num_tables / num_topics))
		logging.info('num_average_tables 2: ' + str(sum(num_tables) / sum(num_topics)))
    
	def intensity_compute(self):
		for level in range(self.depth_limit):
			for topic in self.topics[level]:
				topic.cumu_intensity[0,1] += topic.cumu_phi[0]
				topic.intensity[0,1] += topic.phi[0]
				for i in range(1,5):
				    for j in range(3):
				        v = j*4+i   
				        topic.cumu_intensity[i,j] += topic.cumu_phi[v]
				        topic.intensity[i,j] += topic.phi[v]
                            


	def process_document(self, document):
		if len(document.tables) == 0:
			random.shuffle(document.words)

		# table assignment
		for i in range(len(document.words)):
			word = document.words[i]

			# de-assignment
			old_table = document.word2table[i]

			# if first assignment, pass de-assignment
			if old_table == None:
				pass
			else:
				# remove previous assignment related to word
				old_table.words.remove(word)
				old_topic = old_table.topic
				old_topic.cumu_phi[word] -= 1
				old_topic.phi[word] -= 1
				old_topic.cumu_phi_sum -= 1

				# remove previous assignment of parents' related to word recursively
				parent_topic = old_topic.parent
				while(parent_topic != None):
					parent_topic.cumu_phi[word] -= 1
					parent_topic.cumu_phi_sum -= 1
					parent_topic = parent_topic.parent
				
				# if old_table has no word, remove it
				if len(old_table.words) == 0:
					document.tables.remove(old_table)
					old_topic.big_table_count -= 1
					old_topic.small_table_count -= 1
					parent_topic = old_topic.parent
					while(parent_topic != None):
						parent_topic.big_table_count -= 1
						parent_topic = parent_topic.parent
					

			# table assignment
			roulette = np.zeros((len(document.tables) + 1))
			for j in range(len(document.tables)):
				table = document.tables[j]
				roulette[j] = (table.topic.phi[word] / np.sum(table.topic.phi)) * len(table.words)
			roulette[-1] = self.alpha / self.num_word
			new_table_index = self.choose(roulette)

			# error case
			if new_table_index == -1:
				print ('error 1')
				exit(-1)

			# create new table if last index is chosen
			if new_table_index == len(document.tables):
				new_table = Table()
				document.tables.append(new_table)
				new_topic = self.get_topic_for_table(new_table)
				new_table.topic = new_topic
				new_topic.big_table_count += 1
				new_topic.small_table_count += 1
				new_parent_topic = new_topic.parent
				while(new_parent_topic != None):
					new_parent_topic.big_table_count += 1
					new_parent_topic = new_parent_topic.parent
			else:
				new_table = document.tables[new_table_index]
				new_topic = new_table.topic
			new_table.words.append(word)
			new_topic.cumu_phi[word] += 1
			new_topic.phi[word] += 1            
			new_topic.cumu_phi_sum += 1
			new_parent_topic = new_topic.parent
			while(new_parent_topic != None):
				new_parent_topic.cumu_phi[word] += 1
				new_parent_topic.cumu_phi_sum += 1
				new_parent_topic = new_parent_topic.parent
			document.word2table[i] = new_table

		# topic assignment
		for i in range(len(document.tables)):
			# de-assignment
			table = document.tables[i]
			old_topic = table.topic
			parent_topic = old_topic.parent

			for word in table.words:
				old_topic.cumu_phi[word] -= 1
				old_topic.phi[word] -= 1
			old_topic.cumu_phi_sum -= len(table.words)
			old_topic.big_table_count -= 1	
			old_topic.small_table_count -= 1									
              
			while(parent_topic != None):
				for word in table.words:
					parent_topic.cumu_phi[word] -= 1					
				parent_topic.cumu_phi_sum -= len(table.words)
				parent_topic.big_table_count -= 1									
				parent_topic = parent_topic.parent
               
			new_topic = self.get_topic_for_table(table)
			table.topic = new_topic
			
			for word in table.words:
				new_topic.cumu_phi[word] += 1
				new_topic.phi[word] += 1
			new_topic.cumu_phi_sum += len(table.words)
			new_topic.big_table_count += 1
			new_topic.small_table_count += 1			

			parent_topic = new_topic.parent
			while(parent_topic != None):
				for word in table.words:
					parent_topic.cumu_phi[word] += 1
				parent_topic.cumu_phi_sum += len(table.words)
				parent_topic.big_table_count += 1
				parent_topic = parent_topic.parent


	def get_topic_for_table(self, table):	

		parent_topic = self.root_topic ## root_topic
		for i in range(1, self.depth_limit):
			topics = parent_topic.children
			roulette = np.zeros((len(topics) + 1))
			for j in range(len(topics)):
				topic = topics[j]
				roulette[j] =  topic.big_table_count
				for word in table.words:
					roulette[j] *= (topic.cumu_phi[word] / topic.cumu_phi_sum * self.num_word)

			roulette[-1] = parent_topic.small_table_count
			for word in table.words:
				roulette[-1] *= (parent_topic.phi[word] / np.sum(parent_topic.phi) * self.num_word )

			# index -1, choose current parent topic, and stop
			# otherwise, choose one of children and move down the tree
			topic_index = self.choose(roulette)
			if topic_index == -1:
				logging.info('error in get_topic_for_table')
				logging.info('len(table.words):' + str(len(table.words)))
				exit(-1)
			if topic_index == len(topics):
				return parent_topic
			else:
				parent_topic = topics[topic_index]
		return parent_topic

    
	def online_process_document(self, doc):

		words = [int(word)-1 for word in doc]
		tables = [] 
		word2table = [None] * len(words)

		for iter in range(25):
        	# table assignment
			for i in range(len(words)):
				word = words[i]

				# de-assignment
				old_table = word2table[i]

				# if first assignment, pass de-assignment
				if old_table == None:
					pass
				else:
					# remove previous assignment related to word
					old_table.words.remove(word)
                    
                    # if old_table has no word, remove it
					if len(old_table.words) == 0:
						tables.remove(old_table)

				# re assignment                
				roulette = np.zeros((len(tables) + 1))
				for j in range(len(tables)):
					table = tables[j]
					roulette[j] = (table.topic.phi[word] / np.sum(table.topic.phi)) * len(table.words)
				roulette[-1] = self.alpha / self.num_word
				new_table_index = self.choose(roulette)

				# error case
				if new_table_index == -1:
					print ('error 1')
					exit(-1)
                    
                # create new table if last index is chosen
				if new_table_index == len(tables):
					new_table = Table()
					tables.append(new_table)
					new_topic = self.get_topic_for_table(new_table)
					new_table.topic = new_topic
				else:
					new_table = tables[new_table_index]
					new_topic = new_table.topic
				new_table.words.append(word)
                
				word2table[i] = new_table
                
                
            # topic assignment
			for i in range(len(tables)):
				# re-assignment
				table = tables[i]	
                
				new_topic = self.get_topic_for_table(table)
				table.topic = new_topic

			
        # compute likelihood
		logp = 0
		for j in range(len(words)):
		    word = words[j]
		    logp += np.log(word2table[j].topic.phi[word]/np.sum(word2table[j].topic.phi))
		
		
        # compute cosine_similarity
		topics = [table.topic for table in word2table]
		scores = [cosine_similarity(topic.phi.reshape(1,-1),self.root_topic.cumu_phi.reshape(1,-1))[0][0] for topic in topics]

		return logp,np.average(scores)

		
        

def save_zipped_pickle(obj, filename, protocol=-1):
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
    with open('../data/synthetic_corpus/10_0.1.txt',encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()
        
    reviews = [content.split('\t')[0] for content in raw_reviews]
       
    docs = []
    for doc in reviews:  
        doc = nltk.word_tokenize(doc)
        if len(doc)>0:
            docs.append(doc)

    # model setting
    mu = 1.5
    N_w = 10
    alpha = 0.1 * N_w
    beta = 0.01
    iteration = 100
    depth_limit = 3
    param = "10_0.1_rCRP"


    # initialize model
    model = Model(alpha, beta, param, iteration, depth_limit)
    model.load_data(docs)

    # model training
    model.run()
    '''
    # save model for further use
    save_zipped_pickle(model, exp_dir + model.param + '/model.p')
    
    # or load existing model
    with gzip.open(exp_dir+"10_0.1_enhanced_model/model.p", 'rb') as f:
        model = cPickle.load(f)
    '''
    
    # visualize results - heatmap for topic distribution over vocabulary
    mask = np.array([False]*15)
    mask = mask.reshape(5, 3)
    mask[0,0] = True
    mask[0,2] = True

    fig, ax0 = plt.subplots(5, 3,figsize=(10,10))
    topic = model.root_topic
    sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[0,1],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
    ax0[0,1].set_title('Topic 1')
    

    for i in range(3):
        topic = model.root_topic.children[i]
        sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[1,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)    
        ax0[1,i].set_title('Topic 1-'+str(i+1))
        for j in range(3):
            topic = model.root_topic.children[i].children[j]
            sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[j+2,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
            ax0[j+2,i].set_title('Topic 1-'+str(i+1)+"-"+str(j+1))
    
    
    # model testing
    # load test data
    reviews = []
    with open('../data/synthetic_corpus/10_0.1_test.txt',encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()

    reviews = [content.split('\t')[0] for content in raw_reviews]
    labels1 = [content.split('\t')[1] for content in raw_reviews]
    labels2 = [content.split('\t')[2] for content in raw_reviews]


    replicate = 10 
    logp = np.zeros((len(reviews),replicate))
    avg_scores = np.zeros((len(reviews),replicate))
    N = 0
    for i in range(len(reviews)): 
        logging.info('test doc:' + str(i)) 
        doc = nltk.word_tokenize(reviews[i])
        N += len(doc) 
        avg_logp = 0
        for j in range(replicate):
            logp[i,j],avg_scores[i,j] = model.online_process_document(doc)

    logp = np.sum(logp,axis=0)/N
    # averge held-out likelihood
    print(np.mean(logp),np.std(logp)/np.sqrt(replicate))

    
    # ROC_AUC
    avg_roc_scores1 = np.zeros(replicate)

    for j in range(replicate):
        avg_roc_scores1[j] = roc_auc_score(labels1,avg_scores[:,j])
    # AUC1   
    print(np.mean(avg_roc_scores1),np.std(avg_roc_scores1)/np.sqrt(replicate))

    
    avg_roc_scores2 = np.zeros(replicate)
    for j in range(replicate):
        avg_roc_scores2[j] = roc_auc_score(labels2,avg_scores[:,j])
    # AUC2    
    print(np.mean(avg_roc_scores2),np.std(avg_roc_scores2)/np.sqrt(replicate))

    

    