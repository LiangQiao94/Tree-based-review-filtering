'''
The code is based on rCRP implementation by Joon Hee Kim (joonheekim@gmail.com)
hyperparameter explanations:
alpha = parameter for creating new table
beta = parameter for topic smoothing
gamma1, gamma2 = parameter for creating new topic
where gamma = gamma1 * gamma2^topic_level (root = level 0)
delta = parameter for copying word distributions from parent topics
'''

import gensim, logging, random, math, os, bisect
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from copy import copy
import _pickle as cPickle
import gzip
import nltk
import string
import xlwt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc, roc_auc_score
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
training (offline) data are provided by:
- Yelp Open Dataset: https://www.yelp.com/dataset
- Amazon review dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
Testing (online) data with annotation are provided by:
- Laptop-ACOS and Restaurant-ACOS: https://github.com/NUSTM/ACOS/tree/main/data
'''

exp_dir = './experiment/'
train_dir = '../data/restaurant/restaurant_offline.txt'
test_dir = '../data/restaurant/restaurant_online.txt'
'''
train_dir = '../data/laptop/laptop_offline.txt'
test_dir = '../data/laptop/laptop_online.txt'
'''





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
		self.phi = np.ones((num_word)) * beta
		self.cumu_phi = np.ones((num_word)) * beta
		self.cumu_phi_sum = beta * num_word
		self.level = level
		self.weight = np.ones((num_word))

class Document:
	def __init__(self, docID, words):
		self.docID = docID
		self.words = words
		self.word2table = [None] * len(words)
		self.tables = []


class Model:
	def __init__(self, alpha, beta, gamma1, gamma2, iteration, delta, depth_limit, param):
		self.alpha = alpha 	
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.beta = beta # topic smoothing factor
		self.delta = delta
		self.iteration = iteration
		self.depth_limit = depth_limit
		self.param = param

	def load_data(self,all_docs):
        
		self.dictionary = gensim.corpora.Dictionary(all_docs)      
		self.dictionary.filter_extremes(no_below=50)
		self.dictionary.compactify() 
		self.num_word = len(self.dictionary)
		logging.info("vocabulary: "+str(self.num_word))
		self.corpus = []
		length = 0
        
		self.data_ratio = 1
		logging.info('reading data')	
		for doc in all_docs:
			words = [self.dictionary.token2id[word] for word in doc if word in self.dictionary.token2id.keys()]
			if len(words)>10 or len(words)<1 or np.random.random() > self.data_ratio:
				continue           
			doc2id = Document(None, words)
			self.corpus.append(doc2id)
			length += len(words)
		logging.info('documents: ' + str(len(self.corpus)))    
		logging.info('average document length:' + str(length / float(len(self.corpus))))

		self.topics = {}		# {level: topics}
		root_topic = Topic(self.num_word, self.beta, 0)
		root_topic.big_table_count += 0.1
		root_topic.small_table_count += 0.1
		for i in range(self.depth_limit):
			self.topics[i] = []
		self.topics[0].append(root_topic)
		self.root_topic = root_topic


	def run(self):		
		for i in range(self.iteration):
			logp = 0
			logging.info('iteration: ' + str(i))
			for document in self.corpus:
				self.process_document(document)
				for j in range(len(document.words)):
					word = document.words[j]
					logp += np.log(document.word2table[j].topic.phi[word]/np.sum(document.word2table[j].topic.phi))
			self.weight_update()
			self.print_count()
			logging.info('likelihood: ' + str(logp))
			if ((i+1)%10 == 0):
				self.print_state()


		
	def top_words(self, vector, n):
		vector = copy(vector)
		result = ''
		for i in range(n):
			argmax = np.argmax(vector)
			value = vector[argmax]
			vector[argmax] = -1
			result += self.dictionary[argmax]
			result += '\t'
			#result += ("%.3f"%value)
			#result += '\t'
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
		string += (self.top_words(topic.phi, 20) + '\n')
		return string

	def print_topics_recursively(self, topic, _string):
		string = self.print_topic(topic)
		for i in range(len(topic.children)):
			child = topic.children[i]
			string += self.print_topics_recursively(child, string)
		return string


	def print_state(self):
		logging.info('printing state\t' + self.param)
		if not os.path.isdir(exp_dir + self.param +str(self.data_ratio)):
			os.mkdir(exp_dir + self.param +str(self.data_ratio))
		write_file = open(exp_dir + self.param +str(self.data_ratio) + '/top_words_'+str(self.gamma1)+'_'+str(self.gamma2)+'.txt', 'w')
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

	def weight_update(self):		
		for i in range(self.depth_limit-1):           
			for parent_topic in self.topics[i]:
				topics = parent_topic.children
            
				ent = 0
				net_cumu_phi_sum = sum([topic.cumu_phi_sum for topic in topics])
				for topic in topics:
					p = topic.cumu_phi_sum/net_cumu_phi_sum
					ent -= p*math.log(p)
                    
				for word in range(self.num_word):
					ent_w = 0
					net_cumu_phi_w = sum([topic.cumu_phi[word] for topic in topics])
					for topic in topics:
						p = topic.cumu_phi[word]/net_cumu_phi_w
						ent_w -= p*math.log(p)                       
					if ent > ent_w :
						parent_topic.weight[word] = mu*ent_w/ent
		return

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
				old_topic.phi[word] -= 1
				old_topic.cumu_phi[word] -= 1
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
					
					# if old_topic, and their parents have no table assigned, remove them
					parent_topic = old_topic.parent
					if old_topic.big_table_count == 0:
						parent_topic.children.remove(old_topic)
						self.topics[old_topic.level].remove(old_topic)                        

					while(parent_topic != None):
						if parent_topic.big_table_count == 0:
							parent_topic.parent.children.remove(parent_topic)
							self.topics[parent_topic.level].remove(parent_topic)  
							parent_topic = parent_topic.parent
						else:
							break

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
			new_topic.phi[word] += 1
			new_topic.cumu_phi[word] += 1
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
			if old_topic.big_table_count == 0:
				parent_topic.children.remove(old_topic)
				self.topics[old_topic.level].remove(old_topic)
                
                
			while(parent_topic != None):
				for word in table.words:
					parent_topic.cumu_phi[word] -= 1					
				parent_topic.cumu_phi_sum -= len(table.words)
				parent_topic.big_table_count -= 1									
				if parent_topic.big_table_count == 0:
					#print('remove topic ' + str(parent_topic) + " from topic " + str(parent_topic.parent) + " children " + str(parent_topic.parent.children))
					parent_topic.parent.children.remove(parent_topic)
					self.topics[parent_topic.level].remove(parent_topic)
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
			roulette = np.zeros((len(topics) + 2))
			for j in range(len(topics)):
				topic = topics[j]
				roulette[j] =  topic.big_table_count
				for word in table.words:
					roulette[j] *= (topic.cumu_phi[word] * self.num_word / topic.cumu_phi_sum )
                    
			# index -1, create new child topic, and stop
			# index -2, choose current parent topic, and stop
			# otherwise, choose one of children and move down the tree
			roulette[-1] = self.gamma1 * math.pow(self.gamma2, i)
			roulette[-2] = parent_topic.small_table_count
			for word in table.words:
				roulette[-2] *= (parent_topic.phi[word] * self.num_word / np.sum(parent_topic.phi) * parent_topic.weight[word])
        

			topic_index = self.choose(roulette)

			if topic_index == -1:
				logging.info('error in get_topic_for_table')
				logging.info('len(table.words):' + str(len(table.words)))
				exit(-1)
			if topic_index == len(topics) + 1:
				# create new topic
				topic = Topic(self.num_word, self.beta, parent_topic.level + 1)
				topic.parent = parent_topic
				parent_topic.children.append(topic)
				topic.phi = copy(topic.parent.cumu_phi) * self.delta                
				topic.cumu_phi = copy(topic.parent.cumu_phi) * self.delta
				topic.cumu_phi_sum = copy(topic.parent.cumu_phi_sum) * self.delta
				self.topics[i].append(topic)
				return topic
			if topic_index == len(topics):
				# choose current parent topic
				return parent_topic
			else:
				parent_topic = topics[topic_index]
		return parent_topic
    
	def online_process_document(self, doc):
        # reset count number
		words = [self.dictionary.token2id[word] for word in doc if word in self.dictionary.token2id.keys()]
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
					new_topic = self.online_get_topic_for_table(new_table)
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
                
				new_topic = self.online_get_topic_for_table(table)
				table.topic = new_topic
				#print(new_topic.level)
			

        # compute similarity scores
		topics = [table.topic for table in word2table]
		scores = [1-cosine_similarity(topic.phi.reshape(1,-1),self.root_topic.cumu_phi.reshape(1,-1))[0][0] for topic in topics]
		if len(topics)>0:
			return np.average(scores)
		return 0
        

	def online_get_topic_for_table(self, table):	

		parent_topic = self.root_topic ## root_topic
		for i in range(1, self.depth_limit):
			topics = parent_topic.children
			roulette = np.zeros((len(topics) + 1))
			for j in range(len(topics)):
				topic = topics[j]
				roulette[j] =  topic.big_table_count
				for word in table.words:
					roulette[j] *= (topic.cumu_phi[word] / topic.cumu_phi_sum * self.num_word)

			# index -1, choose current parent topic, and stop
			# otherwise, choose one of children and move down the tree
			roulette[-1] = parent_topic.small_table_count
			for word in table.words:
				roulette[-1] *= (parent_topic.phi[word] / np.sum(parent_topic.phi) * self.num_word * parent_topic.weight[word])

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
	

def save_zipped_pickle(obj, filename, protocol=-1):
	with gzip.open(filename, 'wb') as f:
		cPickle.dump(obj, f, protocol)
        
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object




if __name__ == "__main__":
    
    random.seed(0)
    np.random.seed(0)
    
    # load training data
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
    alpha = 3.0
    beta = 0.1
    gamma1 = 1.0
    gamma2 = 0.2
    iteration = 200
    delta = 0.01
    mu = 1.7
    depth_limit = 3
    param = 'restaurant_proposed'

    # model initialize
    model = Model(alpha, beta, gamma1, gamma2, iteration, delta, depth_limit, param)
    model.load_data(docs)
    
    
    # model training
    model.run() 
    
    save_zipped_pickle(model, exp_dir + param + '/proposed_model.p')

    '''
    # or load existing model 
    with gzip.open(exp_dir + param + '/proposed_model.p', 'rb') as f:
        model = cPickle.load(f)
    '''

            
    
    # load testing data 
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
        replicates = 10
        for j in range(replicates):
            avg_s = model.online_process_document(doc)  
            avg_scores[i] += avg_s
        avg_scores[i] = avg_scores[i]/replicates        


    
    # save avg_scores for later classification    
    f = xlwt.Workbook('encoding = utf-8') 
    sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) 
    sheet1.write(0,0,'proposed') 
    for i in range(len(avg_scores)):
        sheet1.write(i+1,0,avg_scores[i]) 
    f.save('restaurant_test_score_proposed.xls')
    
    # ROC_AUC
    auc_score = 1-roc_auc_score(labels,avg_scores)   
    print(auc_score)
    
    

    


