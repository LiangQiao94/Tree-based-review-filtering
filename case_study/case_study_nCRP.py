'''
The code is based on the hlda package in Python
https://github.com/joewandy/hlda
'''
import csv
from math import log
import sys
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import random,nltk,os
import string
import gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gzip
import _pickle as cPickle
import logging
import xlwt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
training data are provided by:
- Yelp Open Dataset: https://www.yelp.com/dataset
- Amazon review dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
Testing data with annotation are provided by:
- Laptop-ACOS and Restaurant-ACOS: https://github.com/NUSTM/ACOS/tree/main/data
'''

exp_dir = './experiment/'
train_dir = '../data/restaurant/restaurant_offline.txt'
test_dir = '../data/restaurant/restaurant_online.txt'
'''
train_dir = '../data/laptop/laptop_offline.txt'
test_dir = '../data/laptop/laptop_online.txt'

'''



class NCRPNode(object):

    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0

    def __init__(self, num_levels, vocab, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 0
        self.parent = parent
        self.children = []
        self.level = level
        self.total_words = 0
        self.num_levels = num_levels

        self.vocab = vocab
        self.word_counts = np.zeros(len(vocab))

        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state

    def __repr__(self):
        parent_id = None
        if self.parent is not None:
            parent_id = self.parent.node_id
        return 'Node=%d level=%d customers=%d total_words=%d parent=%s' % (self.node_id,
            self.level, self.customers, self.total_words, parent_id)

    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(self.num_levels, self.vocab, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        ''' Check if this node is a leaf node '''
        return self.level == self.num_levels-1

    def get_new_leaf(self):
        ''' Keeps adding nodes along the path until a leaf node is generated'''
        node = self
        for l in range(self.level, self.num_levels-1):
            node = node.add_child()
        return node

    def drop_path(self):
        ''' Removes a document from a path starting from this node '''
        node = self
        node.customers -= 1
        if node.customers == 0:
            node.parent.remove(node)
        for level in range(1, self.num_levels): # skip the root
            node = node.parent
            node.customers -= 1
            if node.customers == 0:
                node.parent.remove(node)


    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        NCRPNode.total_nodes -= 1


    def add_path(self):
        ''' Adds a document to a path starting from this node '''
        node = self
        node.customers += 1
        for level in range(1, self.num_levels):
            node = node.parent
            node.customers += 1

    def select(self, gamma):
        ''' Selects an existing child or create a new one according to the CRP '''

        weights = np.zeros(len(self.children)+1)
        weights[0] = float(gamma) / (gamma+self.customers)
        i = 1
        for child in self.children:
            weights[i] = float(child.customers) / (gamma + self.customers)
            i += 1

        choice = self.random_state.multinomial(1, weights).argmax()
        if choice == 0:
            return self.add_child()
        else:
            return self.children[choice-1]



    def online_select(self):
        ''' Selects an existing child '''

        weights = np.zeros(len(self.children))
        i = 0
        for child in self.children:
            weights[i] = float(child.customers) / (self.customers)
            i += 1

        choice = self.random_state.multinomial(1, weights).argmax()

        return self.children[choice]

    
    def get_top_words(self, n_words, with_weight):
        ''' Get the top n words in this node '''

        pos = np.argsort(self.word_counts)[::-1]      
        sorted_vocab = [self.vocab[idx] for idx in pos[:n_words]]
        print(sorted_vocab)
        sorted_weights = self.word_counts[pos]
        sorted_weights = sorted_weights[:n_words]

        output = ''
        for word, weight in zip(sorted_vocab, sorted_weights):
            if with_weight:
                output += '%s (%d), ' % (word, weight)
            else:
                output += '%s, ' % word
        return output

class HierarchicalLDA(object):

    def __init__(self, corpus, vocab,
                 alpha=1.0, gamma=1.0, eta=0.1,
                 seed=0, verbose=True, num_levels=3):

        NCRPNode.total_nodes = 0
        NCRPNode.last_node_id = 0
        
        self.background_counts = np.zeros(len(vocab))
        self.corpus = corpus
        self.vocab = vocab
        self.alpha = alpha  # smoothing on doc-topic distributions
        self.gamma = gamma  # "imaginary" customers at the next, as yet unused table
        self.eta = eta      # smoothing on topic-word distributions

        self.seed = seed
        self.random_state = RandomState(seed)
        self.verbose = verbose

        self.num_levels = num_levels
        self.num_documents = len(corpus)
        self.num_types = len(vocab)
        self.eta_sum = eta * self.num_types


        # initialise a single path
        path = np.zeros(self.num_levels, dtype=np.object)

        # initialize and fill the topic pointer arrays for every document.
        # Set everything to the single path that we added earlier.
        self.root_node = NCRPNode(self.num_levels, self.vocab)
        self.document_leaves = {}                                   # currently selected path (ie leaf node) through the NCRP tree
        self.levels = np.zeros(self.num_documents, dtype=np.object) # indexed < doc, token >
        for d in range(len(self.corpus)):

            # populate nodes into the path of this document
            doc = self.corpus[d]
            doc_len = len(doc)
            path[0] = self.root_node
            self.root_node.customers += 1 # always add to the root node first
            for level in range(1, self.num_levels):
                # at each level, a node is selected by its parent node based on the CRP prior
                parent_node = path[level-1]
                level_node = parent_node.select(self.gamma)
                level_node.customers += 1
                path[level] = level_node

            # set the leaf node for this document
            leaf_node = path[self.num_levels-1]
            self.document_leaves[d] = leaf_node

            # randomly assign each word in the document to a level (node) along the path
            self.levels[d] = np.zeros(doc_len, dtype=np.int)
            for n in range(doc_len):
                w = doc[n]
                random_level = self.random_state.randint(self.num_levels)
                random_node = path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                self.levels[d][n] = random_level
                self.background_counts[w] += 1
        print("initial total_nodes: ", NCRPNode.total_nodes)

    def search_topics(self, node):
        # recursively search descendant topic nodes under a certain node including itself
        descendants = set()
        queue = [node]
        while (len(queue)>0):
            topic = queue.pop(0)
            descendants.add(topic)
            for child in topic.children:
                queue.append(child)
        return descendants

    def estimate(self, num_samples, display_topics=50, n_words=5, with_weights=True):
       
        print('HierarchicalLDA sampling\n')
        for s in range(num_samples):

            logging.info('iteration: ' + str(s) + '\t processing: ' + str(len(self.corpus)) + ' documents')

            for d in range(len(self.corpus)):
                self.sample_path(d)

            for d in range(len(self.corpus)):
                self.sample_topics(d)
            
            LL = self.compute_likelihood()
            logging.info('topics: ' + str(NCRPNode.total_nodes)+'  likelihood: ' + str(LL))
            
            if (s > 0) and ((s+1) % display_topics == 0):
                self.print_nodes(n_words, with_weights)
                
        

    def sample_path(self, d):

        # define a path starting from the leaf node of this doc
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d]
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d].drop_path()

        ############################################################
        # calculates the prior p(c_d | c_{-d}) in eq. (4)
        ############################################################

        node_weights = {}
        self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        ############################################################
        # calculates the likelihood p(w_d | c, w_{-d}, z) in eq. (4)
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}
        doc_levels = self.levels[d]
        doc = self.corpus[d]

        # remove doc from path
        for n in range(len(doc)): # for each word in the doc

            # count the word at each level
            level = doc_levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1

            # remove word count from the node at that level
            level_node = path[level]
            level_node.word_counts[w] -= 1
            level_node.total_words -= 1
            assert level_node.word_counts[w] >= 0
            assert level_node.total_words >= 0

        self.calculate_doc_likelihood(node_weights, level_word_counts)

        ############################################################
        # pick a new path
        ############################################################

        nodes = list(node_weights.keys())
        weights = np.array([node_weights[node] for node in nodes])
        weights = np.exp(weights - np.max(weights)) # normalise so the largest weight is 1
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]

        # if we picked an internal node, we need to add a new path to the leaf
        if not node.is_leaf():
            node = node.get_new_leaf()

        # add the doc back to the path
        node.add_path()                     # add a customer to the path
        self.document_leaves[d] = node      # store the leaf node for this doc

        # add the words
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            word_counts = level_word_counts[level]
            for w in word_counts:
                node.word_counts[w] += word_counts[w]
                node.total_words += word_counts[w]
            node = node.parent


    def calculate_ncrp_prior(self, node_weights, node, weight):
        ''' Calculates the prior on the path according to the nested CRP '''        

        for child in node.children:
            child_weight = log( float(child.customers) / (node.customers + self.gamma) )
            self.calculate_ncrp_prior(node_weights, child, weight + child_weight)
        if node.is_leaf():
            node_weights[node] = weight
        else:
            node_weights[node] = weight + log( self.gamma / (node.customers + self.gamma))
    



    def calculate_doc_likelihood(self, node_weights, level_word_counts):
        
        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(self.num_levels)
        for level in range(1, self.num_levels):  # skip the root

            word_counts = level_word_counts[level]

            for w in word_counts:
                new_topic_weights[level] += word_counts[w] * log((self.eta) / (self.eta_sum))

        self.calculate_word_likelihood(node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0)
        
        return



    def calculate_word_likelihood(self, node_weights, node, parent_weight, level_word_counts, new_topic_weights, level):

        # first calculate the likelihood of the words at this level, given this topic
        node_weight = parent_weight
        word_counts = level_word_counts[level]

        total_words = 0

        for w in word_counts:
            count = word_counts[w]
            for i in range(count): 
                node_weight += log( (self.eta + node.word_counts[w] + i) /
                                    (self.eta_sum + node.total_words + total_words) )
                total_words += 1


        # propagate that weight to the child nodes
        for child in node.children:
            self.calculate_word_likelihood(node_weights, child, node_weight,
                                           level_word_counts, new_topic_weights, level+1)

        # finally if this is an internal node, add the weight of a new path
        level += 1
        while level < self.num_levels:
            node_weight += new_topic_weights[level]
            level += 1

        if node in node_weights.keys():
            node_weights[node] += node_weight
        
        return



    def sample_topics(self, d):

        doc = self.corpus[d]

        # initialise level counts
        doc_levels = self.levels[d]
        level_counts = np.zeros(self.num_levels, dtype=np.int)
        for c in doc_levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d]
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent


        # sample a new level for each word
        level_weights = np.zeros(self.num_levels)
        for n in range(len(doc)):

            w = doc[n]
            word_level = doc_levels[n]

            # remove from model
            level_counts[word_level] -= 1
            node = path[word_level]
            node.word_counts[w] -= 1
            node.total_words -= 1

            # pick new level
            for level in range(self.num_levels):
                level_weights[level] = (self.alpha + level_counts[level]) *                     \
                    (self.eta + path[level].word_counts[w]) /  (self.eta_sum + path[level].total_words)
            level_weights = level_weights / np.sum(level_weights)
            level = self.random_state.multinomial(1, level_weights).argmax()

            # put the word back into the model
            doc_levels[n] = level
            level_counts[level] += 1
            node = path[level]
            node.word_counts[w] += 1
            node.total_words += 1

    def compute_likelihood(self):
        logp = 0
        for d in range(len(self.corpus)):
            # get the leaf node and populate the path
            path = np.zeros(self.num_levels, dtype=np.object)
            node = self.document_leaves[d]
            for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
                path[level] = node
                node = node.parent
            for n in range(len(self.corpus[d])):
                w = self.corpus[d][n]
                word_level = self.levels[d][n]
                node = path[word_level]
                logp += log( (self.eta + node.word_counts[w] ) / (self.eta_sum + node.total_words ) )
        return logp

           
    def online_estimate(self,doc):
        path = np.zeros(self.num_levels, dtype=np.object)
        # populate nodes into the path of this document

        path[0] = self.root_node
        for level in range(1, self.num_levels):
        # at each level, a node is selected by its parent node based on the CRP prior
            parent_node = path[level-1]
            level_node = parent_node.online_select()
            path[level] = level_node

        # set the leaf node for this document
        leaf_node = path[self.num_levels-1]
        
        levels = np.zeros(len(doc), dtype=np.int)
        docnode = np.zeros(len(doc), dtype=NCRPNode)
        for n in range(len(doc)):
            random_level = self.random_state.randint(self.num_levels)
            random_node = path[random_level]
            levels[n] = random_level
            docnode[n] = random_node
        
        for s in range(20):
            # sample a path
            leaf_node = self.online_sample_path(doc,levels,docnode)
            # assign topics along the path
            self.online_sample_topics(doc,levels,docnode,leaf_node)

        score = 0
        for n in range(len(doc)):
            node = docnode[n]
            score += 1-cosine_similarity(node.word_counts.reshape(1,-1),self.background_counts.reshape(1,-1))[0][0]

        return score/len(doc)
      
    
    def online_sample_path(self, doc,levels,docnode):


        ############################################################
        # calculates the prior p(c_d | c_{-d}) in eq. (4)
        ############################################################

        node_weights = {}
        self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        ############################################################
        # calculates the likelihood p(w_d | c, w_{-d}, z) in eq. (4)
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}


        for n in range(len(doc)): # for each word in the doc
            # count the word at each level
            level = levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1


        self.calculate_doc_likelihood(node_weights, level_word_counts)

        ############################################################
        # pick a new path
        ############################################################

        leaf_nodes = [node for node in list(node_weights.keys()) if node.is_leaf()]
        #leaf_nodes = np.array(leaf_nodes)
        weights = np.array([node_weights[node] for node in leaf_nodes])
        weights = np.exp(weights - np.max(weights)) # normalise so the largest weight is 1
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = leaf_nodes[choice]
        
        return node


    def online_sample_topics(self, doc,levels,docnode,leaf_node):

        # initialise level counts
        level_counts = np.zeros(self.num_levels, dtype=np.int)
        for c in levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(self.num_levels, dtype=np.object)
        node = leaf_node
        for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # sample a new level for each word
        level_weights = np.zeros(self.num_levels)
        for n in range(len(doc)):

            w = doc[n]
            word_level = levels[n]

            # remove from model
            level_counts[word_level] -= 1

            # pick new level
            for level in range(self.num_levels):
                level_weights[level] = (self.alpha + level_counts[level]) *                     \
                    (self.eta + path[level].word_counts[w]) /                                   \
                    (self.eta_sum + path[level].total_words)
            level_weights = level_weights / np.sum(level_weights)
            level = self.random_state.multinomial(1, level_weights).argmax()

            # put the word back into the model
            levels[n] = level
            level_counts[level] += 1
            node = path[level]
            docnode[n] = node


    def print_nodes(self, n_words, with_weights):
        self.print_node(self.root_node, 0, n_words, with_weights)

    def print_node(self, node, indent, n_words, with_weights):
        out = '    ' * indent
        out += 'topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)
        out += node.get_top_words(n_words, with_weights)
        print(out)
        for child in node.children:
            self.print_node(child, indent+1, n_words, with_weights)
    

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
    
    dictionary = gensim.corpora.Dictionary(docs)      
    dictionary.filter_extremes(no_below=50)
    dictionary.compactify() 
    corpus = []
    # pack into corpus
    for i in range(len(docs)):
    	words = [dictionary.token2id[word] for word in docs[i] if word in dictionary.token2id.keys()]
    	if len(words)>0 and len(words)<=50:
            corpus.append(words)
    

    
    # modeling
    n_samples = 200       # no of iterations for the sampler
    alpha = 1.0           # smoothing over level distributions
    gamma = 0.3           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.01            # smoothing over topic-word distributions
    num_levels = 3        # the number of levels in the tree
    display_topics = 10   # the number of iterations between printing a brief summary of the topics so far
    n_words = 10          # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights


    hlda = HierarchicalLDA(corpus=corpus, vocab=dictionary, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)    
   
    hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
    
    save_zipped_pickle(hlda,'./experiment/restaurant_nCRP/nCRP_model.p')
    
    
    '''
    # or load existing model
    with gzip.open('./experiment/restaurant_nCRP/nCRP_model.p', 'rb') as f:
        hlda = cPickle.load(f)
    '''

    
    # model testing
    # load test data
    reviews = []
    with open(test_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()

    reviews = [content.split('\t')[0] for content in raw_reviews]
    labels = [int(content.split('\t')[1]) for content in raw_reviews]

    stemmer = PorterStemmer()
    stopset = stopwords.words('english') + ['will', 'also', 'said']
    del_str = string.punctuation + string.digits
    replace = str.maketrans(del_str,' '*len(del_str))
    docs = []

    for i in range(len(reviews)):  
        # strip punctuations and digits
        doc = reviews[i].translate(replace) 
        doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars
        doc = nltk.word_tokenize(doc)
        doc = [w.lower() for w in doc]
        doc = [w for w in doc if w not in stopset]
        doc = [stemmer.stem(w) for w in doc]
        doc = [hlda.vocab.token2id[word] for word in doc if word in hlda.vocab.token2id.keys()]
        docs.append(doc)
    
    scores = np.zeros(len(docs))
    for i in range(len(docs)): 
        logging.info('test doc:' + str(i))  
        # online estimate
        replicates = 10
        for j in range(replicates):
            scores[i] += hlda.online_estimate(docs[i])           
        scores[i] = scores[i]/replicates      
    
    # save scores for later classification    
    f = xlwt.Workbook('encoding = utf-8') 
    sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) 
    sheet1.write(0,0,'nCRP') 
    for i in range(len(scores)):
        sheet1.write(i+1,0,scores[i]) 
    f.save('restaurant_test_score_nCRP.xls')
    
    # ROC_AUC
    nCRP_auc_score = 1-roc_auc_score(labels,scores)      
    print(nCRP_auc_score)
