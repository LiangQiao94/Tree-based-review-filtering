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
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import _pickle as cPickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_dir = '../data/synthetic_corpus/10_0.1.txt'
test_dir = '../data/synthetic_corpus/10_0.1_test.txt'
exp_dir = './experiment/'
V = 13 # vocabulary size

class NCRPNode(object):

    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0

    def __init__(self, num_levels, vocab, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 1
        self.parent = parent
        self.children = []
        self.level = level
        self.total_words = 0
        self.num_levels = num_levels

        self.vocab = np.array(vocab)
        self.word_counts = np.zeros(len(vocab))

        if random_state is None:
            self.random_state = RandomState(0)
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
        for level in range(1, self.num_levels): # skip the root
            node = node.parent
            node.customers -= 1


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

    def select(self):
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
        sorted_vocab = self.vocab[pos]
        sorted_vocab = sorted_vocab[:n_words]
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

        self.corpus = corpus
        self.vocab = vocab
        self.background_counts = np.zeros(len(vocab))
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

        # initialize a three-level topic tree and fill the topic pointer arrays for
        # every document. Set everything to the single path that
        # we added earlier.
        self.root_node = NCRPNode(self.num_levels, self.vocab)
        for i in range(3):
            p_node = self.root_node
            p_node = p_node.add_child()
            for j in range(3):
                c_node = p_node.add_child()
        
        self.docnode = np.zeros(self.num_documents, dtype=np.object)

        # initialise a single path
        path = np.zeros(self.num_levels, dtype=np.object)
        
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
                level_node = parent_node.select()
                level_node.customers += 1
                path[level] = level_node

            # set the leaf node for this document
            leaf_node = path[self.num_levels-1]
            self.document_leaves[d] = leaf_node

            # randomly assign each word in the document to a level (node) along the path
            self.levels[d] = np.zeros(doc_len, dtype=np.int)
            self.docnode[d] = np.zeros(doc_len, dtype=NCRPNode)
            for n in range(doc_len):
                w = doc[n]
                self.background_counts[w] += 1
                random_level = self.random_state.randint(self.num_levels)
                random_node = path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                self.levels[d][n] = random_level

    def estimate(self, num_samples, display_topics=50, n_words=5, with_weights=True):
        
        print('HierarchicalLDA sampling\n')
        for s in range(num_samples):

            sys.stdout.write('.')

            for d in range(len(self.corpus)):
                self.sample_path(d)

            for d in range(len(self.corpus)):
                self.sample_topics(d)

            if (s > 0) and ((s+1) % display_topics == 0):
                print(f" {s+1}")
                self.print_nodes(n_words, with_weights)
            
        self.intensity_compute()
        
    
    def compute_score(self):
        avg_score = []
        sum_score = []
        for d in range(len(self.corpus)):
            score = 0
            for n in range(len(self.corpus[d])):
                node = self.docnode[d][n]
                score += cosine_similarity(node.word_counts.reshape(1,-1),self.background_counts.reshape(1,-1))[0][0]
            sum_score.append(score)
            avg_score.append(score/len(self.corpus[d]))
        return avg_score,sum_score



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

        leaf_nodes = [node for node in list(node_weights.keys()) if node.is_leaf()]
        #leaf_nodes = np.array(leaf_nodes)
        weights = np.array([node_weights[node] for node in leaf_nodes])
        weights = np.exp(weights - np.max(weights)) # normalise so the largest weight is 1
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = leaf_nodes[choice]

        # if we picked an internal node, we need to add a new path to the leaf
        #if not node.is_leaf():
        #    node = node.get_new_leaf()

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
            child_weight = log( float(child.customers) / (node.customers) )
            self.calculate_ncrp_prior(node_weights, child, weight + child_weight)

        node_weights[node] = weight

    def calculate_doc_likelihood(self, node_weights, level_word_counts):

        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(self.num_levels)
        for level in range(1, self.num_levels):  # skip the root

            word_counts = level_word_counts[level]
            total_tokens = 0

            for w in word_counts:
                count = word_counts[w]
                for i in range(count):  
                    new_topic_weights[level] += log((self.eta + i) / (self.eta_sum + total_tokens))
                    total_tokens += 1

        self.calculate_word_likelihood(node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0)

    def calculate_word_likelihood(self, node_weights, node, weight, level_word_counts, new_topic_weights, level):

        # first calculate the likelihood of the words at this level, given this topic
        node_weight = 0.0
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
            self.calculate_word_likelihood(node_weights, child, weight + node_weight,
                                           level_word_counts, new_topic_weights, level+1)

        # finally if this is an internal node, add the weight of a new path
        level += 1
        while level < self.num_levels:
            node_weight += new_topic_weights[level]
            level += 1

        node_weights[node] += node_weight

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
                    (self.eta + path[level].word_counts[w]) /                                   \
                    (self.eta_sum + path[level].total_words)
            level_weights = level_weights / np.sum(level_weights)

            level = self.random_state.multinomial(1, level_weights).argmax()

            # put the word back into the model
            doc_levels[n] = level
            level_counts[level] += 1
            node = path[level]
            node.word_counts[w] += 1
            node.total_words += 1
            self.docnode[d][n] = node
            
    def online_estimate(self,doc):
        path = np.zeros(self.num_levels, dtype=np.object)
        # populate nodes into the path of this document

        path[0] = self.root_node
        for level in range(1, self.num_levels):
        # at each level, a node is selected by its parent node based on the CRP prior
            parent_node = path[level-1]
            level_node = parent_node.select()
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

        logp = 0
        score = 0
        for n in range(len(doc)):
            w = doc[n]
            node = docnode[n]
            score += cosine_similarity(node.word_counts.reshape(1,-1),self.background_counts.reshape(1,-1))[0][0]
            logp += np.log( (self.eta + node.word_counts[w]) / (self.eta_sum + node.total_words) )
        return logp, score/len(doc)

        
    
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
    
    def intensity_compute(self):
        self.root_node.intensity = np.zeros((5,3))
        self.root_node.intensity[0,1] += self.root_node.word_counts[0]
        for i in range(1,5):
            for j in range(3):
                v = j*4+i   
                self.root_node.intensity[i,j] += self.root_node.word_counts[v]
        for topic in self.root_node.children:
            topic.intensity = np.zeros((5,3))
            topic.intensity[0,1] += topic.word_counts[0]
            for i in range(1,5):
                for j in range(3):
                    v = j*4+i   
                    topic.intensity[i,j] += topic.word_counts[v]
            for leaf_topic in topic.children:
                leaf_topic.intensity = np.zeros((5,3))
                leaf_topic.intensity[0,1] += leaf_topic.word_counts[0]
                for i in range(1,5):
                    for j in range(3):
                        v = j*4+i   
                        leaf_topic.intensity[i,j] += leaf_topic.word_counts[v]


def load_vocab(file_name):
    with open(file_name, 'rb') as f:
        vocab = []
        reader = csv.reader(f)
        for row in reader:
            idx, word = row
            stripped = word.strip()
            vocab.append(stripped)
        return vocab

def load_corpus(file_name):
    with open(file_name, 'rb') as f:
        corpus = []
        reader = csv.reader(f)
        for row in reader:
            doc = []
            for idx_and_word in row:
                stripped = idx_and_word.strip()
                tokens = stripped.split(' ')
                if len(tokens) == 2:
                    idx, word = tokens
                    doc.append(int(idx))
            corpus.append(doc)
        return corpus

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)
        
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


if __name__ == "__main__":
    
    # load training data
    reviews = []
    
    with open(train_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()
        
    reviews = [content.split('\t')[0] for content in raw_reviews]
            
    vocab = [str(i+1) for i in range(V) ]
       
    docs = []
    for doc in reviews:  
        doc = nltk.word_tokenize(doc)
        doc = [int(w)-1 for w in doc]
        if len(doc)>0:
            docs.append(doc)

    # modeling
    param = '10_0.1_nCRP'
    n_samples = 100       # no of iterations for the sampler
    alpha = 1.0           # smoothing over level distributions
    eta = 0.01            # smoothing over topic-word distributions
    num_levels = 3        # the number of levels in the tree
    display_topics = 20   # the number of iterations between printing a brief summary of the topics so far
    n_words = 10          # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights

    hlda = HierarchicalLDA(docs, vocab, alpha=alpha, eta=eta, num_levels=num_levels)
    hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
    '''
    # save model for further use    
    if not os.path.isdir(exp_dir + param):
        os.mkdir(exp_dir + param)
    save_zipped_pickle(hlda, exp_dir + param + '/model.p')
        
    # or load existing model
    with gzip.open(exp_dir + param + '/model.p', 'rb') as f:
        hlda = cPickle.load(f)
    '''
    
    # visualize results - heatmap for topic distribution over vocabulary
    mask = np.array([False]*15)
    mask = mask.reshape(5, 3)
    mask[0,0] = True
    mask[0,2] = True

    fig, ax0 = plt.subplots(5, 3,figsize=(10,10))
    topic = hlda.root_node
    sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[0,1],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
    ax0[0,1].set_title('Topic 1')
   

    for i in range(3):
        topic = hlda.root_node.children[i]
        sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[1,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)    
        ax0[1,i].set_title('Topic 1-'+str(i+1))
        for j in range(3):
            topic = hlda.root_node.children[i].children[j]
            sns.heatmap(topic.intensity/np.sum(topic.intensity),cmap="Blues",annot=True,fmt='.1%',linewidths=1,ax=ax0[j+2,i],vmin=0,vmax=1,xticklabels=False,yticklabels=False,mask=mask,cbar=False)
            ax0[j+2,i].set_title('Topic 1-'+str(i+1)+"-"+str(j+1))  
    
    
    # model testing
    # load test data
    reviews = []
    with open(test_dir,encoding='utf-8') as fcorpus:
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
        doc = [int(w)-1 for w in doc]
        N += len(doc) 
        avg_logp = 0
        for j in range(replicate):
            logp[i,j],avg_scores[i,j] = hlda.online_estimate(doc)

    logp = np.sum(logp,axis=0)/N
    print(np.mean(logp),np.std(logp)/np.sqrt(replicate))

    
    # ROC_AUC
    avg_roc_scores1 = np.zeros(replicate)
    for j in range(replicate):
        avg_roc_scores1[j] = roc_auc_score(labels1,avg_scores[:,j])
    #AUC1    
    print(np.mean(avg_roc_scores1),np.std(avg_roc_scores1)/np.sqrt(replicate))

    
    avg_roc_scores2 = np.zeros(replicate)
    for j in range(replicate):
        avg_roc_scores2[j] = roc_auc_score(labels2,avg_scores[:,j])
    # AUC2    
    print(np.mean(avg_roc_scores2),np.std(avg_roc_scores2)/np.sqrt(replicate))
    