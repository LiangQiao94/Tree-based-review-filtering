# -*- coding: utf-8 -*-

import seaborn as sns
import random, bisect
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data/synthetic_corpus/"

class Table:
	def __init__(self):
		self.words = []
		self.topic = None

class Topic:
	def __init__(self, level,word_idx):
		# self.index = index
		self.children = []
		self.parent = None
		self.word = word_idx
		self.level = level
		self.intensity = np.zeros((5,3))

def choose(roulette):
    total = sum(roulette)
    arrow = total * random.random()
    return bisect.bisect(np.cumsum(roulette), arrow)

def get_topic_for_table(root_topic,depth_limit):
    parent_topic = root_topic 
    for i in range(1, depth_limit):
        topics = parent_topic.children
        roulette = np.ones((len(topics) + 1)) 
        
        # index -1, choose current parent topic, and stop
        # otherwise, choose one of children and move down the tree
        topic_index = choose(roulette)
        if topic_index == len(topics):
            return parent_topic
        else:
            parent_topic = topics[topic_index]
    return parent_topic





if __name__ == "__main__":

    N_w = 10               # averge length of documents
    alpha = 0.1 * N_w      # parameter for creating new table
    depth_limit = 3        # depth of tree
    D = 1000               # num of documents in corpus
    V = 13                 # vocabulary size    

    # random seed for generating training data
    random.seed(10)
    np.random.seed(9)
    '''
    # random seed for generating test data
    random.seed(1)
    np.random.seed(0)    
    '''
    
    # generate 3-level topic tree structure
    topics = {}
    for i in range(depth_limit):
        topics[i] = []
    
    root_topic = Topic(0,1)
    topics[0].append(root_topic)
    width = V
    for level in range(1,depth_limit):
        width = (width-1)//3
        for parent_topic in topics[level-1]:          
            word_idx = parent_topic.word+1
            # generate 3 direct children
            for j in range(3):
                topic = Topic(level,word_idx)
                topic.parent = parent_topic
                parent_topic.children.append(topic)
                topics[level].append(topic)
                word_idx += width
                

    write_file = open(data_dir + 'simulated_corpus_10_0.1.txt', 'w')
    whole_intensity = np.zeros((5,3))
    labels = []
    
    # generate simulated documents
    for i in range(D):
        length_d = np.random.poisson(lam=N_w)
        tables = []
        doc = ''
        label1 = "Pos" # "Pos" - level 1 general document
        label2 = "Pos" # "Pos" - level 2 general document
        for j in range(length_d):
            # table assignment
            roulette = np.zeros((len(tables) + 1))
            for j in range(len(tables)):
                table = tables[j]
                roulette[j] = len(table.words)
            roulette[-1] = alpha
            new_table_index = choose(roulette)
            # create new table if last index is chosen
            if new_table_index == len(tables):
                new_table = Table()
                tables.append(new_table)
                new_topic = get_topic_for_table(root_topic,depth_limit)
                new_table.topic = new_topic
            # choose the existing table
            else:
                new_table = tables[new_table_index]
            # topic assignment
            topic = new_table.topic
            if topic.level > 0:
                label1 = "Neg" 
                if topic.level > 1:
                    label2 = "Neg" 
                
            # generate word under the topic
            candidate_w = []
            candidate_w.append(topic.word)
            p_topic = topic.parent
            while(p_topic != None):
                candidate_w.append(p_topic.word)
                p_topic = p_topic.parent               
            word = random.choice(candidate_w)
            # update topic-word intensity
            if word == 1:
                row=0
                col=1
            else:
                row = (word-2) % 4 + 1 
                col = (word-2) // 4                       
            topic.intensity[row,col] += 1
            whole_intensity[row,col] += 1
            doc += str(word)
            doc += " "
            new_table.words.append(word)
        write_file.write(doc + "\t" + label1 + "\t" + label2 +"\n")
        labels.append(label1)
    write_file.close()
    
    
    # visualize original word intensity of different topics       
    mask = np.array([False]*15)
    mask = mask.reshape(5, 3)
    mask[0,0] = True
    mask[0,2] = True
    
    annot = [["","w1",""],
             ["w1,1","w1,2","w1,3"],
             ["w1,1,1","w1,2,1","w1,3,1"],
             ["w1,1,2","w1,2,2","w1,3,2"],
             ["w1,1,3","w1,2,3","w1,3,3"]]
    annot = np.array(annot)
    
    vmax=700
    fig, ax0 = plt.subplots(5, 3,figsize=(10,10))
    topic = root_topic
    sns.heatmap(topic.intensity,cmap="Blues",linewidths=1,annot=annot,fmt = '',cbar=False,ax=ax0[0,1],vmin=0,vmax=vmax,xticklabels=False,yticklabels=False,mask=mask)
    ax0[0,1].set_title('Topic 1')


    for i in range(3):
        topic = root_topic.children[i]
        sns.heatmap(topic.intensity,cmap="Blues",linewidths=1,annot=annot,fmt = '',cbar=False,ax=ax0[1,i],vmin=0,vmax=vmax,xticklabels=False,yticklabels=False,mask=mask)    
        ax0[1,i].set_title('Topic 1-'+str(i+1))
        for j in range(3):
            topic = root_topic.children[i].children[j]
            sns.heatmap(topic.intensity,cmap="Blues",linewidths=1,annot=annot,fmt = '',cbar=False,ax=ax0[j+2,i],vmin=0,vmax=vmax,xticklabels=False,yticklabels=False,mask=mask)
            ax0[j+2,i].set_title('Topic 1-'+str(i+1)+"-"+str(j+1))    
        
        
        
        