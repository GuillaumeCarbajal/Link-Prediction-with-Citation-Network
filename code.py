
import numpy as np
from sklearn import preprocessing
import nltk
import csv
import pandas as pd

%%time

import networkx as nx
import community

import random
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv
from bs4 import BeautifulSoup



# train
right = pd.read_csv('graph_features_scaled_train.csv')
del right['Unnamed: 0']
left = pd.read_csv('training_features_6.csv', header = None)
train = pd.concat([left, right], axis=1)

X_train = train.as_matrix()

y_train = [i[2] for i in training_set ]

# test
right = pd.read_csv('graph_features_scaled_test.csv')
del right['Unnamed: 0']
left = pd.read_csv('testing_features_6.csv', header = None)
test = pd.concat([left, right], axis=1)

X_test = test.as_matrix()

############################
## Feature engineering
############################

##### Functions
def common_neighbors(features, G):
    nb_common_neighbors = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        nb_common_neighbors.append(len(sorted(nx.common_neighbors(G, a, b)))) # ajoute le nombre de voisins communs
    return nb_common_neighbors

def Jaccard_coef(features, G):
    J = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        pred = nx.jaccard_coefficient(G, [(a, b)])
        for u, v ,p in pred:
            J.append(p)
    return J

def betweeness_diff(features, G):
	btw = nx.betweenness_centrality(G, 50) 
    btw_diff = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        btw_diff.append(btw[b] - btw[a])
    return btw_diff

def in_link_diff(features, G2):
    diff = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        diff.append(len(G2.in_edges(b)) - len(G2.in_edges(a)))
    return diff

def is_same_cluster(partition, features):
    same_cluster = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        if(partition[a] == partition[b]):
            same_cluster.append(1)
        else:
            same_cluster.append(0)
    return same_cluster



##### Creation of features

## Load graph data
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

# Create a non directed graph

G = nx.Graph()
# extract only the linked nodes from the training set
linked_nodes = [(i[0], i[1]) for i in training_set if not i[2] in ['0']]
G.add_edges_from(linked_nodes)
# Extract the nodes that have no link between
no_linked_nodes = [i[0] for i in training_set if not i[2] in ['1']]
no_linked_nodes.extend([i[1] for i in training_set if not i[2] in ['1']])
# add the nodes that have no links
## NB: NetworkX ignores any nodes that are already present in G
G.add_nodes_from(no_linked_nodes)

############################
# Training set
############################

######################
# Graph Features

# Create the 2 columns "node from" & "node to"
features_train = pd.DataFrame([[i[0], i[1]]for i in training_set])
y = [i[2] for i in training_set]
features_train.columns = ['From', 'To']

# Feature: number of common neighbors
number_common_neighbors = common_neighbors(features_train)
features_train['Nb_common_neighbors'] = number_common_neighbors

# Feature: Jaccard coefficient
Jaccard = Jaccard_coef(features_train)
features_train['Jaccard_coef'] = Jaccard

# Feature: Betweeness centrality
# btw_diff = betweeness_diff(features, btw)
# features_train['Betweeness_diff'] = btw_diff

# Create a directed graph
G2 = nx.DiGraph()
# Create the graph
G2.add_edges_from(linked_nodes)
# add the nodes that have no links
## NB: NetworkX ignores any nodes that are already present in G
G2.add_nodes_from(no_linked_nodes)
nx.is_directed(G2)

# Feature: In-link difference
diff = in_link_diff(features_train, G2)
features_train['In_link_diff'] = diff

# feature: Is Same Cluster
#first compute the best partition
partition = community.best_partition(G)
same_cluster_train = is_same_cluster(partition, features_train)
features_train['Is_same_cluster'] = same_cluster_train

## Save graph features_train in a .csv
features_train.to_csv('graph_features_train.csv')

############################
# Word Embeddings

# Load text data
with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

#training word embeddings on the abstract of the node information 
abstracts = [element[5] for element in node_info ]
print("total nulber of abstracts: %d" %len(abstracts))
abstracts_w = [element.lower().split() for element in abstracts]

# Word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 200    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(abstracts_w, workers=num_workers, size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
        if counter%1000. ==0.:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1.
       
        
    return reviewFeatureVecs

#create word list for each abstract without stop words
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
abstracts_stp =  [[word for word in element.split(" ") if word.lower() not in stpwds] for element in abstracts ]

DataVecs = getAvgFeatureVecs( abstracts_stp, model, num_features )

from sklearn.metrics.pairwise import cosine_similarity as cosine
def isselfcite(source_auth, target_auth): 
    selfcite = 0
    for sauth in source_auth: 
        if sauth in target_auth: 
            selfcite = 1
            break 
    return selfcite

def issamejournal(source_journal, target_journal): 
    
    if source_journal == target_journal: 
        same_journal = 1
    else:
        same_journal = 0
    return same_journal
        
        
def cosine_similarity(s_1, s_2): 
    #remove stopwords 
    s_1 = np.reshape(s_1,(1,-1)  )
    s_2 = np.reshape(s_2,(1,-1)  )
    return round(cosine(s_1,s_2), 5)

# in this baseline we will use 6 basic features:
# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []

#is self citation
self_cite = []

#is published in same journal 
same_journal = []

#cosine  similarity 
cosine_sim = []

#####

from nltk.stem.porter import *
stemmer = PorterStemmer()
counter = 0
for i in xrange(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_journal = source_info[4].lower()
    target_journal = target_info[4].lower()
    
    source_abstract = DataVecs[index_source]
    target_abstract = DataVecs[index_target]
    
    
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    self_cite.append(isselfcite(source_auth,target_auth))
    same_journal.append(issamejournal(source_journal, target_journal))
    cosine_sim.append(cosine_similarity(source_abstract, target_abstract))
   
    counter += 1
    if counter % 1000 == True:
        print counter, "training examples processsed"

training_features = np.array([overlap_title, temp_diff, comm_auth,cosine_sim,same_journal, self_cite]).T
training_features = preprocessing.scale(training_features)

np.savetxt('training_features_6.csv', training_features, delimiter=",")


############################
# Testing set
############################

# Create the 2 columns "node from" & "node to"
features_test = pd.DataFrame([[i[0], i[1]]for i in testing_set])
features_test.columns = ['From', 'To']

# Feature: number of common neighbors
number_common_neighbors_test = common_neighbors(features_test, G)
features_test['Nb_common_neighbors'] = number_common_neighbors_test

# Feature: Jaccard coefficient
Jaccard_test = Jaccard_coef(features_test)
features_test['Jaccard_coef'] = Jaccard_test

# Feature: Betweenness centrality
# btw_diff_test = betweeness_diff(features_test, G)

# Feature: In-link difference
diff_test = in_link_diff(features_test, G2)
features_test['In_link_diff'] = diff_test

# Feature: Is same cluster
same_cluster_test = is_same_cluster(partition, features_test)
features_test['Is_same_cluster'] = same_cluster_test


# Save graph_features test into a csv
features_test.to_csv('graph_features_test.csv')




############################
## Text features

#transforming test features 
# number of overlapping words in title
overlap_title_test = []

# temporal distance between the papers
temp_diff_test = []

# number of common authors
comm_auth_test = []

#is self citation
self_cite_test = []

#is published in same journal 
same_journal_test = []

#cosine  similarity 
cosine_sim_test = []

counter = 0
for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_journal = source_info[4].lower()
    target_journal = target_info[4].lower()
    
    source_abstract = DataVecs[index_source]
    target_abstract = DataVecs[index_target]
    
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    self_cite_test.append(isselfcite(source_auth,target_auth))
    same_journal_test.append(issamejournal(source_journal, target_journal))
    cosine_sim_test.append(cosine_similarity(source_abstract, target_abstract))
   
    counter += 1
    if counter % 1000 == True:
        print counter, "test examples processsed"


testing_features = np.array([overlap_title_test, temp_diff_test, comm_auth_test,cosine_sim_test,same_journal_test, self_cite_test]).T

# scale
testing_features = preprocessing.scale(testing_features)

np.savetxt('testing_features_6.csv', testing_features, delimiter=",")






###############################
## Prediction
###############################

# Load features for train & test sets
## Load data
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

# train
right = pd.read_csv('graph_features_train.csv')
del right['Unnamed: 0']
left = pd.read_csv('training_features_6.csv', header = None)
train = pd.concat([left, right], axis=1)

X_train = train.as_matrix()

y_train = [i[2] for i in training_set ]

# test
right = pd.read_csv('graph_features_test.csv')
del right['Unnamed: 0']
left = pd.read_csv('testing_features_6.csv', header = None)
test = pd.concat([left, right], axis=1)

X_test = test.as_matrix()






from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn import cross_validation
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
import xgboost as xgb


# initialize basic SVM
clf = ExtraTreesClassifier(max_features=None, min_samples_leaf= 20, n_estimators = 500, n_jobs= 3)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Extra trees classifier
clf = ExtraTreesClassifier(max_features=None, min_samples_leaf= 10, n_estimators = 500, n_jobs= 3)
cv = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)

# XG Boost classifier
# 1st tuning
gbm = xgb.XGBClassifier(max_depth=6, n_estimators=500, learning_rate=0.01)
cv = cross_validation.cross_val_score(gbm, X_train, y_train, cv=5)
print np.mean(cv)

# 2nd tuning
gbm = xgb.XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.05)
gbm.fit(X_train, y_train)
pred = gbm.predict(X_test)

#Grid Search
from sklearn.grid_search import GridSearchCV
parameters = {'n_estimators':[500,1000],
        'learning_rate': [0.05, 0.01, 0.001]}

clf = GridSearchCV( xgb.XGBClassifier(max_depth=4), parameters, n_jobs=4, cv=5, verbose = 10)
clf.fit(X_train, y_train)

##########################
## Submission
##########################
predictions = list(pred) 
predictions = zip(range(len(pred)), predictions)
with open("improved_predictions2430_3.csv","wb") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["ID", "category"])
    for row in predictions:
        csv_out.writerow(row)