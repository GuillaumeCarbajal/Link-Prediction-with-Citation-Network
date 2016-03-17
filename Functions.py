import numpy as np
import networkx as networkx
import csv

# load list of links "from" "to"
def link_list(path_of_file):
	with open(path_of_file, "r") as f:
    	reader = csv.reader(f)
    	data_set  = list(reader)

	data_set = [element[0].split(" ") for element in data_set]
	return data_set

## Store the nodes "From" & "To" in a Dataframe
features = pd.DataFrame([[i[0], i[1]]for i in data_set])
features.columns = ['From', 'To']
## Store the labels in y
y = [i[2] for i in data_set]


def create_graph(link_data):
	G = nx.Graph()

	# extract only the linked nodes from the training set
	linked_nodes = [[i[0], i[1]] for i in link_data if not i[2] in ['0']]

	# add the edges
	G.add_edges_from(linked_nodes)

	# Extract the nodes that have no link between
	no_linked_nodes = [i[0] for i in link_data if not i[2] in ['1']]
	no_linked_nodes.extend([i[1] for i in link_data if not i[2] in ['1']])

	# add the nodes that have no links
	## NB: NetworkX ignores any nodes that are already present in G
	G.add_nodes_from(no_linked_nodes)

	return G


## Compute the common neighbors between 2 nodes
def common_neighbors(features, G):
    nb_common_neighbors = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        nb_common_neighbors.append(len(sorted(nx.common_neighbors(G, a, b)))) # ajoute le nombre de voisins communs
    return nb_common_neighbors

## Compute the Link-based Jaccard coefficient between 2 nodes
def Jaccard_coef(features, G):
    J = []
    for i in range(features.shape[0]):
        a = features['From'][i]
        b = features['To'][i]
        pred = nx.jaccard_coefficient(G, [(a, b)])
        for u, v ,p in pred:
            J.append(p)
    return J