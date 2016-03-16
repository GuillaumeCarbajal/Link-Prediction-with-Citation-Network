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

def number_common_neighbors(G):
	nb_common_neighbors = []
	# TO FINISH
	return nb_common_neighbors 