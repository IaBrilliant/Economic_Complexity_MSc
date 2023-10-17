#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:10:27 2023

@author: brilliant
"""

import networkx as nx
import matplotlib.pyplot as plt

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1

def kruskal_max_spanning_tree(weighted_matrix):
    n = len(weighted_matrix)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if weighted_matrix[i][j] > 0:
                edges.append((i, j, weighted_matrix[i][j]))

    edges.sort(key=lambda x: x[2], reverse=True)

    max_spanning_tree = []
    union_find = UnionFind(n)

    for edge in edges:
        u, v, weight = edge
        if union_find.find(u) != union_find.find(v):
            union_find.union(u, v)
            max_spanning_tree.append(edge)

    return max_spanning_tree


def product_space_plot(trade_matrix, rca_list, trade_flow, df_h2, rca = 0, seed = None):
    
    max_spanning_tree = kruskal_max_spanning_tree(trade_matrix)
    
    # Create a NetworkX graph
    G = nx.Graph()
    for edge in max_spanning_tree:
        u, v, weight = edge
        G.add_edge(u, v, weight=weight)
        
    # Draw the graph
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_lengths = {edge: 1/weight for edge, weight in edge_weights.items()}  # Adjust edge length based on inverse weight
    
    pos = nx.spring_layout(G, weight=None, scale=2, pos=None, fixed=None, iterations=50, threshold=0.0001, dim=2, k=None, seed=seed)
    labels = {i: df_h2.loc[i, 'Product_code'] for i in range(len(df_h2))}
    node_weights = {i: df_h2.loc[i, 'Weight'] for i in range(len(df_h2))}
    
    # Turn on rca-adjusted coloring
    if rca == 0:
        colors = {i: df_h2.loc[i, 'Color'] for i in range(len(df_h2))}
    else:
        colors = ['Springgreen' if (rca_value >= 1) else 'Grey' for rca_value in rca_list ]
    
    plt.figure(figsize = (8, 6))  
    nx.draw_networkx_nodes(G, pos, node_size = [150 / node_weights[node] for node in G.nodes()], node_color = [colors[node] for node in G.nodes()], alpha = 0.75)
    nx.draw_networkx_edges(G, pos, width=1, edge_color='gray')

    label_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in G.nodes()}
    nx.draw_networkx_labels(G, label_pos, labels, font_size = 8, font_color = 'dimgrey')

    plt.title("Maximum Spanning Tree for %s" % (trade_flow))
    print("Edge count %s" % (trade_flow), G.number_of_edges())
    plt.axis('off')
    plt.show()
    
