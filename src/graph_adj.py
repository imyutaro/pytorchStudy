import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataloader import assoc 

# filename = "T10I4D100K"
filename = "retail"
print(f"Data is {filename}")

adjpath = f"../adj/{filename}.npy"
adj = np.load(adjpath)

data = assoc(filename)
G = nx.from_numpy_matrix(adj)

# relabels
num_items = data.item_len()
mapping = {i: sorted(data.item)[i] for i in range(num_items)}
nx.relabel_nodes(G, mapping, copy=False)

# positions for all nodes
plt.figure(figsize=(15,15))
pos = nx.spring_layout(G, k=1.5)
# pos = nx.kamada_kawai_layout(G)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.2, node_color="b")

# edges
nx.draw_networkx_edges(G, pos, alpha=0.01, edge_color='gainsboro')

nx.draw_networkx(G, pos, with_labels=False)
plt.axis("off")
plt.savefig("tmp_graph.png")

