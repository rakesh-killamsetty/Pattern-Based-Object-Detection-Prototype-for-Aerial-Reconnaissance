import networkx as nx
import numpy as np

def build_object_graph(bboxes):
    """
    bboxes: list of (x1, y1, x2, y2) tuples
    Returns: NetworkX graph with nodes for each object and edges weighted by spatial distance
    """
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, bbox=box)
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            c1 = np.array([(bboxes[i][0]+bboxes[i][2])/2, (bboxes[i][1]+bboxes[i][3])/2])
            c2 = np.array([(bboxes[j][0]+bboxes[j][2])/2, (bboxes[j][1]+bboxes[j][3])/2])
            dist = np.linalg.norm(c1-c2)
            G.add_edge(i, j, weight=dist)
    return G

def find_major_object(G, method='centrality'):
    """
    G: NetworkX graph
    method: 'centrality' or 'size'
    Returns: node index of major object
    """
    if method == 'centrality':
        centrality = nx.degree_centrality(G)
        return max(centrality, key=centrality.get)
    elif method == 'size':
        sizes = {n: (G.nodes[n]['bbox'][2]-G.nodes[n]['bbox'][0]) * (G.nodes[n]['bbox'][3]-G.nodes[n]['bbox'][1]) for n in G.nodes}
        return max(sizes, key=sizes.get)
    else:
        raise ValueError('Unknown method for major object identification') 