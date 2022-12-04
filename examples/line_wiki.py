
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

from pathlib import Path

if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.read_edgelist('../baseline_pos_edges.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    i6000={k: v for k, v in embeddings.items() if int(k) <2206}
    ws={k: v for k, v in embeddings.items() if int(k)  >= 2206}
    qianzhui=Path(__file__).stem
    dataname='baseline_pos_edges'
    np.save(qianzhui+'_'+dataname+'_ws_embeddings.npy', ws)
    np.save(qianzhui+'_'+dataname+'_i6000_embeddings.npy', i6000)

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
