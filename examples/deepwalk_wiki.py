
import numpy as np
import csv
import ast
from pathlib import Path

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
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


if __name__ == "__main__":
    G = nx.read_edgelist('../baseline_full_edges.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    # np.save('embeddings.npy', embeddings)
    dict = sorted(embeddings.items(), key=lambda d: d[0])
    # # embeddings = ast.literal_eval(embeddings)
    # header = list(embeddings.keys())
    # with open('test.csv', 'a', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f,fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
    #     writer.writeheader()  # 写入列名
    #     writer.writerows(embeddings)  # 写入数据
    # datas = []
    # header = list(embeddings.keys())
    # datas.append(embeddings)
    #
    # with open('test.csv', 'a', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
    #     writer.writeheader()  # 写入列名
    #     writer.writerows(datas)  # 写入数据
    qianzhui=Path(__file__).stem
    dataname='baseline_full_edges'
    embeddings = {k: v for k, v in sorted(embeddings.items(), key=lambda item: item[0])}
    np.save(qianzhui+'_'+dataname+'embeddings.npy', embeddings)
    i6000={k: v for k, v in embeddings.items() if int(k) <2206}
    ws={k: v for k, v in embeddings.items() if int(k)  >= 2206}
    np.save(qianzhui+'_'+dataname+'_ws_embeddings.npy', ws)
    np.save(qianzhui+'_'+dataname+'_i6000_embeddings.npy', i6000)

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
