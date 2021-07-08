from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_tsne(mode):
    model = Word2Vec.load(f'embedding_weight/pdf_malware_word2vec_{mode}.model')
    labels = []
    tokens = []
    for word in model.wv.key_to_index.keys():
        tokens.append(model.wv[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40,n_components=2,init='pca')
    new_vales = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_vales:
        x.append(value[0])
        y.append(value[1])
    
    ax, fig = plt.subplots(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i],y[i]),
        xytext=(5,2),
        textcoords='offset points',
        ha='right',
        va='bottom')
    
    plt.savefig(f'word2vec_{mode}.png')
    print(f'save word2vec_{mode}.png')

if __name__ == '__main__':
    show_tsne('cbow')
    show_tsne('skipgram')