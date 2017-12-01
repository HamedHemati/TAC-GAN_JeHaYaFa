from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle


# returns first two principal components of the input data
def getPCA(datapoints):
    X = np.array(datapoints)
    pca = PCA(n_components=2)
    return pca.fit_transform(X)


# return the output of t-SNE embedding (2-dim)
def gettSNE(datapoints):
    X = np.array(datapoints)
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(X)


def plot_PCA(X,Y):
    res_PCA = zip(*getPCA(X))
    x_pca = list(res_PCA[0])
    y_pca = list(res_PCA[1])
    plt.scatter(x_pca, y_pca)
    Y = map(str,Y)
    for idx,(i,j) in enumerate(zip(x_pca,y_pca)):
        plt.annotate(Y[idx], xy=(i,j), bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('PCA')
    plt.savefig('pca.png')
    plt.close()


def plot_tSNE(X,Y):
    res_tSNE = zip(*gettSNE(X))
    x_tsne = list(res_tSNE[0])
    y_tsne = list(res_tSNE[1])
    plt.scatter(x_tsne, y_tsne)
    Y = map(str,Y)
    for idx,(i,j) in enumerate(zip(x_tsne,y_tsne)):
        plt.annotate(Y[idx], xy=(i,j), bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('t-SNE')
    plt.savefig('tsne.png')
    plt.close()    


# reads the data file which contains the extracted features
def readFile(path):
    with open(path, "rb") as fp:
        correct = pickle.load(fp) 
        correct_labels = pickle.load(fp) 
        false = pickle.load(fp) 
        false_labels = pickle.load(fp) 
    # return the features from either false or true matches
    return false, false_labels


def main():
    X,Y = readFile(path='features')
    # set the number of datapoints to be for the drawing
    plot_PCA(X[100:300],Y[100:300])
    plot_tSNE(X[100:300],Y[100:300])
    
    
if __name__ == '__main__':
    main()