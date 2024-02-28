import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.cluster import Birch

from sklearn import preprocessing

def loadR199():
    """
        Loads R199 Data and preprocesses data.
        Returns df, dfnorm
        
        df - raw data
        dfnorm - preprocessed data
    """
    df = pd.read_excel("data/R199_Data.xlsx")
    # Preprocess: fill NA and standard scale
    sc_X = preprocessing.StandardScaler()
    dfnorm = pd.DataFrame(sc_X.fit_transform(df.fillna(0)))
    dfnorm.columns = df.columns.values
    return df, dfnorm

def graphPCA(X, n, y_kmeans):
    """
    """
    pca = PCA(n_components = 2)
    Xn = pca.fit_transform(X)

    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.0*i/n) for i in range(n)]

    for i in range(n):
        plt.scatter(Xn[y_kmeans == i, 0], Xn[y_kmeans == i, 1], s = 100, color = colors[i])
    centroids_pca = np.transpose([np.mean(Xn[y_kmeans == i,:], axis=0) for i in range(n)])
    plt.scatter(centroids_pca[0, :], centroids_pca[1, :], s = 300, c = 'black', label = 'Centroids')

    plt.title('Clusters of areas with pca')
    plt.xlabel('factor1')
    plt.ylabel('factor2')
    plt.legend()

    plt.show()
    return None # return axes



def graphKPCA(X, n, y_kmeans):
    kpca = KernelPCA(n_components = 2, kernel = 'rbf')
    Xn = kpca.fit_transform(X)

    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.0*i/n) for i in range(n)]

    for i in range(n):
        plt.scatter(Xn[y_kmeans == i, 0], Xn[y_kmeans == i, 1], s = 100, color = colors[i])
    centroids_pca = np.transpose([np.mean(Xn[y_kmeans == i,:], axis=0) for i in range(n)])
    plt.scatter(centroids_pca[0, :], centroids_pca[1, :], s = 300, c = 'black', label = 'Centroids')

    plt.title('Clusters of areas with kpca')
    plt.xlabel('factor1')
    plt.ylabel('factor2')
    plt.legend()
    plt.show()
    return None
