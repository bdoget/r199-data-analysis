import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import json

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

from sklearn import preprocessing

def loadR199():
    """
        Loads R199 Data and preprocesses data.
        Returns df, dfnorm
        
        df - raw data,
        dfnorm - preprocessed data
    """
    df = pd.read_excel("data/R199_Data.xlsx")
    # Preprocess: fill NA and standard scale
    sc_X = preprocessing.StandardScaler()
    dfnorm = pd.DataFrame(sc_X.fit_transform(df.fillna(0)))
    dfnorm.columns = df.columns.values
    return df, dfnorm

def loadAggregates():
    """
        Loads Aggregate variables of Research Quantity, Teaching Quantity, and Teaching Quality
    """
    aggregates = json.load(open("data/aggregates.json"))
    r_quant = aggregates["research_quant"]
    t_quant = aggregates["teaching_quant"]
    t_qual = aggregates["teaching_qual"]
    return r_quant, t_quant, t_qual


def graphPCA(X, n, cluster_arr):
    """
    """
    pca = PCA(n_components = 2)
    Xn = pca.fit_transform(X)

    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.0*i/n) for i in range(n)]

    for i in range(n):
        plt.scatter(Xn[cluster_arr == i, 0], Xn[cluster_arr == i, 1], s = 100, color = colors[i])
    centroids_pca = np.transpose([np.mean(Xn[cluster_arr == i,:], axis=0) for i in range(n)])
    plt.scatter(centroids_pca[0, :], centroids_pca[1, :], s = 300, c = 'black', label = 'Centroids')

    plt.title('Clusters of areas with pca')
    plt.xlabel('factor1')
    plt.ylabel('factor2')
    plt.legend()

    plt.show()
    return None # return axes


def graphKPCA(X, n, cluster_arr):
    kpca = KernelPCA(n_components = 2, kernel = 'rbf')
    Xn = kpca.fit_transform(X)

    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.0*i/n) for i in range(n)]

    for i in range(n):
        plt.scatter(Xn[cluster_arr == i, 0], Xn[cluster_arr == i, 1], s = 100, color = colors[i])
    centroids_pca = np.transpose([np.mean(Xn[cluster_arr == i,:], axis=0) for i in range(n)])
    plt.scatter(centroids_pca[0, :], centroids_pca[1, :], s = 300, c = 'black', label = 'Centroids')

    plt.title('Clusters of areas with kpca')
    plt.xlabel('factor1')
    plt.ylabel('factor2')
    plt.legend()
    plt.show()
    return None


def groupByQuartile(df_desc : pd.DataFrame, df_copy : pd.DataFrame, out_file=""):
    """
    Creates new copy of df_copy where each entry is approximated to the quartiles of df_desc.
    Assumes that df_desc and df_copy are normal populated DataFrames.
    NOTE: Must have the same column names.
    Returns df_closest_quartile which places "Qn" in every entry.

    """
    df_description = df_desc.describe()
    df_closest_quartiles = pd.DataFrame(columns=df_copy.columns, index=range(len(df_copy)))

    for feature in df_copy.columns:
        # Extract quartile values for the feature
        Q1 = df_description.at['25%', feature]
        Q2 = df_description.at['50%', feature]  # median
        Q3 = df_description.at['75%', feature]
        Q4 = df_description.at['max', feature]  # Max value for Q4 comparison
        Q0 = df_description.at['min', feature]
        
        # Iterating through each centroid to determine the closest quartile
        for index, entry in df_copy.iterrows():
            value = entry[feature]
            distances = {
                'Q0': abs(value - Q0),
                'Q1': abs(value - Q1),
                'Q2': abs(value - Q2),
                'Q3': abs(value - Q3),
                'Q4': abs(value - Q4) 
            }
            # Determine the closest quartile by finding the min distance
            closest_quartile = min(distances, key=distances.get)
            df_closest_quartiles.at[index, feature] = closest_quartile

    # Now closest_quartiles_df contains the closest quartile for each feature of each centroid
    if out_file != "":
        df_closest_quartiles.to_csv(out_file)
    return df_closest_quartiles


def count_quartiles(df, columns : list[str]):
    """
    Function to count quartiles occurrences within a given set of columns for each centroid
    """
    filtered_df = df[columns]
    counts = filtered_df.apply(lambda x: x.value_counts(), axis=1).fillna(0)
    
    # Ensure all expected quartile values are present in the result, even if they're 0
    expected_quartiles = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']
    for quartile in expected_quartiles:
        if quartile not in counts:
            counts[quartile] = 0
    return counts

