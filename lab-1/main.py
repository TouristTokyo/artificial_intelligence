import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn import preprocessing

file_name = "usa_elections.dat"
linkage_method = 'ward'
linkage_metric = 'euclidean'


def read_data(file_path):
    input_data = []
    with open(file_path, "r", encoding="utf8") as f:
        f.readline()
        for line in f.readlines():
            input_data.append([float(x) for x in line.replace('NA', '0').split(";")[1:]])
    return input_data


def read_headers(file_path):
    headers = {}
    with open(file_path, "r", encoding="utf8") as f:
        headers["headers"] = f.readline().split(";")[1:]
        cities = []
        for line in f.readlines():
            cities.extend(line.split(";")[:1])
        headers["cities"] = cities
    return headers


def hierarchical_clustering(normal_data, labels):
    distance_matrix = linkage(normal_data, method=linkage_method, metric=linkage_metric)
    plt.figure(figsize=(25, 15))
    dendrogram(distance_matrix, leaf_font_size=15, labels=labels)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('Distance')
    plt.show()


def data_set_scaling(data):
    plt.figure(figsize=(10, 7))
    for i in range(1, 4):
        plt.scatter(data[:, i - 1], data[:, i])
    plt.show()


def elbow_method(new_data):
    WCSS = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=400, random_state=0)
        kmeans.fit(new_data)
        WCSS.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 11), WCSS, marker='o')
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()


def normalization(data):
    scaler = preprocessing.MinMaxScaler().fit(data)
    new_data = scaler.transform(data)
    return new_data


if __name__ == '__main__':
    data = np.array(read_data(file_name))
    headers = read_headers(file_name)
    normal_data = normalization(data)
    hierarchical_clustering(normal_data, headers["cities"])
    data_set_scaling(normal_data)
    elbow_method(normal_data)
