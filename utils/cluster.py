import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

n_components = {
    'V1.conv1': 8,
    'V1.conv2': 7,
    'V2.conv2': 5,
    'V4.conv2': 8,
    'IT.conv2': 9,
    'V2.conv_input': 2,
    'V2.conv1': 2,
    'V2.skip': 2,
    'V2.conv3': 2,
    'V4.conv_input': 2,
    'V4.skip': 2,
    'V4.conv1': 2,
    'V4.conv3': 2,
    'IT.conv_input': 2,
    'IT.skip': 2,
    'IT.conv1': 2,
    'IT.conv3': 2,
}

n_components_kernel = {
    'V1.conv1': 2,
    'V1.conv2': 2,
    'V2.conv2': 8,
    'V4.conv2': 4,
    'IT.conv2': 9,
    'V2.conv_input': 5,
    'V2.conv1': 3,
    'V2.skip': 3,
    'V2.conv3': 3,
    'V4.conv_input': 4,
    'V4.skip': 3,
    'V4.conv1': 3,
    'V4.conv3': 4,
    'IT.conv_input': 3,
    'IT.skip': 3,
    'IT.conv1': 3,
    'IT.conv3': 2,
}


def cluster_data(data, max_cluster=15, name='Components'):
    scores = []
    if name not in n_components:
        for i in range(2, max_cluster + 1):
            kmeans = KMeans(init='k-means++', n_clusters=i, max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            scores.append(kmeans.inertia_)
        # print(kmeans.labels_, kmeans.cluster_centers_)
        plt.plot(range(1, max_cluster), scores)
        plt.title(name)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    if len(data.shape) == 2:
        #     The kernel typing, use the same number of components
        kmeans = KMeans(init='k-means++', n_clusters=n_components_kernel[name], max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        scores.append(kmeans.inertia_)
        return kmeans
    else:
        kmeans = KMeans(init='k-means++', n_clusters=n_components[name], max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        scores.append(kmeans.inertia_)
        return kmeans
