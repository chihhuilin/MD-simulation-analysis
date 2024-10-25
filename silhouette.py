import numpy as np
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
# from scipy.cluster.hierarchy import dendrogram


'''
The size of the input data(matrix) should (n_samples, n_features)
'''
###############################################################################
### Create a matrix that stores the pairwise distance                       ###
### for data(n_samples, n_fearures).                                        ###
###############################################################################
def pairwise_distance(data, method =''):
    
    if method == 'scikit':
        import sklearn.metrics as sk
        dis_m = sk.pairwise_distances(data)
    else:
        n_samples = data.shape[0]        
        dis_m = np.zeros((n_samples, n_samples))
        ### calculate the pariwise distance
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dis_m[i, j] = sum((data[i] - data[j]) ** 2) ** 0.5
                # dis_m[j, i] = dis_m[i, j]

    return dis_m


###############################################################################
### Do hierarchical clustering using functions from sicpy.                  ###
###############################################################################
def hierarchical_clustering(data, n_clust):

    ### calculate pairwise distance(pdist()) and do linkage(average())
    Z = average(pdist(data))
    cluster_ids = fcluster(Z, n_clust, criterion='maxclust')

    return Z, cluster_ids


###############################################################################
### Find the neighboring cluster for each sample.                           ###
###############################################################################
def get_neighbor_clust(dis_m, cluster_ids):

    # dis_m = pairwise_distance(data)    
    n_samples = dis_m.shape[0]
    sample_ids = np.array([i for i in range(n_samples)])
    k = len(set(cluster_ids))
    neighbor_clust_ids = []
    
    for i in range(n_samples):
        data_clust = cluster_ids[i]
        
        m = []        
        for j in range(k):
            ### skip the cluster which "i" itself located
            if data_clust == j+1:
                continue
            else:
                m.append([j+1, sample_ids[cluster_ids == j+1]])
                
        ### calculate avg. distances to all other clusters for every sample and
        ### append them to their coresponding index in m matrix.
        for p in range(k-1):
            total_dis = 0
            n = len(m[p][1])
            
            for q in m[p][1]:
                if i <= q:
                    total_dis += dis_m[i][q]
                else:
                    total_dis += dis_m[q][i]
                
            avg_dis = total_dis / n
            m[p].append(avg_dis)
        
        ### compare the avg. distance and assign the closest as the neighbor cluster
        dis_arr = [m[i][2] for i in range(k-1)]
        min_id = np.argmin(dis_arr)
        neighbor_clust_ids.append(m[min_id][0])
        
    return neighbor_clust_ids


###############################################################################
### Calculate silhouette score for each sample.                             ###
###############################################################################
def get_silhouette(pdist_matrix, cluster_ids, neighbor_ids):
    
    ### check input array size
    assert (pdist_matrix.shape[0] == len(cluster_ids) or len(cluster_ids) == len(neighbor_ids))
    
    n_samples = len(cluster_ids)    
    sample_ids = np.array([i for i in range(n_samples)])
    score_arr = []
    
    ### loop through every sample to get the score
    for data_id in range(n_samples):
        data_clust = cluster_ids[data_id]
        neighbor_clust = neighbor_ids[data_id]
        print(data_id)

        ### score = 0 when there's only sample itself in the cluster
        if len(sample_ids[cluster_ids == data_clust]) == 1: 
            score = 0

        else:

            a = 0 ### intercluster distance
            b = 0 ### intracluster distance
            
            
############### Too slow starts from here ###############

            ### loop to get the avg. inter/intra distance for the sample "data_id"
            for i in range(n_samples):
                ### skip datapoint itself
                if i == data_id:
                    continue
                ### skip irrelavent clusters
                elif cluster_ids[i] != data_clust and cluster_ids[i] != neighbor_clust:
                    continue
                ### calculate a (intercluster)
                elif cluster_ids[i] == data_clust:
                    if data_id <= i:
                        a += pdist_matrix[data_id, i]
                    else:
                        a += pdist_matrix[i, data_id]
                ### calculate b (intracluster)
                elif cluster_ids[i] == neighbor_clust:
                    if data_id <= i:
                        b += pdist_matrix[data_id, i]
                    else:
                        b += pdist_matrix[i, data_id]
                else:
                    assert False, "Something is wrong!"
        
            a = a / (len(sample_ids[cluster_ids == data_clust])-1)
            b = b / len(sample_ids[cluster_ids == neighbor_clust])
        
            if a < b:
                score = 1 - (a/b)
            elif a > b:
                score = (b/a) - 1
            elif a == b:
                score = 0
                                
        score_arr.append(score)
        
    return score_arr


def avg_silhouette(score_arr):
    
    avg_score = np.array(score_arr).sum() / len(score_arr)

    return avg_score


###############################################################################
### Try 2-15 clusters and calculate the average silhouette score of each n. ###
### Return unique clustering results and the silhouette score of each n.    ###
###############################################################################
def try_n_clusters(data, minclust, maxclust, method=''):
        
    maxclust += 1
    ### check min/ max cluster to make sure 2 <= no. of clusters <= n_sample-1
    assert (maxclust <= data.shape[0] and minclust >= 2), \
        "maxclust needs to be smaller then (n_samples-1) and minclust needs to be bigger then 1"
        
    ### store the coresponding cluster ids for each number of clusters in m  
    m = []
    
    for i in range(minclust, maxclust):
        Z, cluster_ids = hierarchical_clustering(data, i)
        n_clusters = len(set(cluster_ids))
    ### Try 2-10 clusters and store the unique clustering results in matrix m
        if i == minclust:
            m.append([n_clusters, cluster_ids])
        elif n_clusters != m[-1][0]:
            m.append([n_clusters, cluster_ids])
        else:
            continue
    dis_m = pairwise_distance(data, 'scikit')

    silhouette_scores = []
    
    if method == 'scikit':
        for j in range(len(m)):
    
            ### get the number of clusters in the j th clustering result
            n = m[j][0]
            ### get the j th clustering result
            cluster_ids = m[j][1]
            ### get the score for every sample using scikit
            score_arr = silhouette_samples(data, cluster_ids)
            ### get average score using "cluster_ids" as label using function
            ### from scikit
            avg_score = silhouette_score(data, cluster_ids)
            silhouette_scores.append([n, avg_score, score_arr])

    else:
        for j in range(len(m)):
            
            ### get the number of clusters in the j th clustering result
            n = m[j][0]
            ### get the j th clustering result
            cluster_ids = m[j][1]
            ### get the neighboring cluster id for each sample using the j th clustering result
            neighbor_ids = get_neighbor_clust(dis_m, cluster_ids)
            ### get the score for each sample using the j th clustering result
            score_arr = get_silhouette(dis_m, cluster_ids, neighbor_ids)
            ### get the avg. score using the j th clustering result
            avg_score = avg_silhouette(score_arr)
            silhouette_scores.append([n, avg_score, score_arr])

    return silhouette_scores

def plot_silhouette(silhouette_scores):
    
    
    
    return


###############################################################################
### Read the result from try_n_clusters and see which n has the best        ###
### average silhouette score.                                               ###
### Return the optimal number of clusters(n) and the matrix                 ###
### silhouette_scores that stores all scores for each sample at different n.###
###############################################################################
def optimal_n_clustering(data, minclust=2, maxclust=15, method='scikit' ):
    
    if method == 'scikit':
        silhouette_scores = np.array(try_n_clusters(data, minclust, maxclust, method), dtype=object)
        ### the index of the optimal number of cluster having the highest score
        max_id = np.argmax(silhouette_scores[:, 1])
        ### the optimal number of cluster which has the highest score
        optimal_n, score= silhouette_scores[max_id, 0], silhouette_scores[max_id, 1]        
        
    else:    
        ### cast silhouette_scores into a numpy array for better indexing
        silhouette_scores = np.array(try_n_clusters(data, minclust, maxclust), dtype=object)
        ### the index of the optimal number of cluster having the highest score
        max_id = np.argmax(silhouette_scores[:, 1])
        ### the optimal number of cluster which has the highest score
        optimal_n, score= silhouette_scores[max_id, 0], silhouette_scores[max_id, 1]

    print(f'The optimal number of clusters is {optimal_n}, the average silhouette score is {score:.4f}.')
    
    Z, cluster_ids = hierarchical_clustering(data, optimal_n)

    return optimal_n, cluster_ids, Z, silhouette_scores