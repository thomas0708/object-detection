import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
import math
import random

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 10 or np.count_nonzero(y == 0) > 10:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    # print(box_area + cluster_area - intersection)

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def mini_batch_kmeanspp(boxes, k, batch_size, num_iter, dist=np.median, replacement=True):
    """The mini-batch k-means algorithms (Sculley et al. 2007) for the k-centers problem.
    boxes : data matrix
    k : number of clusters
    batch_size : size of the mini-batches
    num_iter : number of iterations
    replacement: whether to sample batches with replacement or not.
    """
    # clusters = np.zeros((k, 2, num_iter))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = np.array(kpp_centers(boxes, k)) # kmeans++ initialization

    for i in range(num_iter):
        # Sample a mini batch:
        if replacement:
            boxes_batch = boxes[np.random.choice(boxes.shape[0], batch_size, replace=True)]
        else:
            boxes_batch = boxes[batch_size*i:batch_size*(i+1)]

        # clusters = np.array(kpp_centers(boxes_batch, k))
        distances_batch = np.empty((batch_size, k))
        last_clusters_batch = np.zeros((batch_size,))
        rows_batch = batch_size
        temp_clusters = clusters.copy()
        # print(clusters)

        while True:
            for row in range(rows_batch):
                distances_batch[row] = 1 - iou(boxes_batch[row], temp_clusters)

            nearest_clusters = np.argmin(distances_batch, axis=1)

            if (last_clusters_batch == nearest_clusters).all():
                break

            for cluster in range(k):
                temp_clusters[cluster] = dist(boxes_batch[nearest_clusters == cluster], axis=0)

            last_clusters_batch = nearest_clusters
            # print(temp_clusters)
            # clusters[:,:,i] = temp_clusters

    # clusters = np.mean(clusters, axis=1) 
    clusters = temp_clusters.copy()
    # print(clusters)
    return clusters

def mini_batch_kmeans(boxes, clusters, b, t, replacement=True):
    """The mini-batch k-means algorithms (Sculley et al. 2007) for the
    k-centers problem.
    boxes : data matrix
    clusters : initial centers
    b : size of the mini-batches
    t : number of iterations
    replacement: whether to sample batches with replacement or not.
    """
    clusters = clusters.copy()
    for i in range(t):
        # Sample a mini batch:
        if replacement:
            boxes_batch = boxes[np.random.choice(boxes.shape[0], b, replace=True)]
        else:
            boxes_batch = boxes[b*i:b*(i+1)]

        V = np.zeros(clusters.shape[0])
        idxs = np.empty(boxes_batch.shape[0], dtype=np.int)
        # Assign the closest centers without update for the whole batch:
        for j, x in enumerate(boxes_batch):
            idxs[j] = np.argmin(((clusters - x)**2).sum(1))

        # Update centers:
        for j, x in enumerate(boxes_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            clusters[idxs[j]] = (1.0 - eta) * clusters[idxs[j]] + eta * x

    return clusters


#################################################
def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers