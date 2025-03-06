from sklearn.cluster import SpectralBiclustering as Biclustering
import numpy as np
from numpy.linalg import norm


def cosine_similarity(A, B):
    # compute cosine similarity
    cosine = np.dot(A, B) / (norm(A, axis=0) * norm(B))
    return cosine


def euclidean(point1, point2):
    # calculate Euclidean distance
    # using linalg.norm() method
    dist = norm(point1 - point2)
    return dist


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def Model_Biclustering(Feat, Target):
    clustering = Biclustering(n_clusters=2, svd_method='randomized',
                              n_init=10)
    clustering.fit(Feat)
    uni = np.unique(clustering.row_labels_)

    Query = Feat[0]
    Q_T = Target[0]
    F_matrix = clustering.get_submatrix(uni, Feat)
    Q_matrix=F_matrix[0]
    Precision = []
    Recall = []
    F1Score = []

    cosine = []
    euclid = []
    jac = []
    for j in range(len(F_matrix)):
        cosine.append(cosine_similarity(Q_matrix, F_matrix[j]))
        euclid.append(euclidean(Q_matrix, F_matrix[j]))
        jac.append(jaccard(Q_matrix.tolist(), F_matrix[j].tolist()))
    sim = np.asarray(cosine) + (1 / np.asarray(euclid)) + np.asarray(jac)
    ranked = np.argsort(sim)
    ind = ranked[::-1]  # descending order indices
    for k in range(10):  # 10 retrived
        ret_im = ind[0: k]
        rel_im = []
        f1 = []
        pr = []
        re = []
        for m in range(len(ret_im)):
            rel_im1 = np.where([ret_im[m]] == Q_matrix)  # relevant
            rel_im.append(rel_im1)

            ret_im_db = ind  # Retrived Image
            rel_im_db = np.where(ret_im_db == Q_matrix)

            prec = (len(rel_im[0]) / len(ret_im)) * 100  # Precision
            recall = (len(rel_im_db[0]) / len(ret_im_db)) * 100  # Recall
            f1score = (2 * prec * recall) / (prec + recall)

            pr.append(prec)
            re.append(recall)
            f1.append(f1score)

        Precision.append(np.mean(np.asarray(pr)))
        Recall.append(np.mean(np.asarray(re)))
        F1Score.append(np.mean(np.asarray(f1)))

    Precision = np.asarray(Precision)
    Recall = np.asarray(Recall)
    F1Score = np.asarray(F1Score)
    return Precision, Recall, F1Score
