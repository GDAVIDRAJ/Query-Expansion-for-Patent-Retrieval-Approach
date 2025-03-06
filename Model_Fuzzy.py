import numpy as np
import skfuzzy as fuzz
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



def Model_Fuzzy(Feat, Target):
    # Define the number of clusters
    n_clusters = len(np.unique(Target))

    # Apply fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        Feat, n_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    Quer = cntr[:, 0]
    DB = Feat
    Precision = []
    Recall = []
    F1Score = []

    cosine = []
    euclid = []
    jac = []
    for j in range(len(DB)):
        cosine.append(cosine_similarity(Quer, DB[j]))
        euclid.append(euclidean(Quer, DB[j]))
        jac.append(jaccard(Quer.tolist(), DB[j].tolist()))
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
            rel_im1 = np.where([ret_im[m]] == Quer)  # relevant
            rel_im.append(rel_im1)

            ret_im_db = ind  # Retrived Image
            rel_im_db = np.where(ret_im_db == Quer)

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

