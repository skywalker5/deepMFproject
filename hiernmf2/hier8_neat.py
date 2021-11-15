__author__ = 'hannahkim'

import sys
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import svds
import time
from scipy.sparse import csr_matrix

from scipy import linalg
gemv = linalg.get_blas_funcs("gemv")

def hier8_neat(X, k, tol=1e-4, maxiter=10000, trial_allowance=3, unbalanced=0.1):
    t0 = time.time()
    # m, n = np.shape(X)
    m, n = X.shape

    # initialize storing variables
    timings = np.zeros(k-1)
    clusters = np.empty(2 * (k - 1), dtype=object)
    Ws = np.zeros((m, 2 * (k - 1)))
    W_buffer = np.empty(2 * (k - 1), dtype=object)
    H_buffer = np.empty(2 * (k - 1), dtype=object)
    priorities = np.zeros(2 * (k - 1))
    is_leaf = -1 * np.ones(2 * (k - 1), dtype=np.int64)
    tree = np.zeros((2, 2 * (k - 1)), dtype=np.int64)
    splits = np.zeros(k - 1, dtype=np.int64)

    term_subset = np.where(np.sum(X, axis=1) != 0)[0]
    W = np.random.rand(len(term_subset), 2)
    H = np.random.rand(2, n)
    if len(term_subset) == m:
        W, H = nmfsh_comb_rank2(X, W, H, tol, maxiter)
    else:
        W_tmp, H = nmfsh_comb_rank2(X[term_subset, :], W, H, tol, maxiter)
        W = np.zeros((m, 2))
        W[term_subset, :] = W_tmp

    result_used = 0
    for i in range(0, k - 1):
        timings[i] = time.time() - t0

        if i == 0:
            split_node = 0
            new_nodes = [0, 1]
            min_priority = 1e308
            split_subset = np.arange(0, n)
        else:
            leaves = np.where(is_leaf == 1)[0]
            temp_priority = priorities[leaves]
            print(priorities)
            print(temp_priority)
            min_priority = np.min(temp_priority[temp_priority > 0])
            if np.max(temp_priority) < 0:
                print('Cannot generate all ', k, ' leaf clusters')
                return
            split_node = np.argmax(temp_priority)
            split_node = leaves[split_node]
            is_leaf[split_node] = 0
            W = W_buffer[split_node]
            H = H_buffer[split_node]
            split_subset = clusters[split_node]
            new_nodes = [result_used, result_used + 1]
            tree[0, split_node] = new_nodes[0]
            tree[1, split_node] = new_nodes[1]

        result_used = result_used + 2
        cluster_subset = np.argmax(H, axis=0)
        clusters[new_nodes[0]] = split_subset[np.where(cluster_subset == 0)[0]]
        clusters[new_nodes[1]] = split_subset[np.where(cluster_subset == 1)[0]]
        Ws[:, new_nodes[0]] = W[:, 0]
        Ws[:, new_nodes[1]] = W[:, 1]
        splits[i] = split_node
        is_leaf[new_nodes] = 1

        print('priorities', priorities[:2 * (i + 1)])
        print('splits', splits[:i + 1])
        print('tree', tree[:, :2 * (i + 1)])
        print('is_leaf', is_leaf[:2 * (i + 1)])

        subset = clusters[new_nodes[0]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W[:, 0], tol, maxiter)
        clusters[new_nodes[0]] = subset
        W_buffer[new_nodes[0]] = W_buffer_one
        H_buffer[new_nodes[0]] = H_buffer_one
        priorities[new_nodes[0]] = priority_one

        subset = clusters[new_nodes[1]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W[:, 1], tol, maxiter)
        clusters[new_nodes[1]] = subset
        W_buffer[new_nodes[1]] = W_buffer_one
        H_buffer[new_nodes[1]] = H_buffer_one
        priorities[new_nodes[1]] = priority_one

    return tree, splits, is_leaf, clusters, timings, Ws, priorities

def trial_split(trial_allowance, unbalanced, min_priority, X, subset, W_parent, tol, maxiter):
    m, n = np.shape(X)
    trial = 0
    subset_backup = subset
    while trial < trial_allowance:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = actual_split(X, subset, W_parent, tol, maxiter)
        if priority_one < 0:
            break
        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            sys.exit('Error: Invalid number of unique sub-clusters!')
        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        min_length = np.min([length_cluster1, length_cluster2])
        if min_length < unbalanced * len(cluster_subset):
            idx_small = np.argmin([length_cluster1, length_cluster2])
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[0]
            subset_small = subset[subset_small]
            cluster_subset_small, W_buffer_one_small, H_buffer_one_small, priority_one_small = actual_split(X, subset_small, W_buffer_one[:, idx_small], tol, maxiter)
            if priority_one_small < min_priority:
                trial = trial + 1
                if trial < trial_allowance:
                    print('Drop ', len(subset_small), ' documents ...')
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == trial_allowance:
        print('Recycle ', len(subset_backup)-len(subset), ' documents ...')
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2))
        H_buffer_one = np.zeros((2, len(subset)))
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one

#@profile
def actual_split(X, subset, W_parent, tol, maxiter):
    m, n = np.shape(X)
    subset_length = len(subset)  # to match dimension
    if subset_length <= 3:
        cluster_subset = np.ones(subset_length)
        W_buffer_one = np.zeros((m, 2))
        H_buffer_one = np.zeros((2, subset_length))
        priority_one = -1
    else:
        X_subset = X[:,subset]
        term_subset = np.where(np.sum(X_subset.reshape(m, subset_length), axis=1) != 0)[0]
        X_subset = X_subset[term_subset,:]
#        print(np.shape(X_subset))
        W = np.random.rand(len(term_subset), 2)
        H = np.random.rand(2, subset_length)
        W, H = nmfsh_comb_rank2(X_subset, W, H, tol, maxiter)
        cluster_subset = np.argmax(H, axis=0)
        W_buffer_one = np.zeros((m, 2))
        W_buffer_one[term_subset, :] = W
        H_buffer_one = H
        if len(np.unique(cluster_subset)) > 1:
            priority_one = compute_priority(W_parent, W_buffer_one)
        else:
            priority_one = -1

    return cluster_subset, W_buffer_one, H_buffer_one, priority_one

#@profile
def compute_priority(W_parent, W_child):
    n = len(W_parent)
    idx_parent = np.argsort(W_parent.T, axis=0)  # to match dimension
    idx_parent = idx_parent[::-1]
    sorted_parent = W_parent[idx_parent]
    idx_child1 = np.argsort(W_child[:, 0], axis=0)
    idx_child1 = idx_child1[::-1]
    idx_child2 = np.argsort(W_child[:, 1], axis=0)
    idx_child2 = idx_child2[::-1]

    n_part = len(np.where(W_parent != 0)[0])
    if n_part <= 1:
        priority = -3
    else:
        weight = np.log(np.arange(n, 0, -1))
        zero = np.where(sorted_parent == 0)[0]
        if len(zero) > 0:
            weight[zero[0]:n] = 1
        weight_part = np.zeros(n)
        weight_part[0:n_part] = np.log(np.arange(n_part, 0, -1))
        idx1 = np.argsort(idx_child1, axis=0)
        idx2 = np.argsort(idx_child2, axis=0)
        max_pos = np.maximum(idx1, idx2)
        discount = np.log(n - max_pos[idx_parent])
        discount[np.where(discount == 0)] = np.log(2)
        weight = weight / discount
        weight_part = weight_part / discount
        priority = NDCG_part(idx_parent, idx_child1, weight, weight_part) * NDCG_part(idx_parent, idx_child2, weight, weight_part)
    return priority

#@profile
def NDCG_part(ground, test, weight, weight_part):
    seq_idx = np.argsort(ground, axis=0)
    weight_part = weight_part[seq_idx]

    n = len(test)
    uncum_score = weight_part[test]
    uncum_score[1:n] = uncum_score[1:n] / np.log2(np.arange(2, n+1))
    cum_score = np.cumsum(uncum_score)

    ideal_score = np.sort(weight, axis=0)
    ideal_score = ideal_score[::-1]
    ideal_score[1:n] = ideal_score[1:n] / np.log2(np.arange(2, n+1))
    cum_ideal_score = np.cumsum(ideal_score)

    score = cum_score / cum_ideal_score
    score = score[-1]
    return score

#@profile
def nmfsh_comb_rank2(matrixA, init_matrixW, init_matrixH, tol=1e-4, maxiter=10000, vec_norm=2, normW=True):
    t0 = time.time()
    # A fast algorithm for rank-2 nonnegative matrix factorization

    # Input parameters
    # A: m*n data matrix
    # Winit: m*2 matrix for initialization of W
    # Hinit: 2*n matrix for initialization of H
    # params (optional)
    # params.vec_norm (default=2): indicates which norm to use for the normalization of W or H,
    #                              e.g. vec_norm=2 means Euclidean norm; vec_norm=0 means no normalization.
    # params.normW (default=true): true if normalizing columns of W; false if normalizing rows of H
    # params.tol (default=1e-4): tolerance parameter for stopping criterion
    # params.maxiter (default=10000): maximum number of iteration times

    # Output parameters
    # W, H: result of rank-2 NMF
    # iter: number of ANLS iterations actually used
    # grad: relative norm of projected gradient, reflecting the stationarity of the solution

    m, n = np.shape(matrixA)
    k = 2
    eps = np.spacing(1)

    matrixW = init_matrixW
    matrixH = init_matrixH
    if np.shape(matrixW)[1] is not k:
        sys.exit('Error: Invalid size of the matrix W, please check.')
    if np.shape(matrixH)[0] is not k:
        sys.exit('Error: Invalid size of the matrix H, please check.')

#    print '11', time.time() - t0; t0 = time.time()
    
    left = np.dot(init_matrixH, init_matrixH.T)
#    matrixHT_sparse = scipy.sparse.csr_matrix(init_matrixH.T)
#    right = matrixA * matrixHT_sparse
#    right = gemv(1,matrixA, init_matrixH.T)
#     right = np.dot(matrixA, init_matrixH.T)
    right = matrixA.dot(init_matrixH.T)
    # right = np.matmul(matrixA, init_matrixH.T)
    # right = right.toarray()
    
#    print '12', time.time() - t0; t0 = time.time()

    for iterNumber in range(0, maxiter):
        if np.linalg.matrix_rank(left) < 2:
            print('The matrix H is singular')
            matrixW = np.zeros((m, 2))
            matrixH = np.zeros((2,n))
            U, S, V = svds(matrixA, 1)
            if U.sum() < 0:
                U = -U
                V = -V
            matrixW[:, 0] = U
            matrixH[0, :] = V
            return

        # print left
        # print right
        matrixW = anls_entry_rank2_precompute(left, right, matrixW)
        norms_W = np.linalg.norm(matrixW, axis=0)
        if norms_W.min() < eps:
            sys.exit('Error: Some column of W is essentially zero, please check.')
        matrixW = matrixW / norms_W

        left = np.dot(matrixW.T, matrixW)
#        matrixW_sparse = scipy.sparse.csr_matrix(matrixW)
#        right = matrixA.T * matrixW_sparse
#         right = np.dot(matrixA.T, matrixW)
        right = matrixA.T.dot(matrixW)
        # right = right.toarray()

        if np.linalg.matrix_rank(left) < 2:
            print('The matrix W is singular')
            matrixW = np.zeros((m, 2))
            matrixH = np.zeros((2, n))
            U, S, V = svds(matrixA, 1)
            if U.sum() < 0:
                U = -U
                V = -V
            matrixW[:, 0] = U
            matrixH[0, :] = V
            return

        matrixH = anls_entry_rank2_precompute(left, right, matrixH.T)
        matrixH = matrixH.T
        gradH = np.dot(left, matrixH) - right.T

        left = np.dot(init_matrixH, init_matrixH.T)
#        matrixHT_sparse = scipy.sparse.csr_matrix(init_matrixH.T)
#        right = matrixA * matrixHT_sparse
#         right = np.dot(matrixA, init_matrixH.T)
        right = matrixA.dot(init_matrixH.T)
        # right = right.toarray()
        gradW = np.dot(matrixW, left) - right

        dim_w = np.shape(matrixW)
        dim_h = np.shape(matrixH)
        dim_gradW = np.shape(gradW)
        dim_gradH = np.shape(gradH)

        idx_tuple1 = np.where(gradW <= 0.0)
        idx_tuple2 = np.where(matrixW > 0.0)
        idx_linear1 = np.ravel_multi_index(idx_tuple1, dim_gradW)
        idx_linear2 = np.ravel_multi_index(idx_tuple2, dim_w)
        idx_all = np.concatenate((idx_linear1, idx_linear2))
        idx_all = np.unique(idx_all)
        idx_final = np.unravel_index(idx_all, dim_gradW)

        values_w = gradW[idx_final[0], idx_final[1]]
        norm_w = np.linalg.norm(values_w) ** 2.0

        idx_tuple1 = np.where(gradH <= 0.0)
        idx_tuple2 = np.where(matrixH > 0.0)
        idx_linear1 = np.ravel_multi_index(idx_tuple1, dim_gradH)
        idx_linear2 = np.ravel_multi_index(idx_tuple2, dim_h)
        idx_all = np.concatenate((idx_linear1, idx_linear2))
        idx_all = np.unique(idx_all)
        idx_final = np.unravel_index(idx_all, dim_gradH)

        values_h = gradH[idx_final[0], idx_final[1]]
        norm_h = np.linalg.norm(values_h) ** 2.0

        if iterNumber is 0:
            init_grad = np.sqrt(norm_w + norm_h)
            continue
        else:
            project_norm = np.sqrt(norm_w + norm_h)

        if project_norm < tol * init_grad:
            break

    grad = project_norm / init_grad
    
#    if vec_norm is not 0:
#        vec_norm = float(vec_norm)
#        if normW:
#            norms = (matrixW**vec_norm).sum(axis=0) ** (1/vec_norm)
#            matrixW = matrixW/norms
#            matrixH = matrixH*norms[:, np.newaxis]
#        else:
#            matrixW = matrixW*norms
#            matrixH = matrixH/norms[:, np.newaxis]
#            norms = (matrixH**vec_norm).sum(axis=1) ** (1/vec_norm)
#     time.time() - t0; print('13', t0 = time.time())
    print('iterNumber', iterNumber)
    return matrixW, matrixH #, iterNumber+1, grad

#@profile
def anls_entry_rank2_precompute(left, right, matrixH):
    # left: 2 * 2
    # right: n * 2
    # Returning H of size n*2 also

    n = np.shape(right)[0]
    eps = np.spacing(1)

    solve_either = np.zeros((n, 2))
    solve_either[:, 0] = right[:, 0] * (1.0 / left[0, 0])
    solve_either[:, 1] = right[:, 1] * (1.0 / left[1, 1])

    values = np.array([[np.sqrt(left[0, 0]), np.sqrt(left[1, 1])]])
    multiply_values = np.repeat(values, n, axis=0)
    cosine_either = np.multiply(solve_either, multiply_values)
    choose_first = cosine_either[:, 0] >= cosine_either[:, 1]
    solve_either[choose_first, 1] = 0.0
    solve_either[choose_first != True, 0] = 0.0

    # H = (left \ right')';
    if np.abs(left[0, 0]) < eps and np.abs(left[0, 1]) < eps:
        message = 'Error: The 2x2 matrix is close to singular or the input data' + \
                  ' matrix has small values, please check.'
        sys.exit(message)
    else:
        if np.abs(left[0, 0]) >= np.abs(left[0, 1]):
            # pivot == 1, extracting cosine
            t_val = left[1, 0] / left[0, 0]
            a2_val = left[0, 0] + t_val * left[1, 0]
            b2_val = left[0, 1] + t_val * left[1, 1]
            d2_val = left[1, 1] - t_val * left[0, 1]
            if np.abs(d2_val / a2_val) < eps:
                # a2_val is guaranteed to be positive
                sys.exit('Error: The 2x2 matrix is close to singular, please check.')
            e2_val = right[:, 0] + t_val * right[:, 1]
            f2_val = right[:, 1] - t_val * right[:, 0]
        else:
            # pivot == 2, extracting sine
            ct_val = left[0, 0] / left[1, 0]
            a2_val = left[1, 0] + ct_val * left[0, 0]
            b2_val = left[1, 1] + ct_val * left[0, 1]
            d2_val = (-1.0 * left[0, 1]) + (ct_val * left[1, 1])
            if np.abs(d2_val / a2_val) < eps:
                # a2_val is guaranteed to be positive
                sys.exit('Error: The 2x2 matrix is close to singular, please check.')
            e2_val = right[:, 1] + ct_val * right[:, 0]
            f2_val = (-1.0 * right[:, 0]) + (ct_val * right[:, 1])

        matrixH[:, 1] = f2_val * (1.0 / d2_val)
        matrixH[:, 0] = (e2_val - b2_val * matrixH[:, 1]) * (1.0 / a2_val)

    positive_vals = np.all(matrixH > 0, axis=1)
    use_either = positive_vals != True
    matrixH[use_either, :] = solve_either[use_either, :]

    return matrixH


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('data/a391d853147b-NASA_DataSets_Scrub.tsv',delimiter='\t',encoding='utf-8')
    dataset.head()
    dataset = dataset.drop(columns=['issued', 'modified'], axis=1)
    dataset.head()

    # Load the regular expression library
    import re
    dataset['description']
    dataset['description'].map(lambda x: re.sub('[,\.!?]', '',str(x)))
    # Remove punctuation
    dataset['paper_text_processed'] = dataset['description'].map(lambda x: re.sub('[,\.!?]', '', str(x)))
    dataset['paper_text_processed'] = dataset['description'].map(lambda x: re.sub('xxxxxx', ' ', str(x)))
    dataset['paper_text_processed'] = dataset['description'].map(lambda x: re.sub('xxxx', ' ', str(x)))
    # Convert the titles to lowercase
    dataset['paper_text_processed'] = dataset['paper_text_processed'].map(lambda x: x.lower())
    # Print out the first rows of papers
    dataset['paper_text_processed'].head()

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(stop_words='english',smooth_idf=True,use_idf=True)
    tfidf_data = tfidf_vectorizer.fit_transform(dataset['paper_text_processed'])

    A = tfidf_data
    A = csr_matrix(A.T)
    dic = tfidf_vectorizer.get_feature_names()
    k = 5
    print(np.shape(A))
    print(np.shape(dic))
    tree, splits, is_leaf, clusters, timings, Ws, priorities = hier8_neat(A, k)
    leafWs = Ws[:, is_leaf == 1]
    topkeyword = np.argsort(leafWs, axis=0)[::-1][:10]
    for i in range(0, k):
        # for j in topkeyword[:, i].tolist():
        #     print i, j, dic[j][0].tostring()
        str = ', '.join(dic[j] for j in topkeyword[:, i].tolist())
        print(str)
    # print tree, splits, is_leaf, clusters, timings, Ws, priorities