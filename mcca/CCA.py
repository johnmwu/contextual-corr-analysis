# =================================================================================
# (C) 2019 by Weiran Wang (weiranwang@ttic.edu) and Qingming Tang (qmtang@ttic.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# =================================================================================

import numpy as np
import tensorflow as tf

eps_eig = 1e-6

def linCCA(H1, H2, dim, rcov1, rcov2):

    N, d1 = H1.shape
    _, d2 = H2.shape

    # Remove mean.
    m1 = np.mean(H1, axis=0, keepdims=True)
    H1 = H1 - np.tile(m1, [N,1])

    m2 = np.mean(H2, axis=0, keepdims=True)
    H2 = H2 - np.tile(m2, [N,1])

    S11 = np.matmul(H1.transpose(), H1) / (N-1) + rcov1 * np.eye(d1)
    S22 = np.matmul(H2.transpose(), H2) / (N-1) + rcov2 * np.eye(d2)
    S12 = np.matmul(H1.transpose(), H2) / (N-1)

    E1, V1 = np.linalg.eig(S11)
    E2, V2 = np.linalg.eig(S22)

    # For numerical stability.
    idx1 = np.where(E1>eps_eig)[0]

    E1 = E1[idx1]
    V1 = V1[:, idx1]

    idx2 = np.where(E2>eps_eig)[0]
    E2 = E2[idx2]
    V2 = V2[:, idx2]

    K11 = np.matmul( np.matmul(V1, np.diag(np.reciprocal(np.sqrt(E1)))), V1.transpose())
    K22 = np.matmul( np.matmul(V2, np.diag(np.reciprocal(np.sqrt(E2)))), V2.transpose())
    T = np.matmul( np.matmul(K11, S12), K22)
    # print(T)
    U, E, V = np.linalg.svd(T, full_matrices=False)
    V = V.transpose()

    A = np.matmul(K11, U[:, 0:dim])
    B = np.matmul(K22, V[:, 0:dim])
    E = E[0:dim]

    return A, B, m1, m2, E


def CanonCorr(H1, H2, N, d1, d2, dim, rcov1, rcov2):

    # Remove mean.
    m1 = tf.reduce_mean(H1, axis=0, keep_dims=True)
    H1 = tf.subtract(H1, m1)

    m2 = tf.reduce_mean(H2, axis=0, keep_dims=True)
    H2 = tf.subtract(H2, m2)

    S11 = tf.matmul(tf.transpose(H1), H1) / (N-1) + rcov1 * tf.eye(d1)
    S22 = tf.matmul(tf.transpose(H2), H2) / (N-1) + rcov2 * tf.eye(d2)
    S12 = tf.matmul(tf.transpose(H1), H2) / (N-1)

    E1, V1 = tf.self_adjoint_eig(S11)
    E2, V2 = tf.self_adjoint_eig(S22)

    # For numerical stability.
    idx1 = tf.where(E1>eps_eig)[:,0]
    E1 = tf.gather(E1, idx1)
    V1 = tf.gather(V1, idx1, axis=1)

    idx2 = tf.where(E2>eps_eig)[:,0]
    E2 = tf.gather(E2, idx2)
    V2 = tf.gather(V2, idx2, axis=1)

    K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))), tf.transpose(V1))
    K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))), tf.transpose(V2))
    T = tf.matmul(tf.matmul(K11, S12), K22)

    # Eigenvalues are sorted in increasing order.
    E3, U = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))
    idx3 = tf.where(E3 > eps_eig)[:, 0]
    # This is the thresholded rank.
    dim_svd = tf.cond(tf.size(idx3) < dim, lambda: tf.size(idx3), lambda: dim)

    return tf.reduce_sum(tf.sqrt(E3[-dim_svd:])), E3, dim_svd


if __name__ == "__main__":

    H1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]], dtype=np.float32)*0.1
    H2 = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9], [0, 1, 2]], dtype=np.float32)*0.1
    dim = 2

    A,B,m1,m2,E = linCCA(H1, H2, dim, 1e-4, 1e-2)
    print(A)
    print(B)
    print(m1)
    print(m2)
    print(E)

    with tf.device("/cpu:0"):
        sess = tf.Session()
        V1 = tf.Variable(initial_value=H1, name="H1_variable")
        V2 = tf.Variable(initial_value=H2, name="H2_variable")
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        canoncorr, _, _ = CanonCorr(V1,V2,H1.shape[0],H1.shape[1],H2.shape[1],dim,1e-4,1e-2)
        print(sess.run(canoncorr))

        optimizer = tf.train.GradientDescentOptimizer(1e-5)
        grads_and_vars = sess.run( optimizer.compute_gradients(canoncorr, tf.trainable_variables()) )
        vars = tf.trainable_variables()

        count = 0
        for g, v in grads_and_vars:
            if g is not None:
                print("****************this is variable %s *************" % vars[count])
                print(v)
                print("****************this is gradient*************")
                print("gradient's shape:", g.shape)
                print(g)
            count += 1


"""
TEST WITH MATLAB:
H1 = [1,2,3;4,5,6;7 8 9;0, -1, -2]*0.1
H2 = [-1,2,3;4,-5,6;7 8 -9;0, 1, 2]*0.1
[A,B,m1,m2,E] = linCCA(H1, H2, 2, 1e-4, 1e-2)
[corr,grad1,grad2]=DCCA_corr(H1,H2,2,1e-4,1e-2)
"""
