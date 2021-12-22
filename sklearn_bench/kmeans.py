# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import argparse
from typing import Any

import bench
import numpy as np


def main():
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.metrics.cluster import davies_bouldin_score

    # Load and convert generated data
    X_train, X_test, _, y_test = bench.load_data(params)

    X_init: Any
    if params.filei == 'k-means++':
        X_init = 'k-means++'
    # Load initial centroids from specified path
    elif params.filei is not None:
        X_init = {k: v.astype(params.dtype) for k, v in np.load(params.filei).items()}
        if isinstance(X_init, np.ndarray):
            params.n_clusters = X_init.shape[0]
    # or choose random centroids from training data
    else:
        np.random.seed(params.seed)
        centroids_idx = np.random.randint(low=0, high=X_train.shape[0],
                                          size=params.n_clusters)
        if hasattr(X_train, "iloc"):
            X_init = X_train.iloc[centroids_idx].values
        else:
            X_init = X_train[centroids_idx]

    def fit_kmeans(X, X_init):
        alg = KMeans(n_clusters=params.n_clusters, tol=params.tol,
                     max_iter=params.maxiter, init=X_init, n_init=params.n_init,
                     algorithm=params.algorithm, random_state=params.random_state)
        alg.fit(X)
        return alg

    # Time fit
    fit_time, kmeans = bench.measure_function_time(fit_kmeans, X_train,
                                                   X_init, params=params)
    ridge = Ridge()
    ridge.fit(X_test, y_test)
    # train_predict = kmeans.predict(X_train)
    # acc_train = davies_bouldin_score(X_train, train_predict)

    # Time predict
    # predict_time, test_predict = bench.measure_function_time(
    #     kmeans.predict, X_test, params=params)

    # acc_test = davies_bouldin_score(X_test, test_predict)

    full_data = np.concatenate([X_train, X_test], axis=0)
    
    pred_time_set_x1 = np.zeros([100,])
    pred_time_set_x10 = np.zeros([100,])
    pred_time_set_x100 = np.zeros([100,])
    print(X_test.shape)
    for i in range(100):
        ridge.predict(X_train)
        ridge.predict(X_train)
        predict_time_x1, _ = bench.measure_function_time(
            kmeans.predict, full_data[i].reshape(1,-1), params=params)

        ridge.predict(X_train)
        ridge.predict(X_train)
        predict_time_x10, _ = bench.measure_function_time(
            kmeans.predict, full_data[10 * i:10 * (i + 1)], params=params)
        
        ridge.predict(X_train)
        ridge.predict(X_train)
        predict_time_x100, _ = bench.measure_function_time(
            kmeans.predict, full_data[100 * i:100 * (i + 1)], params=params)

        pred_time_set_x1[i] = predict_time_x1
        pred_time_set_x10[i] = predict_time_x10
        pred_time_set_x100[i] = predict_time_x100

    inf_time_x1 = np.mean(pred_time_set_x1)
    inf_time_x10 = np.mean(pred_time_set_x10)
    inf_time_x100 = np.mean(pred_time_set_x100)

    bench.print_output(
        library='sklearn',
        algorithm='kmeans',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['kmeans.predict', 'kmeans.predict', 'kmeans.predict'],
        times=[inf_time_x1, inf_time_x10, inf_time_x100],
        metric_type='none',
        metrics=[None, None, None],
        data=[full_data, full_data, full_data],
        alg_instance=kmeans,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn K-means benchmark')
    parser.add_argument('-i', '--filei', '--fileI', '--init',
                        type=str, help='Initial clusters')
    parser.add_argument('-t', '--tol', type=float, default=0.,
                        help='Absolute threshold')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters')
    parser.add_argument('--algorithm', type=str, default='full',
                        help='K-means algorithm to use')
    parser.add_argument('--n_init', type=int, default=1,
                        help='Number of time the k-means algorithm '
                        'will be run with different centroid seeds')
    parser.add_argument('--random_state', type=int, default=777,
                        help='Random state')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
