# ===============================================================================
# Copyright 2021 Intel Corporation
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

import bench
import numpy as np


def main():
    from sklearn.svm import SVR
    from sklearn.cluster import KMeans
    import numpy as np

    X_train, X_test, y_train, y_test = bench.load_data(params)
    y_train = np.asfortranarray(y_train).ravel()

    X_train_t = X_train.iloc[:2000]
    y_train_t = y_train[:2000]

    if params.gamma is None:
        params.gamma = 1.0 / X_train.shape[1]

    cache_size_bytes = bench.get_optimal_cache_size(X_train.shape[0],
                                                    max_cache=params.max_cache_size)
    params.cache_size_mb = cache_size_bytes / 1024**2
    params.n_classes = len(np.unique(y_train))

    regr = SVR(C=params.C, epsilon=params.epsilon, kernel=params.kernel,
               cache_size=params.cache_size_mb, tol=params.tol, gamma=params.gamma,
               degree=params.degree)
    kmeans = KMeans(n_clusters=16)
    kmeans.fit(X_train_t)

    fit_time, _ = bench.measure_function_time(regr.fit, X_train_t, y_train_t, params=params)
    params.sv_len = regr.support_.shape[0]
    print('train')

    # predict_train_time, y_pred = bench.measure_function_time(
    #     regr.predict, X_train, params=params)
    # train_rmse = bench.rmse_score(y_train, y_pred)
    # train_r2 = bench.r2_score(y_train, y_pred)

    # _, y_pred = bench.measure_function_time(
    #     regr.predict, X_test, params=params)
    # test_rmse = bench.rmse_score(y_test, y_pred)
    # test_r2 = bench.r2_score(y_test, y_pred)

    full_data = np.concatenate([X_train, X_test], axis=0)
    pred_time_set_x1 = np.zeros([100,])
    pred_time_set_x10 = np.zeros([100,])
    pred_time_set_x100 = np.zeros([100,])
    print(X_test.shape)
    print('start pred')
    for i in range(100):
        print(i)
        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x1, _ = bench.measure_function_time(
            regr.predict, full_data[i].reshape(1,-1), params=params)

        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x10, _ = bench.measure_function_time(
            regr.predict, full_data[10 * i:10 * (i + 1)], params=params)
        
        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x100, _ = bench.measure_function_time(
            regr.predict, full_data[100 * i:100 * (i + 1)], params=params)

        pred_time_set_x1[i] = predict_time_x1
        pred_time_set_x10[i] = predict_time_x10
        pred_time_set_x100[i] = predict_time_x100

    inf_time_x1 = np.mean(pred_time_set_x1)
    inf_time_x10 = np.mean(pred_time_set_x10)
    inf_time_x100 = np.mean(pred_time_set_x100)

    bench.print_output(
        library='sklearn',
        algorithm='SVR',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['SVR.predict', 'SVR.predict', 'SVR.predict'],
        times=[inf_time_x1, inf_time_x10, inf_time_x100],
        metric_type=['rmse', 'r2_score', 'n_sv'],
        metrics=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ],
        data=[full_data, full_data, full_data],
        alg_instance=regr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn SVR benchmark')

    parser.add_argument('-C', dest='C', type=float, default=1.,
                        help='SVR regularization parameter')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=.1,
                        help='Epsilon in the epsilon-SVR model')
    parser.add_argument('--kernel', choices=('linear', 'rbf', 'poly', 'sigmoid'),
                        default='linear', help='SVR kernel function')
    parser.add_argument('--degree', type=int, default=3,
                        help='Degree of the polynomial kernel function')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Parameter for kernel="rbf"')
    parser.add_argument('--max-cache-size', type=int, default=8,
                        help='Maximum cache size, in gigabytes, for SVR.')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Tolerance passed to sklearn.svm.SVR')

    params = bench.parse_args(parser, loop_types=('fit', 'predict'))
    bench.run_with_context(params, main)
