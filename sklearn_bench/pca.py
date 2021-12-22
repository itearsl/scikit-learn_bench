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

import bench


def main():
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import numpy as np

    # Load random data
    X_train, X_test, _, _ = bench.load_data(params, generated_data=['X_train'])

    X_test_t = X_test.iloc[:2000]

    if params.n_components is None:
        p, n = X_train.shape
        params.n_components = min((n, (2 + min((n, p))) // 3))

    # Create our PCA object
    pca = PCA(svd_solver=params.svd_solver, whiten=params.whiten,
              n_components=params.n_components)

    # Time fit
    fit_time, _ = bench.measure_function_time(pca.fit, X_train, params=params)

    kmeans = KMeans(n_clusters=16)
    kmeans.fit(X_test_t)
    # Time transform
    # transform_time, _ = bench.measure_function_time(
    #     pca.transform, X_train, params=params)

    full_data = np.concatenate([X_train, X_test], axis=0)
    pred_time_set_x1 = np.zeros([100,])
    pred_time_set_x10 = np.zeros([100,])
    pred_time_set_x100 = np.zeros([100,])
    print(X_test.shape)
    print('start pred')
    for i in range(100):
        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x1, _ = bench.measure_function_time(
            pca.transform, full_data[i].reshape(1,-1), params=params)

        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x10, _ = bench.measure_function_time(
            pca.transform, full_data[10 * i:10 * (i + 1)], params=params)
        
        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x100, _ = bench.measure_function_time(
            pca.transform, full_data[100 * i:100 * (i + 1)], params=params)

        pred_time_set_x1[i] = predict_time_x1
        pred_time_set_x10[i] = predict_time_x10
        pred_time_set_x100[i] = predict_time_x100

    inf_time_x1 = np.mean(pred_time_set_x1)
    inf_time_x10 = np.mean(pred_time_set_x10)
    inf_time_x100 = np.mean(pred_time_set_x100)

    bench.print_output(
        library='sklearn',
        algorithm='PCA',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['PCA.transform', 'PCA.transform', 'PCA.transform'],
        times=[inf_time_x1, inf_time_x10, inf_time_x100],
        metric_type='noise_variance',
        metrics=[None, None, None],
        data=[full_data, full_data, full_data],
        alg_instance=pca,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn PCA benchmark')
    parser.add_argument('--svd-solver', type=str, choices=['full'],
                        default='full', help='SVD solver to use')
    parser.add_argument('--n-components', type=int, default=None,
                        help='The number of components to find')
    parser.add_argument('--whiten', action='store_true', default=False,
                        help='Perform whitening')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
