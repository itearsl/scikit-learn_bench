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
import numpy as np
import bench


def main():
    from sklearn.linear_model import Lasso
    from sklearn.cluster import KMeans

    # Load data
    X_train, X_test, y_train, y_test = bench.load_data(params)


    # Create our regression object
    regr = Lasso(fit_intercept=params.fit_intercept, alpha=params.alpha,
                 tol=params.tol, max_iter=params.maxiter, copy_X=False)
    kmeans = KMeans(n_clusters=16)
    kmeans.fit(X_test)
    # Time fit
    fit_time, _ = bench.measure_function_time(regr.fit, X_train, y_train, params=params)

    # Time predict
    # predict_time, yp = bench.measure_function_time(
    #     regr.predict, X_train, params=params)

    # train_rmse = bench.rmse_score(y_train, yp)
    # train_r2 = bench.r2_score(y_train, yp)
    # yp = regr.predict(X_test)
    # test_rmse = bench.rmse_score(y_test, yp)
    # test_r2 = bench.r2_score(y_test, yp)
    full_data = np.concatenate([X_train, X_test], axis=0)
    
    pred_time_set_x1 = np.zeros([100,])
    pred_time_set_x10 = np.zeros([100,])
    pred_time_set_x100 = np.zeros([100,])
    print(X_test.shape)
    for i in range(100):
        kmeans.predict(X_train)
        kmeans.predict(X_train)
        predict_time_x1, _ = bench.measure_function_time(
            regr.predict, full_data[i].reshape(1,-1), params=params)

        kmeans.predict(X_train)
        kmeans.predict(X_train)
        predict_time_x10, _ = bench.measure_function_time(
            regr.predict, full_data[10 * i:10 * (i + 1)], params=params)
        
        kmeans.predict(X_train)
        kmeans.predict(X_train)
        predict_time_x100, _ = bench.measure_function_time(
            regr.predict, full_data[100 * i:100 * (i + 1)], params=params)

        pred_time_set_x1[i] = predict_time_x1
        pred_time_set_x10[i] = predict_time_x10
        pred_time_set_x100[i] = predict_time_x100

    inf_time_x1 = np.mean(pred_time_set_x1)
    inf_time_x10 = np.mean(pred_time_set_x1)
    inf_time_x100 = np.mean(pred_time_set_x1)

    bench.print_output(
        library='sklearn',
        algorithm='lasso',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['Lasso.predict', 'Lasso.predict', 'Lasso.predict'],
        times=[inf_time_x1, inf_time_x10, inf_time_x100],
        metric_type=['rmse', 'r2_score', 'iter'],
        metrics=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ],
        data=[full_data, full_data, full_data],
        alg_instance=regr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn lasso regression '
                                     'benchmark')
    parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=False,
                        action='store_false',
                        help="Don't fit intercept (assume data already centered)")
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.0,
                        help='Regularization parameter')
    parser.add_argument('--maxiter', type=int, default=1000,
                        help='Maximum iterations for the iterative solver')
    parser.add_argument('--tol', type=float, default=0.0,
                        help='Tolerance for solver.')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
