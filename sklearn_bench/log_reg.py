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
import numpy as np
import pandas as pd


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans

    # Load generated data
    X_train, X_test, y_train, y_test = bench.load_data(params)

    params.n_classes = len(np.unique(y_train))

    if params.multiclass == 'auto':
        params.multiclass = 'ovr' if params.n_classes == 2 else 'multinomial'

    if not params.tol:
        params.tol = 1e-3 if params.solver == 'newton-cg' else 1e-10

    # Create our classifier object
    clf = LogisticRegression(penalty='l2', C=params.C, n_jobs=params.n_jobs,
                             fit_intercept=params.fit_intercept,
                             verbose=params.verbose,
                             tol=params.tol, max_iter=params.maxiter,
                             solver=params.solver, multi_class=params.multiclass)
    kmeans = KMeans(n_clusters=16)
    kmeans.fit(X_test)
    print('done0')
    # Time fit and predict
    fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
    print('done1')
    # y_pred = clf.predict(X_train)
    # y_proba = clf.predict_proba(X_train)
    # train_acc = bench.accuracy_score(y_train, y_pred)
    # train_log_loss = bench.log_loss(y_train, y_proba)
    # train_roc_auc = bench.roc_auc_score(y_train, y_proba)

    # predict_time, y_pred = bench.measure_function_time(
    #     clf.predict, X_test, params=params)
    # y_proba = clf.predict_proba(X_test)
    # test_acc = bench.accuracy_score(y_test, y_pred)
    # test_log_loss = bench.log_loss(y_test, y_proba)
    # test_roc_auc = bench.roc_auc_score(y_test, y_proba)

    full_data1 = pd.concat([X_train, X_test])
    full_data2 = pd.concat([X_train, X_test])
    full_data3 = pd.concat([full_data1, full_data2])
    full_data4 = pd.concat([full_data1, full_data2])
    print('done1')
    full_data = pd.concat([full_data3, full_data4]).to_numpy()
    print('done1')
    pred_time_set_x1 = np.zeros([100,])
    pred_time_set_x10 = np.zeros([100,])
    pred_time_set_x100 = np.zeros([100,])
    print(X_test.shape)
    j = 0
    print('done2')
    for i in range(100):
        if ((i % 2) == 1):
            j = 0
        # print(full_data[100000*j:100000*(j+1)].shape)
        # kmeans.predict(X_test)
        # kmeans.predict(X_test)
        # predict_time_x1, _ = bench.measure_function_time(
        #     clf.predict, full_data[1000*j:1000*(j+1)], params=params)

        # kmeans.predict(X_test)
        # kmeans.predict(X_test)
        # predict_time_x10, _ = bench.measure_function_time(
        #     clf.predict, full_data[10000*j:10000*(j+1)], params=params)
        
        kmeans.predict(X_test)
        kmeans.predict(X_test)
        predict_time_x100, _ = bench.measure_function_time(
            clf.predict, full_data[1000000*j:1000000*(j+1)], params=params)

        # pred_time_set_x1[i] = predict_time_x1
        # pred_time_set_x10[i] = predict_time_x10
        pred_time_set_x100[i] = predict_time_x100
        j += 1

    inf_time_x1 = np.mean(pred_time_set_x1)
    inf_time_x10 = np.mean(pred_time_set_x10)
    inf_time_x100 = np.mean(pred_time_set_x100)
    print('done4')
    bench.print_output(
        library='sklearn',
        algorithm='log_reg',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['LogReg.predict', 'LogReg.predict', 'LogReg.predict'],
        times=[inf_time_x1, inf_time_x10, inf_time_x100],
        metric_type=['accuracy', 'log_loss', 'roc_auc'],
        metrics=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ],
        data=[full_data, full_data, full_data],
        alg_instance=clf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn logistic '
                                                 'regression benchmark')
    parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                        action='store_false', default=True,
                        help="Don't fit intercept")
    parser.add_argument('--multiclass', default='auto',
                        choices=('auto', 'ovr', 'multinomial'),
                        help='How to treat multi class data. '
                             '"auto" picks "ovr" for binary classification, and '
                             '"multinomial" otherwise.')
    parser.add_argument('--solver', default='lbfgs',
                        choices=('lbfgs', 'newton-cg', 'saga'),
                        help='Solver to use.')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Maximum iterations for the iterative solver')
    parser.add_argument('-C', dest='C', type=float, default=1.0,
                        help='Regularization parameter')
    parser.add_argument('--tol', type=float, default=None,
                        help='Tolerance for solver. If solver == "newton-cg", '
                             'then the default is 1e-3. Otherwise, the default '
                             'is 1e-10.')
    params = bench.parse_args(parser, loop_types=('fit', 'predict'))
    bench.run_with_context(params, main)
