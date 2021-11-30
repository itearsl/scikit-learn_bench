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
from timeit import default_timer as timer
import bench
import numpy as np



def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans

    # Load and convert data
    X_train, X_test, y_train, y_test = bench.load_data(params)

    # Create our random forest classifier
    clf = RandomForestClassifier(criterion=params.criterion,
                                 n_estimators=params.num_trees,
                                 max_depth=params.max_depth,
                                 max_features=params.max_features,
                                 min_samples_split=params.min_samples_split,
                                 max_leaf_nodes=params.max_leaf_nodes,
                                 min_impurity_decrease=params.min_impurity_decrease,
                                 bootstrap=params.bootstrap,
                                 random_state=params.seed,
                                 n_jobs=params.n_jobs)

    model = KMeans(n_clusters=16)
    params.n_classes = len(np.unique(y_train))
    # X_train['2000'] = y_train
    # X_test['2000'] = y_test
    # X_train.to_csv('train_air.csv', header=False, index=False)
    # X_test.to_csv('test_air.csv', header=False, index=False)
    # X_test.iloc[2:4].to_csv('test2.csv', header=False, index=False)
    # print('done')
    fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
    model.fit(X_train)
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
    X_full1 = np.concatenate([X_train, X_test], axis=0)
    X_full2 = np.concatenate([X_train, X_test], axis=0)
    # X_full3 = np.concatenate([X_train, X_test], axis=0)
    X_full = np.concatenate([X_full1, X_full2], axis=0)
    pred_time_set_x1_1 = np.zeros([100,])
    pred_time_set_x1_100 = np.zeros([100,])
    # pred_time_set_x1_test = np.zeros([100,])
    pred_time_set_x1_10000 = np.zeros([100,])
    # pred_time_set_x10 = np.zeros([100,])
    # pred_time_set_x100 = np.zeros([100,])
    for i in range(100):
        model.predict(X_test)
        model.predict(X_test)
        pred_time_x1_1, _ = bench.measure_function_time(
            clf.predict, X_full[i].reshape(1, -1), params=params)
        model.predict(X_test)
        model.predict(X_test)
        pred_time_x1_100, _ = bench.measure_function_time(
            clf.predict, X_full[i*100].reshape(1, -1), params=params)
        model.predict(X_test)
        model.predict(X_test)
        pred_time_x1_10000, _ = bench.measure_function_time(
            clf.predict, X_full[i*10000].reshape(1, -1), params=params)
        # pred_time_x10, _ = bench.measure_function_time(
        #     clf.predict, X_train.iloc[10*i:10*(i+1), :], params=params)
        # pred_time_x100, _ = bench.measure_function_time(
        #     clf.predict, X_train.iloc[100*i:100*(i+1), :], params=params)
        pred_time_set_x1_1[i] = pred_time_x1_1
        pred_time_set_x1_100[i] = pred_time_x1_100
        pred_time_set_x1_10000[i] = pred_time_x1_10000
        # pred_time_set_x10[i] = pred_time_x10
        # pred_time_set_x100[i] = pred_time_x100
    print('------------------------------------------------------------------')
    print(pred_time_set_x1_1[:])
    print('------------------------------------------------------------------')

    inf_time_x1_1 = np.mean(pred_time_set_x1_1)
    inf_time_x1_100 = np.mean(pred_time_set_x1_100)
    inf_time_x1_10000 = np.mean(pred_time_set_x1_10000)
    # inf_time_x10 = np.mean(pred_time_set_x10)
    # inf_time_x100 = np.mean(pred_time_set_x100)

    bench.print_output(
        library='sklearn',
        algorithm='df_clsf',
        stages=['inferenceX1', 'inferenceX10', 'inferenceX100'],
        params=params,
        functions=['df_clsf.predict', 'df_clsf.predict', 'df_clsf.predict'],
        times=[inf_time_x1_1, inf_time_x1_100, inf_time_x1_10000],
        metric_type=['accuracy', 'log_loss', 'roc_auc'],
        metrics=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ],
        data=[X_train, X_train, X_train],
        alg_instance=clf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                                 'classification benchmark')

    parser.add_argument('--criterion', type=str, default='gini',
                        choices=('gini', 'entropy'),
                        help='The function to measure the quality of a split')
    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features', type=bench.float_or_int, default=None,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Upper bound on depth of constructed trees')
    parser.add_argument('--min-samples-split', type=bench.float_or_int, default=2,
                        help='Minimum samples number for node splitting')
    parser.add_argument('--max-leaf-nodes', type=int, default=None,
                        help='Maximum leaf nodes per tree')
    parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                        help='Needed impurity decrease for node splitting')
    parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                        action='store_false', help="Don't control bootstraping")
    parser.add_argument('--device', default='None', type=str,
                        choices=('host', 'cpu', 'gpu', 'None'),
                        help='Execution context device')

    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
