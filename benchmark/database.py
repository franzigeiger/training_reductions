import logging
import os
import sqlite3
import statistics

import numpy as np

from base_models.global_data import convergence_epoch, convergence_images

logger = logging.getLogger(__name__)


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn


def get_connection(name='scores'):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    path = f'{dir_path}/../{name}.sqlite'
    return create_connection(path)


def load_model_parameter(conn):
    sql = '''select * from model_parameter'''
    cursor = conn.cursor()
    cursor.execute(sql)
    records = cursor.fetchall()
    results = {}
    for row in records:
        if row[0] in results:
            results[row[0]].append(row[1])
        else:
            results[row[0]] = []
            results[row[0]].append(row[1])

    squeezed = {}
    for key, value in results.items():
        squeezed[key] = np.mean(value)
    return squeezed


model_versions = {
    'CORnet-S_cluster2_v2_IT_trconv3_bi': ['CORnet-S_cluster2_v2_IT_trconv3_bi',
                                           'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42'
        , 'CORnet-S_cluster2_v2_IT_trconv3_bi_seed94'],
    'CORnet-S_cluster2_v2_V4_trconv3_bi': ['CORnet-S_cluster2_v2_V4_trconv3_bi',
                                           'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42'],
    'CORnet-S_train_gmk1_cl2_7_7tr_bi': ['CORnet-S_train_gmk1_cl2_7_7tr_bi', 'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42'],
    'CORnet-S_train_conv3_bi': ['CORnet-S_train_conv3_bi', 'CORnet-S_train_conv3_bi_seed42'],
    'CORnet-S_train_conv3_V2_bi': ['CORnet-S_train_conv3_V2_bi', 'CORnet-S_train_conv3_V2_bi_seed42'],
    'CORnet-S_train_conv3_V4_bi': ['CORnet-S_train_conv3_V4_bi', 'CORnet-S_train_conv3_V4_bi_seed42'],
    'CORnet-S_cluster9_V4_trconv3_bi': ['CORnet-S_cluster9_V4_trconv3_bi', 'CORnet-S_cluster9_V4_trconv3_bi_seed42'],
    'CORnet-S_cluster9_IT_trconv3_bi': ['CORnet-S_cluster9_IT_trconv3_bi', 'CORnet-S_cluster9_IT_trconv3_bi_seed42'],
    'CORnet-S_train_wmk1_cl2_7_7tr_bi': ['CORnet-S_train_wmk1_cl2_7_7tr_bi', 'CORnet-S_train_wmk1_cl2_7_7tr_bi_seed42'],
    'CORnet-S_full': ['CORnet-S_full', 'CORnet-S_full_seed42'],
    'CORnet-S_train_V4': ['CORnet-S_train_V4', 'CORnet-S_train_V4_seed42'],
    'CORnet-S_train_IT_seed_0': ['CORnet-S_train_IT_seed_0', 'CORnet-S_train_IT_seed_0_seed42'],
    'CORnet-S_train_random': ['CORnet-S_train_random', 'CORnet-S_train_random_seed42'],
    # 'mobilenet_v1_1.0_224'  : ['mobilenet_v1_1.0_224']
}


def load_error_bared(conn, models, benchmarks, convergence=True, epochs=[]):
    names = []
    for model in models:
        if model in model_versions.keys():
            for runs in model_versions[model]:
                if convergence and runs in convergence_epoch:
                    postfix = f'_epoch_{convergence_epoch[runs]:02d}'
                    names.append(f'{runs}{postfix}')
                if convergence and runs in convergence_images:
                    postfix = f'_epoch_{convergence_images[runs]:02d}'
                    names.append(f'{runs}{postfix}')
                for e in epochs:
                    if e % 1 == 0:
                        postfix = f'_epoch_{e:02d}'
                    else:
                        postfix = f'_epoch_{e:.1f}'
                    names.append(f'{runs}{postfix}')
        else:
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
                names.append(f'{model}{postfix}')
            if convergence and model in convergence_images:
                postfix = f'_epoch_{convergence_images[model]:02d}'
                names.append(f'{model}{postfix}')
            for e in epochs:
                postfix = f'_epoch_{e:02d}'
                names.append(f'{model}{postfix}')
    model_dict = load_scores(conn, names, benchmarks)
    results = {}
    for model in models:
        if model in model_versions:
            if convergence:
                res = np.zeros([len(benchmarks), 0])
                for runs in model_versions[model]:
                    if runs in convergence_epoch:
                        postfix = f'_epoch_{convergence_epoch[runs]:02d}'
                        name = f'{runs}{postfix}'
                        if name in model_dict:
                            val = model_dict[name].reshape(-1, 1)
                            if np.mean(val > 0):
                                res = np.concatenate([res, val], axis=1)
                results[model] = np.concatenate([np.mean(res, axis=1), np.std(res, axis=1)], axis=0)
            for e in epochs:
                if e % 1 == 0:
                    postfix = f'_epoch_{e:02d}'
                else:
                    postfix = f'_epoch_{e:.1f}'
                res = np.zeros([len(benchmarks), 0])
                for runs in model_versions[model]:
                    name = f'{runs}{postfix}'
                    if name in model_dict:
                        val = model_dict[name].reshape(-1, 1)
                        if np.mean(val > 0):
                            res = np.concatenate([res, val], axis=1)
                results[f'{model}{postfix}'] = np.concatenate([np.mean(res, axis=1), np.std(res, axis=1)], axis=0)
        else:
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
                results[model] = np.concatenate([model_dict[f'{model}{postfix}'], np.zeros(6)], axis=0)
            if convergence and model in convergence_images:
                postfix = f'_epoch_{convergence_images[model]:02d}'
                results[model] = np.concatenate([model_dict[f'{model}{postfix}'], np.zeros(6)], axis=0)
            for e in epochs:
                if e % 1 == 0:
                    postfix = f'_epoch_{e:02d}'
                else:
                    postfix = f'_epoch_{e:.1f}'
                results[f'{model}{postfix}'] = np.concatenate([model_dict[f'{model}{postfix}'], np.zeros(6)], axis=0)
    return results


def store_score(conn, score):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO raw_scores(model,benchmark,time, score, modification, batchnorm)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, score)
    conn.commit()
    return cur.lastrowid


def load_like(conn, model_name, benchmarks):
    sql = "select model, benchmark, time, score, modification, batchnorm from raw_scores where modification like ?".format(
        model_name)
    # load_from_statement(conn, sql, models, benchmarks)


def load_scores(conn, models, benchmarks):
    private = []
    public = []
    for bench in benchmarks:
        if 'public' in bench:
            public.append(bench)
        else:
            private.append(bench)
    sql_private = "select model, benchmark, time, score, modification, batchnorm from raw_scores where modification in ({seq}) and benchmark in ({seq2})".format(
        seq=','.join(['?'] * len(models)), seq2=','.join(['?'] * len(private)))
    sql_public = "select model, benchmark, time, score, modification, batchnorm from raw_scores where modification in ({seq}) and benchmark in ({seq2})".format(
        seq=','.join(['?'] * len(models)), seq2=','.join(['?'] * len(public)))
    private_res = load_from_statement(conn, sql_private, models, private)
    if len(public) > 0:
        conn_pub = get_connection(name='scores_public')
        public_res = load_from_statement(conn_pub, sql_public, models, public)
        for model, list in public_res.items():
            public_res[model] = np.concatenate((list, private_res[model]), axis=0)
        return public_res
    return private_res


def load_from_statement(conn, sql, models, benchmarks):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    results = {}
    cursor = conn.cursor()
    cursor.execute(sql, models + benchmarks)
    records = cursor.fetchall()
    # Structure:
    # score = { 'Model_name': [[1.3,20.1],[13.3, 1.2]]}
    # score list ordered by benchmark inputs
    scores = {}
    for model in models:
        array = np.empty((len(benchmarks),), dtype=object)
        scores[model] = array
        for i, v in enumerate(array):
            array[i] = []
    for row in records:
        model = row[4]
        time = row[2]
        scores[model][benchmarks.index(row[1])].append(row[3])

    squeezed_scores = {}
    for k, score in scores.items():
        array = np.empty(len(benchmarks), dtype=float)
        for i in range(len(benchmarks)):
            try:
                array[i] = statistics.mean(score[i])
            except:
                logger.error(f'No score for model {k} and benchmark {benchmarks[i]}')
                array[i]=0.0
        squeezed_scores[k] =  array

    return squeezed_scores


def store_analysis(conn, model):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO model_parameter(model_id, train_time, weights_fixed, weights_to_train, additional_params, flops)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, model)
    conn.commit()
    return cur.lastrowid


def clean_database():
    sql = '''delete from raw_scores where MIN(ROWID) not in 
    (select model, modification, score, MIN(ROWID) as row 
    from raw_scores 
    group by modification, benchmark, score)'''
