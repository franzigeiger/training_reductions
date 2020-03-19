import logging
import sqlite3
import statistics

import numpy as np

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
    sql = "select model, benchmark, time, score, modification, batchnorm from raw_scores where modification in ({seq}) and benchmark in ({seq2})".format(
        seq=','.join(['?'] * len(models)), seq2=','.join(['?']* len(benchmarks)))
    return load_from_statement(conn, sql, models, benchmarks)

def load_from_statement(conn, sql, models, benchmarks):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    results = {}
    cursor = conn.cursor()
    cursor.execute(sql, models+ benchmarks)
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
                logger.error(f'No score for model{k} and benchmark {benchmarks[i]}')
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
