import os.path as path
import csv
import pandas as pd


def load_csv(file_name):
    """Read list of rows in csv, separated by comma"""
    res = pd.DataFrame.from_csv(file_name, header=None)
    # results = []
    # if path.isfile(file_name):
    #     with open(file_name, 'rt') as f:
    #         for row in csv.reader(f, delimiter=','):
    #             results.append(row)
    return res


def read_dcase(fname):
    raw_scores = pd.DataFrame.from_csv(fname, header=None)
    score_df = raw_scores.pivot(columns=1, values=2)
    assert score_df.isnull().values.any() == False
    # scores info
    print(score_df.describe().T)
    return score_df


# def read_dcase_scores(fname):
#     pass
#
#
# def dcase2pandas():
#     pass
#
#
# def pandas2dcase():
#     pass