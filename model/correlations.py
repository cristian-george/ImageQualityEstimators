import pandas as pd
import scipy


def plcc(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def srcc(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return plcc(xranks, yranks)
