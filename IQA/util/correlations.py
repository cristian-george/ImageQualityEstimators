import scipy


def plcc(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def srcc(x, y):
    return scipy.stats.spearmanr(x, y)[0]
