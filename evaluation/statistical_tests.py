# evaluation/statistical_tests.py
from scipy import stats


def perform_statistical_test(scores1, scores2):
    t_statistic, p_value = stats.ttest_rel(scores1, scores2)
    return t_statistic, p_value
