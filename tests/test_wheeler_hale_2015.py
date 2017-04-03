import pytest
import pandas
from wheeler_hale_2015.wheeler_hale_2015 import *

@pytest.fixture
def logs(scope='module'):
    return load_logs(['tests/testlog1.las', 'tests/testlog2.las'])

def test_load_logs(logs):
    assert(len(logs) == 2)
    for log in logs:
        assert(type(log) == pandas.core.frame.DataFrame)

def test_prepare_logs(logs):
    orig_logs = [log.copy() for log in logs]
    prepare_logs(logs)
    for i, log in enumerate(logs):
        for col in log.columns:
            print(col, log[col].median())
            assert(np.isclose(log[col].median(), 0.0, atol=0.5))
            valid_idxs = np.invert(orig_logs[i][col].isnull())
            cq1 = np.percentile(log[col].values[valid_idxs], 25)
            cq3 = np.percentile(log[col].values[valid_idxs], 75)
            assert(np.isclose(cq3 - cq1, 1.0))

def test_get_rgt_1():
    logs=[pandas.DataFrame({'d': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]}),
          pandas.DataFrame({'d': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}),
          pandas.DataFrame({'d': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})]
    get_rgt(logs)
    assert(np.isclose(logs[0]['RGT'].values[5], logs[1]['RGT'].values[3]))
    assert(np.isclose(logs[1]['RGT'].values[3], logs[2]['RGT'].values[1]))

def test_get_rgt_2():
    logs=[pandas.DataFrame({'c1': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, np.nan],
                            'c2': [np.nan, np.nan, np.nan, -1.0, 0.0, 1.0, 2.0, 3.0]}),
          pandas.DataFrame({'c1': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
                            'c2': [np.nan, np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}),
          pandas.DataFrame({'c1': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
                            'c2': [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})]
    prepare_logs(logs)
    get_rgt(logs)
    print(logs)
    assert(np.isclose(logs[0]['RGT'].values[5], logs[1]['RGT'].values[3]))
    assert(np.isclose(logs[1]['RGT'].values[3], logs[2]['RGT'].values[1]))
