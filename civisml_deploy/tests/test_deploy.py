import json
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest

from civisml_deploy.controllers.deployed_model import (_predict,
                                                       _query_string_to_df)
from civisml_deploy import utils
from civisml_deploy.tests.conftest import (fetch_model, reg_app, multireg_app,
                                           clf_app, multiclf_app)


def assert_dicts_equal(true_dict, test_dict):
    assert true_dict.keys() == test_dict.keys()
    for k, v in true_dict.items():
        assert np.allclose(v, test_dict[k])


def _get_true_preds(model_type, X):
    model, model_pipe, feats = fetch_model(model_type)

    metadata = model_pipe.train_result_.metadata
    if metadata['model']['type'] == 'regressor':
        method = 'predict'
    else:
        method = 'predict_proba'

    target_cols = utils._get_target_columns(metadata)

    preds = getattr(model, method)(X)

    if model_type == 'multiclassification':
        preds = np.hstack(preds)

    return dict(zip(target_cols, preds.flatten())), feats


@pytest.mark.parametrize("app,model_type,X", [
        (reg_app(), 'regression', [[3, 1]]),
        (clf_app(), 'classification', [[0, 10]]),
        (multireg_app(), 'multiregression', [[3, 1]]),
        (multiclf_app(), 'multiclassification', [[0, 1]])
    ])
def test_predict_app(app, model_type, X):
    truth, feats = _get_true_preds(model_type, X)
    querystr = urlencode(dict(zip(feats, np.array(X).flatten())))

    pred = app.get('/predict?' + querystr).data.decode('utf-8')
    pred = json.loads(pred)

    assert_dicts_equal(truth, pred)


@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('regression'), autospec=True)
def test_predict_private_reg(mock_fetch):
    model_dict = utils.get_model_dict()
    df = pd.DataFrame({'x0': [3], 'x1': [1]})
    assert np.allclose(_predict(df, model_dict), np.array([2.5]))


@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('multiregression'), autospec=True)
def test_predict_private_multireg(mock_fetch):
    model_dict = utils.get_model_dict()
    df = pd.DataFrame({'x0': [3], 'x1': [1]})
    assert np.allclose(_predict(df, model_dict), np.array([[2.5, -2.5]]))


@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('classification'), autospec=True)
def test_predict_private_clf(mock_fetch):
    model_dict = utils.get_model_dict()
    df = pd.DataFrame({'x0': [0], 'x1': [10]})
    assert np.allclose(_predict(df, model_dict), np.array([[1, 0]]))


@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('multiclassification'), autospec=True)
def test_predict_private_multiclf(mock_fetch):
    model_dict = utils.get_model_dict()
    df = pd.DataFrame({'x0': [0], 'x1': [1]})
    assert np.allclose(_predict(df, model_dict),
                       np.array([[0.7, 0.3, 0.2, 0.4, 0.4]]))


@pytest.mark.parametrize("app,name", [
        (reg_app(), 'regmodel'),
        (clf_app(), 'clfmodel'),
        (multireg_app(), 'multiregmodel'),
        (multiclf_app(), 'multiclfmodel')
    ])
def test_index(app, name):
    html = app.get('/').data.decode('utf-8')
    assert 'html' in html


def test_query_to_df():
    query_bstr = b'x0=3&x1=2&x2=1'
    df = pd.DataFrame({'x0': ['3'], 'x1': ['2'], 'x2': ['1']})
    pd.testing.assert_frame_equal(df, _query_string_to_df(query_bstr))

    query_bstr = b'x0=3&x1=2&x2=1&civis_service_token=foobar'
    pd.testing.assert_frame_equal(df, _query_string_to_df(query_bstr))
