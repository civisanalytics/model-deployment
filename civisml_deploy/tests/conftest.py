from unittest.mock import patch, Mock

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from civisml_deploy import app, utils


def build_regression():
    lr = LinearRegression()
    # Coef: 1, -1    Int: 0.5
    lr.fit([[0, 0], [1, 0], [1, 1]], [0.5, 1.5, 0.5])

    return lr, ['x0', 'x1']


def build_multi_regression():
    lr = LinearRegression()
    lr.fit([[0, 0], [1, 0], [1, 1]], [[0.5, -0.5], [1.5, -1.5], [0.5, -0.5]])

    return lr, ['x0', 'x1']


def build_classifier():
    rng = np.random.RandomState(17)
    lr = LogisticRegression(C=10**8, random_state=rng)
    lr.fit([[0, 0], [0, 0], [1, 0], [0, 1]], [0., 1., 1., 0.])

    return lr, ['x0', 'x1']


def build_multi_classifier():
    rng = np.random.RandomState(7)
    rf = RandomForestClassifier(n_estimators=10, random_state=rng)

    x = np.array([[20,  0], [25,  0], [30,  1], [40,  0], [50,  1],
                  [60,  0], [65,  1], [70,  1], [80,  0]])
    y = np.array([[1, 0], [0, 1], [0, 2], [1, 0], [0, 1], [1, 2],
                  [0, 0], [1, 1], [1, 2]])
    rf.fit(x, y)

    return rf, ['x0', 'x1']


def fetch_model(model_type='regression'):
    model_pipe = Mock()
    if model_type == 'regression':
        model_pipe.train_result_.metadata = {'model': {'type': 'regressor'},
                                             'data':
                                             {'target_columns': ['target'],
                                              'column_names_read': ['x0', 'x1',
                                                                    'target']}
                                             }
        model, model_features = build_regression()
        model_pipe.model_name = 'regmodel'
    elif model_type == 'multiregression':
        model_pipe.train_result_.metadata = {'model': {'type': 'regressor'},
                                             'data':
                                             {'target_columns': ['target1',
                                                                 'target2'],
                                              'column_names_read': ['x0', 'x1',
                                                                    'target1',
                                                                    'target2']}
                                             }
        model, model_features = build_multi_regression()
        model_pipe.model_name = 'multiregmodel'
    elif model_type == 'classification':
        model_pipe.train_result_.metadata = {'model':
                                             {'type': 'classification'},
                                             'data':
                                             {'target_columns': ['target'],
                                              'column_names_read': ['x0', 'x1',
                                                                    'target'],
                                              'class_names': [0, 1]}}

        model, model_features = build_classifier()
        model_pipe.model_name = 'clfmodel'
    elif model_type == 'multiclassification':
        model_pipe.train_result_.metadata = {'model':
                                             {'type': 'classification'},
                                             'data':
                                             {'target_columns': ['target1',
                                                                 'target2'],
                                              'column_names_read': ['x0', 'x1',
                                                                    'target1',
                                                                    'target2'],
                                              'class_names': [[0, 1],
                                                              ['a', 'b', 'c']]}
                                             }

        model, model_features = build_multi_classifier()
        model_pipe.model_name = 'multiclfmodel'
    else:
        raise ValueError("Unexpected model type passed to fetch_model.")

    model_pipe.estimator = model

    return model, model_pipe, model_features


def create_test_app():
    _app = app.create_app()
    _app.testing = True

    return _app.test_client()


@pytest.fixture
@patch.object(app, 'output_analytics_log_columns', autospec=True)
@patch.object(utils, '_fetch_civisml_model', return_value=fetch_model(),
              autospec=True)
def reg_app(mock_fetch, mock_output_logs):
    return create_test_app()


@pytest.fixture
@patch.object(app, 'output_analytics_log_columns', autospec=True)
@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('multiregression'), autospec=True)
def multireg_app(mock_fetch, mock_output_logs):
    return create_test_app()


@pytest.fixture
@patch.object(app, 'output_analytics_log_columns', autospec=True)
@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('classification'), autospec=True)
def clf_app(mock_fetch, mock_output_logs):
    return create_test_app()


@pytest.fixture
@patch.object(app, 'output_analytics_log_columns', autospec=True)
@patch.object(utils, '_fetch_civisml_model',
              return_value=fetch_model('multiclassification'), autospec=True)
def multiclf_app(mock_fetch, mock_output_logs):
    return create_test_app()
