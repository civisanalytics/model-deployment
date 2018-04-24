from unittest.mock import patch

import pytest

from civisml_deploy import utils
from civisml_deploy.tests.conftest import fetch_model


@pytest.fixture
def mock_reg_pipe():
    model, model_pipe, model_features = fetch_model('regression')

    return model_pipe


@pytest.fixture
def mock_multireg_pipe():
    model, model_pipe, model_features = fetch_model('multiregression')

    return model_pipe


@pytest.fixture
def mock_clf_pipe():
    model, model_pipe, model_features = fetch_model('classification')

    return model_pipe


@pytest.fixture
def mock_multiclf_pipe():
    model, model_pipe, model_features = fetch_model('multiclassification')

    return model_pipe


@pytest.mark.parametrize("mock_pipe", [mock_clf_pipe(), mock_multiclf_pipe(),
                                       mock_reg_pipe(), mock_multireg_pipe()])
def test_get_model_features(mock_pipe):
    true_features = ['x0', 'x1']
    assert utils._get_model_features(mock_pipe) == true_features


@patch.object(utils, 'ModelPipeline', autospec=True)
def test_fetch_civisml(mock_civmp, mock_clf_pipe):
    mock_civmp.from_existing.return_value = mock_clf_pipe
    model, pipe, feats = utils._fetch_civisml_model(123)

    mock_civmp.from_existing.assert_called_with(train_job_id=123)
    assert model == mock_clf_pipe.estimator
    assert pipe.model_name == 'clfmodel'
    assert feats == ['x0', 'x1']

    mock_civmp.reset_mock()
    mock_civmp.from_existing.return_value = mock_clf_pipe
    utils._fetch_civisml_model(123, 456)

    mock_civmp.from_existing.assert_called_with(train_job_id=123,
                                                train_run_id=456)
    assert model == mock_clf_pipe.estimator
    assert pipe.model_name == 'clfmodel'
    assert feats == ['x0', 'x1']


@patch.object(utils, 'ModelPipeline', autospec=True)
def test_get_model_dict_clf(mock_civmp, mock_clf_pipe):
    mock_civmp.from_existing.return_value = mock_clf_pipe

    model_dict = utils.get_model_dict()

    assert model_dict['method'] == 'predict_proba'
    assert model_dict['model'] == mock_clf_pipe.estimator
    assert model_dict['model_name'] == 'clfmodel'
    assert model_dict['model_pipe'] == mock_clf_pipe
    assert model_dict['model_features'] == ['x0', 'x1']
    assert model_dict['model_target_columns'] == ['target_0', 'target_1']


@patch.object(utils, 'ModelPipeline', autospec=True)
def test_get_model_dict_reg(mock_civmp, mock_reg_pipe):
    mock_civmp.from_existing.return_value = mock_reg_pipe

    model_dict = utils.get_model_dict()

    assert model_dict['method'] == 'predict'
    assert model_dict['model'] == mock_reg_pipe.estimator
    assert model_dict['model_name'] == 'regmodel'
    assert model_dict['model_pipe'] == mock_reg_pipe
    assert model_dict['model_features'] == ['x0', 'x1']
    assert model_dict['model_target_columns'] == ['target']


@patch.object(utils, 'ModelPipeline', autospec=True)
def test_get_model_dict_multiclf(mock_civmp, mock_multiclf_pipe):
    mock_civmp.from_existing.return_value = mock_multiclf_pipe

    model_dict = utils.get_model_dict()

    assert model_dict['method'] == 'predict_proba'
    assert model_dict['model'] == mock_multiclf_pipe.estimator
    assert model_dict['model_name'] == 'multiclfmodel'
    assert model_dict['model_pipe'] == mock_multiclf_pipe
    assert model_dict['model_features'] == ['x0', 'x1']
    assert model_dict['model_target_columns'] == ['target1_0', 'target1_1',
                                                  'target2_a', 'target2_b',
                                                  'target2_c']


@patch.object(utils, 'ModelPipeline', autospec=True)
def test_get_model_dict_multireg(mock_civmp, mock_multireg_pipe):
    mock_civmp.from_existing.return_value = mock_multireg_pipe

    model_dict = utils.get_model_dict()

    assert model_dict['method'] == 'predict'
    assert model_dict['model'] == mock_multireg_pipe.estimator
    assert model_dict['model_name'] == 'multiregmodel'
    assert model_dict['model_pipe'] == mock_multireg_pipe
    assert model_dict['model_features'] == ['x0', 'x1']
    assert model_dict['model_target_columns'] == ['target1', 'target2']
