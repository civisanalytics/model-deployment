import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from civisml_deploy import utils
from civisml_deploy.tests.conftest import reg_app, clf_app


def reg_log_df():
    return pd.DataFrame({
        0: [0, 10],
        1: [10, 0],
        2: [-9.5, 10.5],
        3: [100, 200],
        4: [123, 123]
    })


def clf_log_df():
    return pd.DataFrame({
        0: [0, 10],
        1: [10, 0],
        2: [1., 0.],
        3: [0., 1.],
        4: [100, 200],
        5: [123, 123]
    })


@pytest.mark.parametrize("app, ref_df", [(reg_app(), reg_log_df()),
                                         (clf_app(), clf_log_df())])
@patch.object(utils, 'datetime', autospec=True)
def test_analytic_logs(mock_dt, app, ref_df):
    utils.CIVIS_SERVICE_VERSION = '123'
    with tempfile.NamedTemporaryFile() as tfile:
        with patch.dict(utils.os.environ, {'LOGPATH': tfile.name}):
            utils.setup_logs()

            mock_utc = Mock()
            mock_utc.strftime.side_effect = ['100', '200']
            mock_dt.utcnow.return_value = mock_utc

            app.get('/predict?x0=0&x1=10')
            app.get('/predict?x0=10&x1=0')

            test_df = pd.read_csv(os.environ.get('LOGPATH'), header=None)
            pd.testing.assert_frame_equal(test_df, ref_df)


@pytest.mark.parametrize("app, ref_df", [(reg_app(), reg_log_df()),
                                         (clf_app(), clf_log_df())])
@patch.object(utils, 'datetime', autospec=True)
def test_analytic_logs_no_svc_version(mock_dt, app, ref_df):
    utils.CIVIS_SERVICE_VERSION = None
    with tempfile.NamedTemporaryFile() as tfile:
        with patch.dict(utils.os.environ, {'LOGPATH': tfile.name}):
            utils.setup_logs()

            mock_utc = Mock()
            mock_utc.strftime.side_effect = ['100', '200']
            mock_dt.utcnow.return_value = mock_utc

            app.get('/predict?x0=0&x1=10')
            app.get('/predict?x0=10&x1=0')

            test_df = pd.read_csv(os.environ.get('LOGPATH'), header=None)
            pd.testing.assert_frame_equal(test_df, ref_df.iloc[:, :-1])
