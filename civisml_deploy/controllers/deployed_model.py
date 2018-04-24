from urllib.parse import parse_qs

from flask import Blueprint, current_app, jsonify, \
    request, after_this_request, \
    send_from_directory
import numpy as np
import pandas as pd

from civisml_deploy.utils import setup_logs, log_inputs_outputs


general_logger, analytics_logger = setup_logs()

model_blueprint = Blueprint('model_blueprint', __name__)


@model_blueprint.route('/', methods=['GET'])
def index():
    """The app's index page. An index page (even if blank) is required for
    health checks on Platform. This index page dynamically generates form
    fields based on the features endpoing.
    """
    return send_from_directory('/static', 'index.html')


@model_blueprint.route('/features', methods=['GET'])
def features():
    """This defines the /features endpoint for the deployed model. It returns
    a list of the possible covariates.
    """
    return jsonify(current_app.model_dict['model_features'])


@model_blueprint.route('/predict', methods=['GET'])
def predict():
    """This defines the /predict endpoint for the deployed model. Covariates
    are taken in as query parameters, and a CSV file is sent in reponse.
    """
    general_logger.debug('predict called')
    input_df = _query_string_to_df(request.query_string)

    general_logger.debug(input_df)
    preds = np.atleast_2d(_predict(input_df, current_app.model_dict))

    pred_dict = {}
    # Pack predictions into dict
    for i, col in enumerate(current_app.model_dict['model_target_columns']):
        pred_dict[col] = preds[:, i].tolist()

    general_logger.debug('Writing preds to JSON: \n%s', pred_dict)

    @after_this_request
    def write_analytics(response):
        log_inputs_outputs(input_df, preds, analytics_logger)
        return response

    return jsonify(pred_dict)


def _predict(df, model_dict):
    """We break this function out separately to take advantage of memoization,
    if that's something we'd like to add (see Flask-Cache docs for more).
    """
    df = df[model_dict['model_features']]
    general_logger.debug('Input covariate df: %s', df)

    preds = getattr(model_dict['model'], model_dict['method'])(df)

    # Multioutput classifiers return lists, rather than np.arrays
    if isinstance(preds, list):
        preds = np.hstack(preds)

    return preds


def _query_string_to_df(query_string):
    """Parse query string and plug it into a `pd.DataFrame`.
    """
    qs_dict = parse_qs(query_string.decode('utf-8'))
    qs_dict.pop('civis_service_token', None)

    general_logger.debug('query string (token removed, if passed): %s',
                         qs_dict)
    return pd.DataFrame(qs_dict, index=range(1))
