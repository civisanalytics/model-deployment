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
    """This defines the /predict endpoint for the deployed model. A user ID and
    item ID are taken in as query parameters, and a JSON is sent in reponse.
    """
    general_logger.debug('predict called')
    input_dict = _query_string_to_dict(request.query_string)

    for k in input_dict:
        if k not in current_app.model_dict['model_features']:
            general_logger.info("Removing extraneous parameter '%s' from "
                                "input.", k)
            input_dict.pop(k)

    general_logger.debug(input_dict)

    pred = current_app.model_dict['model'].predict(**input_dict)
    # Pack predictions into dict
    pred_dict = {}
    pred_dict[current_app.model_dict['model_target_columns'][0]] = [pred.est]

    general_logger.debug('Writing preds to JSON: \n%s', pred_dict)

    _write_analytics(input_dict, pred.est, current_app)

    return jsonify(pred_dict)


@model_blueprint.route('/predict_top_n', methods=['GET'])
def predict_top_n():
    """This defines the /predict_top_n endpoint for the deployed model. A user
    ID 'uid' and desired number of recommendations 'n' are taken in as query
    parameters, and a JSON is sent in reponse.
    """
    general_logger.debug('predict_top_n called')
    input_dict = _query_string_to_dict(request.query_string)

    # LOOP OVER model.trainset.all_items()
    return jsonify({})


def _write_analytics(input_dict, preds, current_app):
    preds = np.atleast_2d(preds)

    input_df = pd.DataFrame(input_dict, index=range(preds.shape[0]))
    # Ensure columns are in appropriate order
    log_inputs_outputs(input_df[current_app.model_dict['model_features']],
                       preds, analytics_logger)


def _query_string_to_dict(query_string):
    """Parse query string and plug it into a `pd.DataFrame`.
    """
    raw_qs_dict = parse_qs(query_string.decode('utf-8'))
    raw_qs_dict.pop('civis_service_token', None)

    qs_dict = {}
    # Take 0th item of single-entry lists returned by parse_qs
    for k, v in raw_qs_dict.items():
        qs_dict[k] = v[0]

    general_logger.debug('query string (token removed, if passed): %s',
                         qs_dict)
    return qs_dict
