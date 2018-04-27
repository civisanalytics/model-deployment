from urllib.parse import parse_qs

from flask import (Blueprint, current_app, jsonify, request,
                   after_this_request, send_from_directory)
import numpy as np
import pandas as pd

from civisml_deploy.utils import (setup_logs, log_inputs_outputs,
                                  make_input_matrix)


general_logger, analytics_logger = setup_logs()

model_blueprint = Blueprint('model_blueprint', __name__)


@model_blueprint.route('/', methods=['GET'])
def index():
    """The app's index page. An index page (even if blank) is required for
    health checks on Platform. This index page dynamically generates form
    fields based on the features endpoing.

    Returns
    -------
    str
        This returns the model's name, per CivisML.
    """
    return send_from_directory(current_app.static_folder, 'index.html')


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
    input_matrix, n_top_items, user = _query_string_to_inputs(
        request.query_string, current_app.model_dict
    )

    # Returns dictionary: {'rating': [...], 'item': [...]}
    pred_dict = _predict(input_matrix, n_top_items, current_app.model_dict)
    general_logger.debug('Writing preds to JSON: \n%s', pred_dict)

    @after_this_request
    def write_analytics(response):
        log_df = pd.DataFrame(pred_dict)
        log_df['user'] = n_top_items*[user]
        log_inputs_outputs(log_df, current_app.model_dict, analytics_logger)

        return response

    return jsonify(pred_dict)


def _predict(input_matrix, n_top_items, model_dict):
    preds = getattr(model_dict['model'], model_dict['method'])(input_matrix)
    # This slice reverses the list and returns the first n_top_items from the
    # reversed list.
    inds = np.argsort(preds)[:-(n_top_items + 1):-1]
    general_logger.debug("indices for top %s items: %s", n_top_items, inds)

    return {'rating': preds[inds].tolist(),
            'item': model_dict['items'][inds].tolist()}


def _query_string_to_inputs(query_string, model_dict):
    """Parse query string and plug it into a `pd.DataFrame`.
    """
    general_logger.debug('query string: %s', query_string)
    qs_dict = parse_qs(query_string.decode('utf-8'))
    qs_dict.pop('civis_service_token', None)
    general_logger.debug('query string (token removed, if passed): %s',
                         qs_dict)

    if 'ntopitems' not in qs_dict:
        general_logger.info("Returning only top item, as user didn't pass "
                            "'ntopitems' to query string.")
        n_top_items = 1
    else:
        n_top_items = int(qs_dict['ntopitems'][0])

    input_matrix, user = make_input_matrix(qs_dict['user'][0], model_dict)

    general_logger.debug("Shape of input_matrix: %s", input_matrix.shape)
    return input_matrix, n_top_items, user
