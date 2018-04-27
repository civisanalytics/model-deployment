from copy import copy
from datetime import datetime
from io import BytesIO, StringIO
import logging
import pickle
import os

import civis
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

general_logger = logging.getLogger("civisml_deploy")
CIVIS_SERVICE_VERSION = os.environ.get('CIVIS_SERVICE_VERSION')
USER_PREFIX = os.environ.get('USER_PREFIX')
ITEM_PREFIX = os.environ.get('ITEM_PREFIX')


def _fetch_recommender(model_file_id, user_item_file_id):
    """Pull down pickled muffnn recommendation engine and comma-separated list
    of user-item columns in training data.
    """
    # Load muffnn model from /files endpoint
    modelbytes = BytesIO()
    general_logger.debug('Fetching muffnn model with file_id %s...',
                         model_file_id)
    civis.io.civis_to_file(model_file_id, modelbytes)

    modelbytes.seek(0)
    model = pickle.load(modelbytes)
    general_logger.debug('model loaded.')

    # Get user-item matrix columns from original long data
    colbytes = BytesIO()
    civis.io.civis_to_file(user_item_file_id, colbytes)

    colbytes.seek(0)
    user_item_columns = colbytes.read().decode('utf-8').strip().split(',')

    return model, user_item_columns


def _get_users_items(user_item_columns):
    """Given the list of user-item columns, this function returns the number of
    users, the list of items, and a sparse diagonal matrix of size
    (n_items, n_items).
    """
    items = []
    n_users = 0
    for col in user_item_columns:
        # Ensure only users and items in column list
        if col.startswith(USER_PREFIX):
            n_users += 1
        elif col.startswith(ITEM_PREFIX):
            items.append(col)
        else:
            raise ValueError("Column '{}' doesn't start with either the user "
                             "prefix '{}' or the item prefix '{}', as "
                             "expected.".format(col, USER_PREFIX, ITEM_PREFIX))

    n_items = len(items)
    diag_item_matrix = sparse.diags([1], shape=(n_items, n_items),
                                    format='csr')

    return n_users, np.array(items), diag_item_matrix


def setup_logs():
    """Creates and configures logger instances for general stderr logging and
    also for logging analytics I/O to a file.

    Returns
    -------
    logging.Logger, logging.logger
        Returns stderr stream logger and analytics file logger, respectively.
    """
    if os.environ.get('DEBUG'):
        stream_level = logging.DEBUG
    else:
        stream_level = logging.WARNING

    # Setup standard stream logger
    general_logger.setLevel(stream_level)
    lh = logging.StreamHandler()
    lh.setLevel(stream_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')
    lh.setFormatter(formatter)
    general_logger.addHandler(lh)

    # Setup file logger for analytics
    logpath = os.environ.get('LOGPATH', os.path.join(os.getcwd(),
                                                     'logfile.log'))
    general_logger.info('Analytics logs will be written to %s', logpath)

    analytics_logger = logging.getLogger('analytics_logger')
    analytics_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.INFO)
    analytics_logger.addHandler(fh)

    return general_logger, analytics_logger


def log_inputs_outputs(log_df, model_dict, analytics_logger):
    """Write the input and output of a prediction call.

    Parameters
    ----------
    log_df : pd.DataFrame
        DataFrame holding the data to log.
    analytics_logger : logging.Logger
        Instance of analytics logger.
    """
    log_df['timestamp'] = datetime.utcnow().strftime('%Y_%m_%d_%H:%M:%S.%f')
    output_cols = copy(model_dict['analytic_logging_columns'])

    if CIVIS_SERVICE_VERSION:
        log_df['civis_service_version'] = CIVIS_SERVICE_VERSION
        output_cols.append('civis_service_version')

    outdf = StringIO()
    log_df.to_csv(outdf, index=False, header=False)
    outdf.seek(0)
    analytics_logger.info(outdf.read().rstrip())


def output_analytics_log_columns(model_dict):
    """This will write the names of the analytics log columns to the general
    logger.

    Parameters
    ----------
    model_dict : dict
        Dictionary created by `get_model_dict` that contains a
        `civis.ml.ModelPipeline` object and associated metadata.
    """
    # Output the column order of the analytics logs to stdout
    output_cols = copy(model_dict['analytic_logging_columns'])

    if CIVIS_SERVICE_VERSION:
        output_cols.append('civis_service_version')
    else:
        general_logger.warning("No value for $CIVIS_SERVICE_VERSION found!")

    general_logger.info("Analytics logging columns: ")
    general_logger.info(','.join(output_cols))


def get_model_dict():
    """This reads the job and (optionally) run IDs for a CivisML model job from
    the environemnt, pulls that model from Civis Platform, and loads the
    ModelPipeline and some of its associated metadata into a dictionary.

    Returns
    -------
    dict
        Dictionary containing objects and metadata for the desired CivisML
        modeling job.
    """
    # Pull existing model and list of user and item columns
    model_file_id = os.environ.get('MODEL_FILE_ID')
    user_item_file_id = os.environ.get('USER_ITEM_COLUMNS_FILE_ID')
    model, user_item_columns = _fetch_recommender(model_file_id,
                                                  user_item_file_id)

    # Make sparse diagonal item block; count users and items
    n_users, items, diag_item_matrix = _get_users_items(user_item_columns)

    model_dict = {}
    model_dict['model'] = model
    model_dict['items'] = items
    model_dict['n_users'] = n_users
    model_dict['diag_item_matrix'] = diag_item_matrix
    model_dict['user_item_columns'] = user_item_columns
    model_dict['model_features'] = ['user']
    model_dict['model_name'] = "File ID: {}".format(model_file_id)

    model_dict['method'] = 'predict'
    model_dict['model_target_columns'] = 'rating'
    model_dict['analytic_logging_columns'] = ['user', 'item', 'rating',
                                              'timestamp']

    general_logger.debug("model_dict: \n%s", model_dict)

    return model_dict


def make_input_matrix(user, model_dict):
    """Given a user and
    """
    user = str(user)

    if not user.startswith(USER_PREFIX):
        user = USER_PREFIX + user

    columns = model_dict['user_item_columns']
    n_users = model_dict['n_users']
    n_items = len(model_dict['items'])

    if user not in columns:
        general_logger.info("User %s not found in in training columns", user)
        user_block = sparse.csr_matrix((n_items, n_users))
    else:
        user_ind = np.argwhere(np.array(columns) == user).ravel()[0]
        # first user
        if user_ind == 0:
            user_block = sparse.hstack((np.ones(shape=(n_items, 1)),
                                        csr_matrix((n_items, n_users - 1))))
        # last user
        elif user_ind == n_users - 1:
            user_block = sparse.hstack((csr_matrix((n_items, n_users - 1)),
                                        np.ones(shape=(n_items, 1))))
        # in between user
        else:
            user_block = sparse.hstack((csr_matrix((n_items, user_ind)),
                                        np.ones(shape=(n_items, 1)),
                                        csr_matrix((n_items,
                                                    n_users - user_ind - 1))))

    return sparse.hstack((user_block, model_dict['diag_item_matrix'])), user
