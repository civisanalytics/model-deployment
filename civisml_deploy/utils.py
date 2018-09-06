from datetime import datetime
from io import BytesIO, StringIO
import logging
import os

import civis
from civis.io import civis_to_file
import joblib
import pandas as pd

general_logger = logging.getLogger("civisml_deploy")
CIVIS_SERVICE_VERSION = os.environ.get('CIVIS_SERVICE_VERSION')


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


def log_inputs_outputs(input_df, preds, analytics_logger):
    """Write the input and output of a prediction call.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame holding the input data passed by the user.
    preds : np.ndarray
        The predictions output by your model.
    analytics_logger : logging.Logger
        Instance of analytics logger.
    """

    # Write inputs, output predictions, and time stamp to logfile
    logdf = pd.merge(input_df.reset_index(), pd.DataFrame(preds).reset_index(),
                     on='index')
    logdf['timestamp'] = datetime.utcnow().strftime('%Y_%m_%d_%H:%M:%S.%f')

    if CIVIS_SERVICE_VERSION:
        logdf['civis_service_version'] = CIVIS_SERVICE_VERSION

    outdf = StringIO()
    logdf.drop('index', axis=1).to_csv(outdf, index=False, header=False)
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
    output_cols = []
    output_cols.extend(model_dict['model_features'])
    output_cols.extend(model_dict['model_target_columns'])

    output_cols.append('timestamp')

    if CIVIS_SERVICE_VERSION:
        output_cols.append('civis_service_version')
    else:
        general_logger.warning("No value for $CIVIS_SERVICE_VERSION found!")

    general_logger.info("Analytics logging columns: ")
    general_logger.info(','.join(output_cols))


def _fetch_surprise_model(file_id):
    general_logger.debug("Pulling model from Civis file %s", file_id)

    # Get model
    with BytesIO() as buf:
        civis_to_file(file_id, buf)
        buf.seek(0)
        model = joblib.load(buf)

    general_logger.debug('model loaded.')

    # Get model name
    client = civis.APIClient()
    model_name = client.files.get(file_id)['name']

    # Model features are always user id (uid) and item id (iid)
    model_features = ['uid', 'iid']

    return model, model_name, model_features


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
    # Pull existing model
    file_id = os.environ.get('MODEL_FILE_ID')

    model, model_name, model_features = _fetch_surprise_model(file_id)

    model_dict = {}

    model_dict['model'] = model
    model_dict['model_name'] = model_name
    model_dict['model_features'] = model_features
    model_dict['model_target_columns'] = ['prediction']

    general_logger.debug("model_dict: \n%s", model_dict)

    return model_dict
