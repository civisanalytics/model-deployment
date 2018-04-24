from datetime import datetime
from io import StringIO
import logging
import os

from civis.ml import ModelPipeline
import pandas as pd

general_logger = logging.getLogger("civisml_deploy")
CIVIS_SERVICE_VERSION = os.environ.get('CIVIS_SERVICE_VERSION')


def _get_model_features(model_pipe):
    """Get features from a `civis.ml.ModelPipeline`
    """
    model_features = []
    target_cols = model_pipe.train_result_.metadata['data']['target_columns']
    for feat in model_pipe.train_result_.metadata['data']['column_names_read']:
        if feat not in target_cols:
            model_features.append(feat)

    return model_features


def _fetch_civisml_model(job_id, run_id=None):
    """Fetch a `civis.ml.ModelPipeline` given a job_id and optionally a run_id
    """
    if run_id is not None:
        general_logger.debug('Fetching model with job_id %s and run_id %s...',
                             job_id, run_id)
        model_pipe = ModelPipeline.from_existing(train_job_id=job_id,
                                                 train_run_id=run_id)
    else:
        general_logger.debug('Fetching latest run of model with job_id %s...',
                             job_id)
        model_pipe = ModelPipeline.from_existing(train_job_id=job_id)

    model_features = _get_model_features(model_pipe)
    model = model_pipe.estimator
    general_logger.debug('model loaded.')

    return model, model_pipe, model_features


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


def _get_target_columns(metadata):
    """Given ModelPipeline metadata, construct the list of columns output by
    the model's prediction method.
    """
    target_cols = []
    if metadata['model']['type'] == 'classification':
        # Deal with multilabel models, if necessary
        if len(metadata['data']['target_columns']) > 1:
            for i, col in enumerate(metadata['data']['target_columns']):
                for cls in metadata['data']['class_names'][i]:
                    target_cols.append(col + '_' + str(cls))
        else:
            col = metadata['data']['target_columns'][0]
            for cls in metadata['data']['class_names']:
                target_cols.append(col + '_' + str(cls))
    else:
        for col in metadata['data']['target_columns']:
            target_cols.append(col)

    return target_cols


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
    job_id = os.environ.get('MODEL_JOB_ID')
    run_id = os.environ.get('MODEL_RUN_ID')

    model, model_pipe, model_features = _fetch_civisml_model(job_id, run_id)

    model_dict = {}
    model_dict['model'] = model
    model_dict['model_pipe'] = model_pipe
    model_dict['model_name'] = model_pipe.model_name
    model_dict['model_features'] = model_features

    metadata = model_pipe.train_result_.metadata

    # Get relevant prediction method
    if metadata['model']['type'] == 'classification':
        model_dict['method'] = 'predict_proba'
    else:
        model_dict['method'] = 'predict'

    # Get target columns
    model_dict['model_target_columns'] = _get_target_columns(metadata)

    general_logger.debug("model_dict: \n%s", model_dict)

    return model_dict
