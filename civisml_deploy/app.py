from flask import Flask

from civisml_deploy.controllers.deployed_model import model_blueprint
from civisml_deploy.utils import get_model_dict, output_analytics_log_columns


def create_app():
    """Create a Flask app that deploys a model from CivisML. This will load the
    model, setup logging, and create the endpoints necessary for a deployed
    model.

    Returns
    -------
    flask.Flask
        Instantiated Flask app.
    """
    app = Flask(__name__)
    app.register_blueprint(model_blueprint)

    app.model_dict = get_model_dict()
    output_analytics_log_columns(app.model_dict)

    return app
