# CivisML Model Deployment
This repository contains the source code for a python Flask app that enables
users to deploy machine learning models built in
[CivisML](https://www.civisanalytics.com/platform/algorithms/). This Flask app
is built in the container specified by the `Dockerfile`, and the container is
deployed in its own Kubernetes pod hosted on Civis Platform. A Jupyter notebook
demonstrating how to train, deploy, and make requests of a deployed CivisML
model can be found in the `examples/` directory.

The version of CivisML that was used to train your model dictates which version
of the model-deployment Docker image you should use when you deploy. The
easiest way to check which model-deployment image you should use is to check
which version of the Civis python API client your model was trained with. The
below table tells you which image tag to use for a given version of the Civis
python API client.

| Model Deployment image tag | Civis python API client version | CivisML version |
| -------------------------- | ------------------------------- | --------------- |
| 1.0                        | 1.8                             | 2.1             |
| 1.1                        | 1.9                             | 2.2             |


## Resources of the Deployed Model
Once deployed, you will be able to make HTTP GET requests to three resources.
The "/predict" endpoint is the one model consumers will be most concerned with.
Requests for predictions can be made by passing the values of your model's
covariates as query parameters to this endpoint. You can pass your access token
either through the HTTP header, or your can pass it as a query parameter using
the "civis_service_token" variable name. An example GET to "/predict" is
constructed in the example notebook.

A GET call to the index page "/" will return a web form that will allow you to
enter values for your covariates into a GUI to get predictions. This form is
primarily meant to be consumed through Civis Platform by using the "Publish to
Report" option in your deployment's page on Platform. There is a "/features"
endpoint that returns the features of your deployed model in JSON. This is
primarily meant to populate the web form returned by GET calls to the index.

## Deploying non-CivisML Models
This repo primarily focuses on the use case of deploying models trained in
CivisML. In spite of that, we have written a development branch called
"muffnn_recommendation_engine" that shows how to deploy a simple recommendation
engine. The model uses the [muffnn](https://github.com/civisanalytics/muffnn)
library, which makes picklable TensorFlow models that follow
[the scikit-learn API](https://arxiv.org/abs/1309.0238). While users can
certainly deploy a muffnn model using this code, it is also meant as an
illustration of the changes one would need to make to repurpose this repo for
non-CivisML models.
