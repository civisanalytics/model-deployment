FROM civisanalytics/datascience-python:4.0.1
MAINTAINER support@civisanalytics.com

EXPOSE 3838

RUN mkdir -p /var/log/spectrum
ENV LOGPATH=/var/log/spectrum/spectrum.log

RUN pip install civis==1.9
RUN pip install civisml-extensions==0.1.7
RUN pip install muffnn==2.1.0

ADD . /mod-deploy
ADD civisml_deploy/static /static
RUN cd /mod-deploy && python setup.py install

CMD if [ -z $CIVIS_SERVICE_PORT ]; then CIVIS_SERVICE_PORT=3838; fi && cd /mod-deploy/civisml_deploy && gunicorn -b 0.0.0.0:$CIVIS_SERVICE_PORT -w 4 run:app
