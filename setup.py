import os
from setuptools import find_packages, setup


def main():
    def read(fname):
        with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
            return _in.read()

    required = read('requirements.txt').strip().split()

    __version__ = None
    exec(read('civisml_deploy/_version.py'))

    setup(version=__version__,
          name="civisml_deploy",
          author="Civis Analytics Inc",
          author_email="dsrd@civisanalytics.com",
          url="https://www.civisanalytics.com",
          description="Easily deploy CivisML models as web apps.",
          packages=find_packages(),
          long_description=read('README.md'),
          install_requires=required
          )


if __name__ == "__main__":
    main()
