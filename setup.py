import io
import os
from setuptools import setup


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.rst', encoding='utf-8') as fp:
    description = fp.read()
req = [
    'grpcio',
    'SQLAlchemy',
    'scikit-learn',
    'scikit-image',
    'scikit-learn',
    'pandas',
    'dsbox-datacleaning',
    'smac']
setup(name='d3m_ta2_nyu',
      version='0.7-pre',
      packages=['d3m_ta2_nyu'],
      entry_points={
          'console_scripts': [
              'ta2_search = d3m_ta2_nyu.main:main_search',
              'ta2_serve = d3m_ta2_nyu.main:main_serve',
              'ta2_test = d3m_ta2_nyu.main:main_test']},
      install_requires=req,
      description="NYU's TA2 system",
      long_description=description,
      author="Remi Rampin",
      author_email='remi.rampin@nyu.edu',
      maintainer="Remi Rampin",
      maintainer_email='remi.rampin@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'])
