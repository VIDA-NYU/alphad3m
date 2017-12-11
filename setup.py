import io
import os
from setuptools import setup


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


req = [
    'grpcio',
    'scikit-learn',
    'scikit-image']
setup(name='d3m_ta2_nyu',
      version='0.1',
      packages=['d3m_ta2_nyu'],
      entry_points={
          'console_scripts': [
              'ta2_search = d3m_ta2_nyu.main:main_search',
              'ta2_serve = d3m_ta2_nyu.main:main_serve',
              'ta2_test = d3m_ta2_nyu.main:main_test']},
      install_requires=req,
      description="NYU's TA2 system",
      author="Remi Rampin",
      author_email='remi.rampin@nyu.edu',
      maintainer="Remi Rampin",
      maintainer_email='remi.rampin@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'])
