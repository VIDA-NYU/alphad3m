import io
import os
from setuptools import setup


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.md', encoding='utf-8') as fp:
    description = fp.read()
req = [
    'grpcio',
    'PyYAML',
    'SQLAlchemy',
    'scikit-learn',
    'scikit-image',
    'pandas',
    'smac']
setup(name='d3m_ta2_nyu',
      version='2020.12.08',
      packages=['d3m_ta2_nyu'],
      package_data={'d3m_ta2_nyu': ['pipelines/*.yaml']},
      entry_points={
          'console_scripts': [
              'ta2_search = d3m_ta2_nyu.main:main_search',
              'ta2_serve = d3m_ta2_nyu.main:main_serve',
              'ta2_test = d3m_ta2_nyu.main:main_test']},
      install_requires=req,
      description="AlphaD3M: NYU's AutoML System",
      long_description=description,
      author="Remi Rampin, Roque Lopez, Raoni Lourenco",
      author_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      maintainer="Remi Rampin, Roque Lopez, Raoni Lourenco",
      maintainer_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'],
      license='Apache-2.0',
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
      ])
