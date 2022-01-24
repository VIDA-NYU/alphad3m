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
setup(name='alphad3m',
      version='0.10.dev0',
      packages=['alphad3m'],
      package_data={'alphad3m': ['pipelines/*.yaml']},
      entry_points={
          'console_scripts': [
              'ta2_search = alphad3m.main:main_search',
              'ta2_serve = alphad3m.main:main_serve',
              'ta2_test = alphad3m.main:main_test']},
      install_requires=req,
      description="AlphaD3M: NYU's AutoML System",
      long_description=description,
      long_description_content_type='text/markdown',
      author="Remi Rampin, Roque Lopez, Raoni Lourenco",
      author_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      maintainer="Remi Rampin, Roque Lopez, Raoni Lourenco",
      maintainer_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'],
      license='Apache-2.0',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Scientific/Engineering',
      ])
