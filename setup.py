import io
import os
import setuptools


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.md', encoding='utf-8') as fp:
    description = fp.read()

req = [
    # Non-D3M dependencies:
    'SQLAlchemy==1.2.16',
    'datamart-materialize==0.6.1',
    'datamart-profiler==0.9',
    'nltk==3.6.7',
    'numpy==1.18.2',
    'scipy==1.4.1',
    'smac==0.13.1',
    'tensorflow==2.2.0',
    'ConfigSpace==0.4.20',
    'scikit-learn==0.22.2.post1',
    'scikit-image==0.17.2',
    'torch==1.7',
    'PyYAML==5.1.2',
    'metalearn==0.6.2',
    # D3M dependencies:
    'd3m==2021.12.19',
    'd3m-automl-rpc==1.2.0',
    'd3m-sklearn-wrap==2022.2.8',
    'dsbox-primitives==1.6.1',
    'dsbox-corex==1.1.0',
    'sri-d3m==1.9.6',
    'd3m-common-primitives==2022.1.5']

setuptools.setup(name='alphad3m',
      version='0.10.dev1',
      packages=setuptools.find_packages(),
      entry_points={
          'console_scripts': [
              'alphad3m_serve = alphad3m.main:main_serve',
              'alphad3m_search = alphad3m.main:main_search',
              'alphad3m_serve_dmc = alphad3m.main:main_serve_dmc'
              ]},
      install_requires=req,
      description="AlphaD3M: NYU's AutoML System",
      long_description=description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      author='Remi Rampin, Roque Lopez, Raoni Lourenco',
      author_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      maintainer='Remi Rampin, Roque Lopez, Raoni Lourenco',
      maintainer_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'],
      license='Apache-2.0',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Scientific/Engineering',
      ])
