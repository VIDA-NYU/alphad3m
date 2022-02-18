import io
import os
import setuptools


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.md', encoding='utf-8') as fp:
    description = fp.read()

req = [
    # Non-D3M dependencies:
    'Cython',
    'SQLAlchemy==1.2.16',
    'datamart-materialize>=0.9,<0.10',
    'datamart-profiler>=0.9,<0.10',
    'nltk',
    'numpy',
    'scipy',
    'smac>=0.13,<0.14',
    'ConfigSpace>=0.4.20,<0.5',
    'scikit-learn==0.22.2.post1',
    'torch>=1.7',
    'PyYAML',

    # D3M dependencies:
    'd3m==2021.12.19',
    'd3m-automl-rpc==1.2.0',
    'metalearn==0.6.2',
]

setuptools.setup(name='alphad3m',
      version='0.11.0.dev0',
      packages=setuptools.find_packages(),
      entry_points={
          'console_scripts': [
              'alphad3m_serve = alphad3m.main:main_serve',
              'alphad3m_search = alphad3m.main:main_search'
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
