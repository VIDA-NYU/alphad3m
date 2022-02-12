import io
import os
import setuptools


# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.md', encoding='utf-8') as fp:
    description = fp.read()

with open('requirements.txt') as fp:
    req = [line for line in fp if line and not line.startswith('#')]

# Temporary workaround until BYU primitives get updated
req.extend([
    'metalearn==0.6.2',
])

setuptools.setup(name='alphad3m',
      version='0.10',
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
