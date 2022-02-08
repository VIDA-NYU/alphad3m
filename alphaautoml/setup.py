import io
import os
from setuptools import setup

# pip workaround
os.chdir(os.path.abspath(os.path.dirname(__file__)))


with io.open('README.rst', encoding='utf-8') as fp:
    description = fp.read()
req = [
    'scikit-learn',
    'pandas']
setup(name='alphaAutoMLEdit',
      version='0.1.0',
      packages=['alphaAutoMLEdit'],
      entry_points={
          'console_scripts': [
              'sklearn_example = alphaAutoMLEdit.test.SklearnPipelineGenerator:main']},
      install_requires=req,
      description="NYU's AlphaAutoML system",
      long_description=description,
      author="Yamuna Krishnamurthy",
      author_email='yamuna@nyu.edu',
      maintainer="Yamuna Krishnamurthy",
      maintainer_email='yamuna@nyu.edu',
      keywords=['datadrivendiscovery', 'automl', 'nyu'])
