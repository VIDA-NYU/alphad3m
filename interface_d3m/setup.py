import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="interface_d3m",
    version="0.0.1",
    author="Roque Lopez",
    author_email="rlopez@nyu.edu",
    description="Library to use D3M AutoML Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'gitdb2==2.0.6',
        'd3m @ git+https://gitlab.com/datadrivendiscovery/d3m.git@v2020.1.9',
        'ta3ta2-api @ git+https://gitlab.com/datadrivendiscovery/ta3ta2-api.git@31b8d2573e702aed3d70fa8192f9ef9b006ccd97',
    ],
    python_requires='>=3.6',
)