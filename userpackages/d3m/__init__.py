"""D3M package for VisTrails.

This package contains primitives and utilities modules creates for D3M project.

"""

from __future__ import division

from vistrails.core.requirements import require_python_module


identifier = 'org.vistrails.vistrails.d3m'
name = 'D3M'
version = '0.0.1'


def package_dependencies():
    return ['org.vistrails.vistrails.sklearn']


def package_requirements():
    require_python_module('dsbox', {
        'pip': ['pandas >= 0.20.1', 'langdetect >= 1.0.7',
                'dsbox-dataprofiling', 'dsbox-datacleaning']})
