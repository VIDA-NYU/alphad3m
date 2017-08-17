import logging
import os
import sys


logging.basicConfig(level=logging.INFO)


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from d3m_ta2_vistrails.main import main  # noqa


main()
