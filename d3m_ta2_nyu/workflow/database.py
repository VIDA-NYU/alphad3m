"""The SQLAlchemy models we use to persist data.
"""

import enum
import functools
import logging
from sqlalchemy import Column, ForeignKey, create_engine, func, not_, select, \
    Binary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import column_property, relationship, sessionmaker
from sqlalchemy.sql import functions
from sqlalchemy.types import Boolean, DateTime, Enum, Float, String

from d3m_ta2_nyu.sql_uuid import UUID, UuidMixin


logger = logging.getLogger(__name__)


Base = declarative_base()


class Pipeline(UuidMixin, Base):
    __tablename__ = 'pipelines'

    origin = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False,
                          server_default=functions.now())
    task = Column(String, nullable=True)
    parameters = relationship('PipelineParameter', lazy='joined')
    modules = relationship('PipelineModule')
    connections = relationship('PipelineConnection')
    runs = relationship('Run')


class PipelineModule(UuidMixin, Base):
    __tablename__ = 'pipeline_modules'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)
    package = Column(String, nullable=False)
    version = Column(String, nullable=False)
    name = Column(String, nullable=False)
    #connections_from = relationship('PipelineConnection')
    #connections_to = relationship('PipelineConnection')


class PipelineConnection(Base):
    __tablename__ = 'pipeline_connections'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)

    from_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                            primary_key=True)
    from_module = relationship('PipelineModule',
                               foreign_keys=[from_module_id],
                               backref='connections_from')
    from_output_name = Column(String, primary_key=True)

    to_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                          primary_key=True)
    to_module = relationship('PipelineModule',
                             foreign_keys=[to_module_id],
                             backref='connections_to')
    to_input_name = Column(String, primary_key=True)


class PipelineParameter(Base):
    __tablename__ = 'pipeline_parameters'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)

    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    name = Column(String, primary_key=True)
    value = Column(String, nullable=True)


class CrossValidation(UuidMixin, Base):
    __tablename__ = 'cross_validations'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False,
                  server_default=functions.now())
    scores = relationship('CrossValidationScore', lazy='joined')


class CrossValidationScore(Base):
    __tablename__ = 'cross_validation_scores'

    cross_validation_id = Column(UUID, ForeignKey('cross_validations.id'),
                                 primary_key=True)
    cross_validation = relationship('CrossValidation')
    metric = Column(String, primary_key=True)
    value = Column(Float, nullable=False)


class RunType(enum.Enum):
    TRAIN = 1
    TEST = 2


class Run(UuidMixin, Base):
    __tablename__ = 'runs'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False,
                  server_default=functions.now())
    reason = Column(String, nullable=False)
    type = Column(Enum(RunType), nullable=False)
    special = Column(Boolean, nullable=False, default=False)
    inputs = relationship('Input')
    outputs = relationship('Output')


class Input(Base):
    __tablename__ = 'inputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    input_name = Column(String, primary_key=True, nullable=True)
    value = Column(Binary, nullable=False)


class Output(Base):
    __tablename__ = 'outputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    output_name = Column(String, primary_key=True, nullable=True)
    value = Column(Binary, nullable=False)


# Trained true iff there's a Run with special=False and type=TRAIN
Pipeline.trained = column_property(
    select(
        [func.count(Run.id)]
    ).where((Run.pipeline_id == Pipeline.id) &
            (not_(Run.special)) &
            (Run.type == RunType.TRAIN))
    .as_scalar() != 0
)


def connect(filename):
    """Connect to the database using an environment variable.
    """
    logger.info("Connecting to SQL database")
    url = 'sqlite:///{0}'.format(filename)
    engine = create_engine(url, echo=False)

    if not engine.dialect.has_table(engine.connect(), 'pipelines'):
        logger.warning("The tables don't seem to exist; creating")
        Base.metadata.create_all(bind=engine)

    return engine, sessionmaker(bind=engine,
                                autocommit=False,
                                autoflush=False)


def with_db(wrapped):
    @functools.wraps(wrapped)
    def wrapper(*args, db_filename=None, **kwargs):
        engine, DBSession = connect(db_filename)
        db = DBSession()
        try:
            return wrapped(*args, **kwargs, db=db)
        finally:
            db.close()
    return wrapper
