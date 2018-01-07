"""The SQLAlchemy models we use to persist data.
"""

import enum
import logging
import os
from sqlalchemy import Column, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from sqlalchemy.sql import functions
from sqlalchemy.types import Boolean, DateTime, Enum, Float, String
import uuid

from d3m_ta2_nyu.sql_uuid import UUID


logger = logging.getLogger(__name__)


Base = declarative_base()


class Module(Base):
    __tablename__ = 'modules'

    package = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    name = Column(String, primary_key=True)


class Pipeline(Base):
    __tablename__ = 'pipelines'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    origin = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False,
                          server_default=functions.now())
    task = Column(String, nullable=True)
    parameters = relationship('PipelineParameter', lazy='joined')
    runs = relationship('Run')


class PipelineModule(Base):
    __tablename__ = 'pipeline_modules'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    package = Column(String, ForeignKey('modules.package'), nullable=False)
    version = Column(String, ForeignKey('modules.version'), nullable=False)
    module_name = Column(String, ForeignKey('modules.name'), nullable=False)
    module = relationship('Module', lazy='joined')
    connections_from = relationship(
        'PipelineConnection',
        remote_side='pipeline_connections.from_module')
    connections_to = relationship(
        'PipelineConnection',
        remote_side='pipeline_connections.to_module')


class PipelineConnection(Base):
    __tablename__ = 'pipeline_connections'

    from_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                            primary_key=True)
    from_output_name = Column(String, primary_key=True)
    from_module = relationship('PipelineModule',
                               foreign_keys=['from_module_id'])

    to_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                          primary_key=True)
    to_input_name = Column(String, primary_key=True)
    to_module = relationship('PipelineModule',
                             foreign_keys=['to_module_id'])


class PipelineParameter(Base):
    __tablename__ = 'pipeline_parameters'

    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    name = Column(String, primary_key=True)
    value = Column(String, nullable=True)
    module = relationship('PipelineModule', lazy='joined')


class CrossValidation(Base):
    __tablename__ = 'cross_validations'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False)
    scores = relationship('CrossValidationScore', lazy='joined')


class CrossValidationScore(Base):
    __tablename__ = 'cross_validation_scores'

    cross_validation_id = Column(UUID, ForeignKey('cross_validations.id'),
                                 primary_key=True)
    cross_validation = relationship('CrossValidation')
    metric = Column(String, primary_key=True)
    value = Column(Float, nullable=False)


class RunType(enum.Enum):
    TRAINING = 1
    TEST = 2


class Run(Base):
    __tablename__ = 'runs'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False)
    reason = Column(String, nullable=False)
    type = Column(Enum(RunType))
    special = Column(Boolean, nullable=False, default=False)
    inputs = relationship('Input')
    outputs = relationship('Output')


class Input(Base):
    __tablename__ = 'inputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    output_name = Column(String, primary_key=True, nullable=True)
    hash = Column(String, nullable=False)


class Output(Base):
    __tablename__ = 'outputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    output_name = Column(String, primary_key=True, nullable=True)
    hash = Column(String, nullable=False)


def connect(url=None):
    """Connect to the database using an environment variable.
    """
    logging.info("Connecting to SQL database")
    if url is None:
        if 'POSTGRES_HOST' in os.environ and 'SQLITE_FILE' in os.environ:
            raise EnvironmentError("Both POSTGRES_HOST and SQLITE_FILE are set")
        elif 'POSTGRES_HOST' in os.environ:
            url = 'postgresql://{user}:{password}@{host}/{database}'.format(
                user=os.environ.get('POSTGRES_USER', 'postgres'),
                password=os.environ.get('POSTGRES_PASSWORD', ''),
                host=os.environ['POSTGRES_HOST'],
                database=os.environ.get('POSTGRES_DB', 'ta2'),
            )
        elif 'SQLITE_FILE' in os.environ:
            url = 'sqlite:///{0}'.format(os.environ['SQLITE_FILE'])
        else:
            raise EnvironmentError("No SQL database selected; please set "
                                   "SQLITE_FILE or POSTGRES_HOST")
    engine = create_engine(url, echo=False)

    if not engine.dialect.has_table(engine.connect(), 'pipelines'):
        logger.warning("The tables don't seem to exist; creating")
        Base.metadata.create_all(bind=engine)

    return engine, scoped_session(sessionmaker(bind=engine,
                                               autocommit=False,
                                               autoflush=False))
