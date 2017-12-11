import enum
import logging
import os
from sqlalchemy import Column, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from sqlalchemy.sql import functions
from sqlalchemy.types import Boolean, DateTime, Enum, Float, String
import uuid


Base = declarative_base()


class Pipeline(Base):
    __tablename__ = 'pipelines'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    origin = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False,
                          server_default=functions.now())
    task = Column(String, nullable=True)


class PipelineModule(Base):
    __tablename__ = 'pipeline_modules'

    id = Column(String, primary_key=True)
    package = Column(String, ForeignKey('modules.package'), nullable=False)
    version = Column(String, ForeignKey('modules.version'), nullable=False)
    module_name = Column(String, ForeignKey('modules.name'), nullable=False)
    module = relationship('Module')


class Module(Base):
    __tablename__ = 'modules'

    package = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    name = Column(String, primary_key=True)


class PipelineConnection(Base):
    __tablename__ = 'pipeline_connections'

    from_module_id = Column(String, ForeignKey('pipeline_modules.id'),
                            primary_key=True)
    from_output_name = Column(String, primary_key=True)
    from_module = relationship('Module', foreign_keys=['from_module_id',
                                                       'from_output_name'])

    to_module_id = Column(String, ForeignKey('pipeline_modules.id'),
                          primary_key=True)
    to_input_name = Column(String, primary_key=True)
    to_module = relationship('Module', foreign_keys=['to_module_id',
                                                     'to_input_name'])


class PipelineParameter(Base):
    __tablename__ = 'pipeline_parameters'

    module_id = Column(String, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    name = Column(String, primary_key=True)
    value = Column(String, nullable=True)


class CrossValidation(Base):
    __tablename__ = 'cross_validations'

    id = Column(String, primary_key=True)
    pipeline_id = Column(String, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False)


class CrossValidationScore(Base):
    __tablename__ = 'cross_validation_scores'

    cross_validation_id = Column(String, ForeignKey('cross_validations.id'),
                                 primary_key=True)
    cross_validation = relationship('CrossValidation', backref='scores')
    metric = Column(String, primary_key=True)
    value = Column(Float, nullable=False)


class RunType(enum.Enum):
    TRAINING = 1
    TEST = 2


class Run(Base):
    __tablename__ = 'runs'

    id = Column(String, primary_key=True)
    pipeline_id = Column(String, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False)
    reason = Column(String, nullable=False)
    type = Column(Enum(RunType))
    special = Column(Boolean, nullable=False, default=False)


class Input(Base):
    __tablename__ = 'inputs'

    run_id = Column(String, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(String, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    output_name = Column(String, primary_key=True, nullable=True)
    hash = Column(String, nullable=False)


class Output(Base):
    __tablename__ = 'outputs'

    run_id = Column(String, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(String, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
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

    return engine, scoped_session(sessionmaker(bind=engine,
                                               autocommit=False,
                                               autoflush=False))
