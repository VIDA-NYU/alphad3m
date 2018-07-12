import logging
import sys

from d3m.container import Dataset

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train


logger = logging.getLogger(__name__)


@database.with_db
def train(pipeline_id, msg_queue, db):
    # Get dataset from database
    dataset, = (
        db.query(database.Pipeline.dataset)
        .filter(database.Pipeline.id == pipeline_id)
        .one()
    )

    logger.info("About to train pipeline, id=%s, dataset=%r",
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    # Training step - run pipeline on full training_data
    logger.info("Running training on full data")

    try:
        execute_train(db, pipeline_id, dataset)
    except Exception:
        logger.exception("Error running training on full data")
        sys.exit(1)

    db.commit()
