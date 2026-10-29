"""Central configuration. Values can be overridden through create_app(test_config)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    ARTIFACT_NAME = "hr_attrition_artifact.joblib"
    LOCAL_CSV_NAME = "ibm_hr_attrition.csv"

    # Training defaults (all can be overridden from the Model Training page)
    DEFAULT_SEED = 42
    DEFAULT_FOLDS = 5
    TEST_SIZE = 0.2
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_SELECTION_METRIC = "average_precision"

    # Train automatically in the background at startup when no valid artifact exists
    AUTO_TRAIN = True
    # Load the dataset / artifact in a background thread at startup
    BOOTSTRAP_ON_START = True

    MAX_PAGE_SIZE = 100
    JSON_SORT_KEYS = False
