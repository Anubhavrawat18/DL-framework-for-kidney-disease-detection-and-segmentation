"""Central paths and constants for local (Windows) runs."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'bbdd_kidney')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')

MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'manifest_with_splits.pkl')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

for _d in (CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

IMG_SIZE = (512, 512)
NUM_LOCAL_PATHOLOGY_IDS = 9
NUM_GLOBAL_PATHOLOGIES = 2
SEED = 42
