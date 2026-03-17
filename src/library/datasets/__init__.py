from src.library.datasets.ihdp import IHDP
from src.library.datasets.simulation import Simulation

from src.library.datasets.active_learning import ActiveLearningDataset
from src.library.datasets.active_learning import RandomFixedLengthSampler

DATASETS = {
    "ihdp": IHDP,
    "simulation": Simulation,
}
