import numpy as np

from params import params


def get_random_seed(dataset_name: str) -> int:
    """Retrieve the random seed from parameters
    that's been set for given dataset

    Args:
        dataset_name (str): name of the dataset

    Returns:
        int: the random seed value

    """
    return params[dataset_name]['random_seed']


def get_random_generator(dataset_name: str) -> np.random.Generator:
    """Creates NumPy random generator with random seed
    set that's been set in params for given dataset

    Args:
        dataset_name: the name of the dataset

    Returns:
        Generator: NumPy random generator object

    """
    random_seed = get_random_seed(dataset_name)
    return np.random.default_rng(random_seed)
