import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import PROCESSED_DATA_DIR
from params import params
from randomness import get_random_seed

DATASET = sys.argv[1]
ALL_PATH = os.path.join(PROCESSED_DATA_DIR, DATASET + '.csv')
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, DATASET + '_train.csv')
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, DATASET + '_test.csv')
SEED = get_random_seed(DATASET)
PARAMS = params[DATASET]

data_all = pd.read_csv(ALL_PATH)
data_train, data_test = train_test_split(
    data_all,
    test_size=PARAMS["test_frac"],
    random_state=SEED,
    stratify=(data_all['class_'] if PARAMS['stratify'] else None)
)
data_train.to_csv(TRAIN_PATH, index=False)
data_test.to_csv(TEST_PATH, index=False)
