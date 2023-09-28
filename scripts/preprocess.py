import os
import sys

import numpy as np
import pandas as pd

DATASET = sys.argv[1]
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
RESULT_PATH = os.path.join(PROCESSED_DATA_DIR, DATASET + '.csv')


if DATASET == 'wine':
    X_path = os.path.join(RAW_DATA_DIR, DATASET + '_X.csv')
    y_path = os.path.join(RAW_DATA_DIR, DATASET + '_y.csv')
    X = pd.read_csv(X_path)
    y = np.loadtxt(y_path, encoding='utf-8').astype('int')
    X['class_'] = y
    X.to_csv(RESULT_PATH, index=False)
