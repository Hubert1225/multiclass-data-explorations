import os
import sys

import numpy as np
from sklearn.datasets import load_wine

DATASET = sys.argv[1]
RAW_DATA_DIR = os.path.join('data', 'raw')


if DATASET == 'wine':
    X, y = load_wine(return_X_y=True, as_frame=True)
    X.to_csv(os.path.join(RAW_DATA_DIR, 'wine_X.csv'), index=False)
    np.savetxt(os.path.join(RAW_DATA_DIR, 'wine_y.csv'), y, encoding='utf-8')
