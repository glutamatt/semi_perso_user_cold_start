import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn
import time
import statistics
import pickle
import random
from sklearn.metrics import ndcg_score, dcg_score
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from data_generation import generate

if __name__ == "__main__":
    master_path= "./deezer"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about XXGB of your hard disk space.
        generate(master_path)

