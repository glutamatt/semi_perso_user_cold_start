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
from model_training import training

if __name__ == "__main__":
    master_path= "./deezer"
    dataset_path = os.getcwd() + "/data"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about XXGB of your hard disk space.
        generate(dataset_path, master_path)

    # training model.
    training(dataset_path, master_path, eval=True, model_save=True, model_filename="20210709_svd_sample")

