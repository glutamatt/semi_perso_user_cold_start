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

def generate(master_path):
    dataset_path = os.getcwd() + "/data"

    #songs

    song_embeddings_path = dataset_path + "/song_embeddings_sample.parquet"
    song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet').fillna(0)

    if not os.path.exists(master_path+"/m_song_dict.pkl"):
        song_dict = {}
        for idx, row in song_embeddings.iterrows():
            song_dict[row['song_index']] = idx
        pickle.dump(song_dict, open("{}/m_song_dict.pkl".format(master_path), "wb"))
    else:
        song_dict = pickle.load(open("{}/m_song_dict.pkl".format(master_path), "rb"))

    # user embeddings (target = only for train users)

    user_embeddings = pd.read_parquet(dataset_path + "/user_embeddings_sample.parquet", engine = 'fastparquet')
    list_embeddings = ["embedding_"+str(i) for i in range(len(user_embeddings["svd_embeddings"][0]))]
    user_embeddings[list_embeddings] = pd.DataFrame(user_embeddings.mf_embeddings.tolist(), index= user_embeddings.index)
    embeddings_train = user_embeddings[list_embeddings].values

    # user features train

    features_train_path = dataset_path + "/user_features_train_sample.parquet"
    features_train = pd.read_parquet(features_train_path, engine = 'fastparquet').fillna(0)
    features_train = features_train.sort_values("user_index")
    features_train_ = features_train.values[:,2:]

    # training dataset creation

    state = "train"
    if not os.path.exists(master_path+"/"):
        os.mkdir(master_path+"/")
    if not os.path.exists(master_path+"/"+state+"/"):
        os.mkdir(master_path+"/"+state+"/")
    for idx in range(len(features_train)):
        x_train = torch.FloatTensor(features_train.iloc[idx,2:])
        y_train = torch.FloatTensor(user_embeddings[list_embeddings].iloc[idx,:])
        pickle.dump(x_train, open("{}/{}/x_train_{}.pkl".format(master_path, state, idx), "wb"))
        pickle.dump(y_train, open("{}/{}/y_train_{}.pkl".format(master_path, state, idx), "wb"))


''''
# songs


# user embeddings + features of training data

user_features_path = data_user_path + "/userFeatures/part-00000-0208489c-291e-4279-8c98-6a0dccb49150-c000.snappy.parquet"
user_features = pd.read_parquet(user_features_path, engine = 'fastparquet').fillna(0)
user_features = user_features[["user_index", "svd_features.values", "mf_features.values"]]
user_features.columns = ["user_index", "svd_features", "mf_features"]

list_embeddings = ["embedding_"+str(i) for i in range(len(user_features["mf_features"][0]))]
user_features[list_embeddings] = pd.DataFrame(user_features.mf_features.tolist(), index= user_features.index)
embeddings_train = user_features[list_embeddings].values

-
features_train_path = data_path + "/tableTrainMF/part-00000-f9d99b97-621a-434b-8c9d-92be08c37e87-c000.snappy.parquet"
features_train = pd.read_parquet(features_train_path, engine = 'fastparquet').fillna(0)
features_train = features_train.sort_values("user_index")
features_train_ = features_train.values[:,2:]
-

idx = 0
user_list = dict()
state = "train"
if not os.path.exists(master_path+"/"):
    os.mkdir(master_path+"/")
if not os.path.exists(master_path+"/"+state+"/"):
    os.mkdir(master_path+"/"+state+"/")
for idx in range(len(features_train)):
    x_train = torch.FloatTensor(features_train.iloc[idx,2:])
    y_train = torch.FloatTensor(user_features[list_embeddings].iloc[idx,:])
    pickle.dump(x_train, open("{}/{}/x_train_{}.pkl".format(master_path, state, idx), "wb"))
    pickle.dump(y_train, open("{}/{}/y_train_{}.pkl".format(master_path, state, idx), "wb"))
    idx += 1


# validation data

features_validation_path = data_path + "/tableValidationMF/part-00000-50e8c107-cf8f-4813-919b-70a5b364b827-c000.snappy.parquet"
features_validation = pd.read_parquet(features_validation_path, engine = 'fastparquet').fillna(0)
features_validation = features_validation.sort_values("user_index")
features_validation = features_validation.reset_index(drop=True)

idx = 70000
user_list = dict()
state = "validation"
if not os.path.exists(master_path+"/"+state+"/"):
    os.mkdir(master_path+"/"+state+"/"+"/")
for i in range(len(features_validation)):
    if idx % 1000 == 0:
        print(idx)
    x_validation = torch.FloatTensor(features_validation.iloc[i,2:])
    y_validation = [song_dict[song_index]  for song_index in features_validation["d1d30_songs"][i]]

    pickle.dump(x_validation, open("{}/{}/x_validation_{}.pkl".format(master_path, state, idx), "wb"))
    pickle.dump(y_validation, open("{}/{}/y_listened_songs_validation_{}.pkl".format(master_path, state, idx), "wb"))

    idx += 1

# groundtruth of validation data : songs listened after d0 between D1 and D30

idx = 70000
user_list = dict()
state = "validation"
if not os.path.exists(master_path+"/"+state+"/"):
    os.mkdir(master_path+"/"+state+"/"+"/")
for i in range(len(features_validation)):
    if idx % 1000 == 0:
        print(idx)
    y_validation = [song_dict[song_id]
                    for song_id in features_validation["d1d30_songs"][i]]
    groundtruth_validation_list = [1.0*
                                   (song in y_validation)
                                   for song in range(len(song_embeddings))]
    pickle.dump(groundtruth_validation_list, open("{}/{}/groundtruth_list_{}.pkl".format(master_path, state, idx), "wb"))

    idx += 1


# test data

features_test_path = data_path + "/tableTestMF/part-00000-3a278884-d54f-49e2-ad85-642d77e30677-c000.snappy.parquet"
features_test = pd.read_parquet(features_test_path, engine = 'fastparquet').fillna(0)
features_test = features_test.sort_values("user_index")
features_test = features_test.reset_index(drop=True)

idx = 90000
user_list = dict()
state = "test"
if not os.path.exists(master_path+"/"+state+"/"):
    os.mkdir(master_path+"/"+state+"/"+"/")
for i in range(len(features_test)):
    if idx % 1000 == 0:
        print(idx)
    x_test = torch.FloatTensor(features_test.iloc[i,2:])
    y_test = [song_dict[song_index]  for song_index in features_test["d1d30_songs"][i]]

    pickle.dump(x_test, open("{}/{}/x_test_{}.pkl".format(master_path, state, idx), "wb"))
    pickle.dump(y_test, open("{}/{}/y_listened_songs_test_{}.pkl".format(master_path, state, idx), "wb"))

    idx += 1

# groundtruth of test data : songs listened after d0 between D1 and D30

idx = 90000
user_list = dict()
state = "test"
if not os.path.exists(master_path+"/"+state+"/"):
    os.mkdir(master_path+"/"+state+"/"+"/")
for i in range(len(features_test)):#features_train.keys())):
    if idx % 1000 == 0:
        print(idx)
    y_test = [song_dict[song_id]
              for song_id in features_test["d1d30_songs"][i]]
    groundtruth_test_list = [1.0*
                             (song in y_test)
                             for song in range(len(song_embeddings))]
    pickle.dump(groundtruth_test_list, open("{}/{}/groundtruth_list_{}.pkl".format(master_path, state, idx), "wb"))

    idx += 1
'''