import os
import pandas as pd
import torch
import torch.nn
import pickle
from options import dataset_eval

def generate(dataset_path, master_path):

    #songs

    song_embeddings_path = dataset_path + "/song_embeddings.parquet"
    song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet').fillna(0)

    if not os.path.exists(master_path+"/m_song_dict.pkl"):
        song_dict = {}
        for idx, row in song_embeddings.iterrows():
            song_dict[row['song_index']] = idx
        pickle.dump(song_dict, open("{}/m_song_dict.pkl".format(master_path), "wb"))
    else:
        song_dict = pickle.load(open("{}/m_song_dict.pkl".format(master_path), "rb"))


    # user embeddings (target = only for train users)

    user_embeddings = pd.read_parquet(dataset_path + "/user_embeddings.parquet", engine = 'fastparquet')
    list_embeddings = ["embedding_"+str(i) for i in range(len(user_embeddings["svd_embeddings"][0]))]
    user_embeddings[list_embeddings] = pd.DataFrame(user_embeddings.svd_embeddings.tolist(), index= user_embeddings.index)
    embeddings_train = user_embeddings[list_embeddings].values

    # user features train

    features_train_path = dataset_path + "/user_features_train.parquet"
    features_train = pd.read_parquet(features_train_path, engine = 'fastparquet').fillna(0)
    features_train = features_train.sort_values("user_index")
    features_train = features_train.reset_index(drop=True)#to check it is ok for train data

    # training dataset creation

    dataset = "train"
    if not os.path.exists(master_path+"/"):
        os.mkdir(master_path+"/")
    if not os.path.exists(master_path+"/"+dataset+"/"):
        os.mkdir(master_path+"/"+dataset+"/")
    for idx in range(len(features_train)):
        x_train = torch.FloatTensor(features_train.iloc[idx,2:])
        y_train = torch.FloatTensor(user_embeddings[list_embeddings].iloc[idx,:])
        pickle.dump(x_train, open("{}/{}/x_train_{}.pkl".format(master_path, dataset, idx), "wb"))
        pickle.dump(y_train, open("{}/{}/y_train_{}.pkl".format(master_path, dataset, idx), "wb"))

    # user features validation & test

    for dataset in dataset_eval :
        features_validation_path = dataset_path + "/user_features_" + dataset + ".parquet"
        features_validation = pd.read_parquet(features_validation_path, engine = 'fastparquet').fillna(0)
        features_validation = features_validation.sort_values("user_index")
        features_validation = features_validation.reset_index(drop=True)

        if not os.path.exists(master_path+"/"+dataset+"/"):
            os.mkdir(master_path+"/"+dataset+"/"+"/")
        for i in range(len(features_validation)):
            x_validation = torch.FloatTensor(features_validation.iloc[i,2:])
            y_validation = [song_dict[song_index]  for song_index in features_validation["d1d30_songs"][i]]
            groundtruth_validation_list = [1.0 * (song in y_validation) for song in range(len(song_embeddings))]
            pickle.dump(x_validation, open("{}/{}/x_validation_{}.pkl".format(master_path, dataset, i), "wb"))
            pickle.dump(y_validation, open("{}/{}/y_listened_songs_validation_{}.pkl".format(master_path, dataset, i), "wb"))
            pickle.dump(groundtruth_validation_list, open("{}/{}/groundtruth_list_{}.pkl".format(master_path, dataset, i), "wb"))
