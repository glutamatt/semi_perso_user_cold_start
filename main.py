import os
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

