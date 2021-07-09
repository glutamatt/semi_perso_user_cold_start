import os
import torch
import pickle
import random

from model import RegressionTripleHidden
from options import config, states


def training(dataset_path, master_path, eval=True, model_save=True, model_filename=None):#XXXX change None
    if config['use_cuda']:
        cuda = torch.device(0)

    input_dim = 2579 #dataset.shape[1]
    target_dim = config['embeddings_dim']

    nb_epochs = config['nb_epochs']
    learning_rate = config['learning_rate']
    reg_param = config['reg_param']
    drop_out = config['drop_out']
    batch_size = config['batch_size']
    eval_every = config['eval_every']
    k_val = config['k_val']

    regression_model = RegressionTripleHidden(input_dim=input_dim, output_dim = target_dim).cuda(device = cuda)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(regression_model.parameters(), lr = learning_rate, weight_decay=reg_param )

    if not os.path.exists(model_filename):

        # Load training dataset.
        training_set_size = int(len(os.listdir("{}/train".format(master_path))) / 2)
        train_xs = []
        train_ys = []
        for idx in range(training_set_size):
            train_xs.append(pickle.load(open("{}/train/x_train_{}.pkl".format(master_path, idx), "rb")))
            train_ys.append(pickle.load(open("{}/train/y_train_{}.pkl".format(master_path, idx), "rb")))
        total_dataset = list(zip(train_xs, train_ys))
        del(train_xs, train_ys)

        if eval:

            # Load validation dataset.

            validation_set_size = int(len(os.listdir("{}/validation".format(master_path))) / 3)
            validation_xs = []
            listened_songs_validation_ys = []
            for idx in range(validation_set_size):
                validation_xs.append(pickle.load(open("{}/validation/x_validation_{}.pkl".format(master_path, idx), "rb")))
                listened_songs_validation_ys.append(pickle.load(open("{}/validation/y_listened_songs_validation_{}.pkl".format(master_path, idx), "rb")))
            total_validation_dataset = list(zip(validation_xs, listened_songs_validation_ys))
            del(validation_xs, listened_songs_validation_ys)

            # Load song embeddings for evaluation

            song_embeddings_path = dataset_path + "/song_embeddings.parquet"
            song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet').fillna(0)
            list_features = ["feature_"+str(i) for i in range(len(song_embeddings["features_svd"][0]))]
            song_embeddings[list_features] = pd.DataFrame(song_embeddings.features_svd.tolist(), index= song_embeddings.index)
            song_embeddings_values = song_embeddings[list_features].values
            song_embeddings_values_ = torch.FloatTensor(song_embeddings_values.astype(np.float32))

        training_set_size = len(total_dataset)
        print("training set size : "+str(training_set_size))
        print("validation set size : "+str(validation_set_size))
        print("regression model : "+ str(regression_model))
        print("training running")
        loss_train = []
        for nb in range(nb_epochs):
            print("nb epoch : "+str(nb))
            start_time_epoch = time.time()
            random.Random(nb).shuffle(total_dataset)
            a,b = zip(*total_dataset)
            num_batch = int(training_set_size / batch_size)
            max_loc = batch_size*num_batch
            current_loss = 0
            regression_model = regression_model.to(device = cuda)
            for i in range(num_batch):
                optimizer.zero_grad()
                batch_features_tensor = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                batch_target_tensor = torch.stack(b[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                output_tensor = regression_model(batch_features_tensor)
                loss = criterion(output_tensor, batch_target_tensor)
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())
            print('epoch ' + str(nb) +  " training loss : "+ str(sum(loss_train)/float(len(loss_train))))
            print("--- seconds ---" + str(time.time() - start_time_epoch))

            if nb != 0 and (nb % eval_every == 0 or nb == nb_epochs - 1):
                print('testing model')
                start_time_eval = time.time()
                reg = regression_model.eval()
                reg = reg.to(device=cuda)
                validation_set_size = len(total_validation_dataset)
                a,b = zip(*total_validation_dataset)
                num_batch_validation = int(validation_set_size / batch_size)
                current_recalls = []
                with torch.set_grad_enabled(False):
                    for i in range(num_batch_validation):
                        batch_features_tensor_validation = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                        predictions_validation = reg(batch_features_tensor_validation)
                        groundtruth_validation = list(b[batch_size*i:batch_size*(i+1)])
                        predictions_songs_validation = torch.mm(predictions_validation.cpu(), song_embeddings_values_.transpose(0, 1))
                        recommendations_validation = (predictions_songs_validation.topk(k= k_val, dim = 1)[1]).tolist()
                        recalls = list(map(lambda x, y: len(set(x) & set(y))/float(min(len(x), 50)), groundtruth_validation, recommendations_validation))
                        current_recalls.extend(recalls)
                print('epoch ' + str(nb) +  " recall test : "+ str(sum(current_recalls) / float(len(current_recalls))) )
                print("--- %s seconds ---" + str(time.time() - start_time_eval))

        if model_save:
            torch.save(regression_model.state_dict(), master_path + "/"+model_filename+".pt")

    else:
        trained_state_dict = torch.load(master_path + "/"+model_filename+".pt")
        RegressionTripleHidden.load_state_dict(trained_state_dict)
