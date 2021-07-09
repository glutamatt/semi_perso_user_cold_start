config = {
    # user
    'embeddings_dim': 128,
    # cuda setting
    'use_cuda': True,
    # model setting
    'nb_epochs': 130,
    'learning_rate': 0.00001,
    'batch_size': 512,
    'reg_param': 0,
    'drop_out': 0,
    # model training
    'eval_every': 10,
}

dataset_eval = ["validation", "test"]
