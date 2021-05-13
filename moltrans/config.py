def CONFIG():
    config = {}
    config['batch_size'] = 16
    config['input_dim_d'] = 23532
    config['input_dim_p'] = 16693
    config['train_epoch'] = 13
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 545
    config['embed_size'] = 384
    config['dropout_rate'] = 0.1

    # DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernel_dense_size'] = 3

    # Encoder
    config['inter_size'] = 1536
    config['num_heads'] = 12
    config['attention_dropout_rate'] = 0.1
    config['output_dropout_rate'] = 0.1
    config['flat_dim'] = 78192
    return config
