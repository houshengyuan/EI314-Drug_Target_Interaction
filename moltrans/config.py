import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size'
    )
    parser.add_argument(
        '--input_dim_d',
        type=int,
        default=23532,
        help='input dimd'
    )
    parser.add_argument(
        '--input_dim_p',
        type=int,
        default=16693,
        help='input dimp'
    )
    parser.add_argument(
        '--train_epoch',
        type=int,
        default=13,
        help='train epoch'
    )
    parser.add_argument(
        '--max_drug_seq',
        type=int,
        default=50,
        help='max drug seq'
    )
    parser.add_argument(
        '--max_protein_seq',
        type=int,
        default=545,
        help='max protein seq'
    )
    parser.add_argument(
        '--embed_size',
        type=int,
        default=384,
        help='embed size'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.1,
        help='dropout rate'
    )
    parser.add_argument(
        '--scale_down_ratio',
        type=float,
        default=0.25,
        help='scale down ratio'
    )
    parser.add_argument(
        '--growth_rate',
        type=int,
        default=20,
        help='growth rate'
    )
    parser.add_argument(
        '--transition_rate',
        type=float,
        default=0.5,
        help='transition rate'
    )
    parser.add_argument(
        '--num_dense_blocks',
        type=int,
        default=4,
        help='num dense blocks'
    )
    parser.add_argument(
        '--kernel_dense_size',
        type=int,
        default=3,
        help='kernel dense size'
    )
    parser.add_argument(
        '--inter size',
        type=int,
        default=1536,
        help='inter size'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=12,
        help='num heads'
    )
    parser.add_argument(
        '--attention_dropout_rate',
        type=float,
        default=0.1,
        help='attention dropout rate'
    )
    parser.add_argument(
        '--output_dropout_rate',
        type=float,
        default=0.1,
        help='output dropout rate'
    )
    parser.add_argument(
        '--flat_dim',
        type=int,
        default=78192,
        help='flat dim'
    )
    flags, unparsed = parser.parse_known_args()
    return flags

def CONFIG():
    flags=argparser()
    config = {}
    config['batch_size'] = flags.batch_size
    config['input_dim_d'] = flags.input_dim_d
    config['input_dim_p'] = flags.input_dim_p
    config['train_epoch'] = flags.train_epoch
    config['max_drug_seq'] = flags.max_drug_seq
    config['max_protein_seq'] = flags.max_protein_seq
    config['embed_size'] = flags.embed_size
    config['dropout_rate'] = flags.dropout_rate

    # DenseNet
    config['scale_down_ratio'] = flags.scale_down_ratio
    config['growth_rate'] = flags.growth_rate
    config['transition_rate'] = flags.transition_rate
    config['num_dense_blocks'] = flags.num_dense_blocks
    config['kernel_dense_size'] = flags.kernel_dense_size

    # Encoder
    config['inter_size'] = flags.inter_size
    config['num_heads'] = flags.num_heads
    config['attention_dropout_rate'] = flags.attention_dropout_rate
    config['output_dropout_rate'] = flags.output_dropout_rate
    config['flat_dim'] = flags.flat_dim
    return config
