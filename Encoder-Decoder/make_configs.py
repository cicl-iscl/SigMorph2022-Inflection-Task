import os
from datetime import date


def fill_template(lang: str):
    train_source_file = os.path.join("opennmt_data", lang, "train.src")
    with open(train_source_file) as tf:
        num_train_words = len(tf.readlines())
    
    batch_size = 8
    epochs = 48
    num_train_steps = (num_train_words * epochs) // batch_size
    num_valid_steps = num_train_words // batch_size
    
    
    return f"""
## Where the samples will be written
save_data: openmt_models/{lang}/{str(date.today())}
## Where the vocab(s) will be written
src_vocab: openmt_models/{lang}/example.vocab.src
tgt_vocab: openmt_models/{lang}/example.vocab.tgt
# Prevent overwriting existing files in the folder
overwrite: true

data:
    corpus_1:
        path_src: openmt_data/{lang}/train.src
        path_tgt: openmt_data/{lang}/train.trg
    valid:
        path_src: openmt_data/{lang}/dev.src
        path_tgt: openmt_data/{lang}/dev.trg
save_model: openmt_models/{lang}/{lang}_model
save_checkpoint_steps: {num_train_words}
keep_checkpoint: 1
seed: 3435
train_steps: {num_train_steps}
valid_steps: {num_valid_steps}
report_every: 200

early_stopping: 2
early_stopping_criteria: accuracy

position_encoding: false

encoder_type: brnn
decoder_type: rnn
word_vec_size: 32
rnn_size: 128
layers: 2

optim: adam
learning_rate: 0.001
max_grad_norm: 5

batch_size:  {batch_size}
dropout: 0.0

copy_attn: 'true'
global_attention: mlp
reuse_copy_attn: 'true'
bridge: 'true'

world_size: 1
gpu_ranks:
- 0
"""


if __name__ == "__main__":
    langs = os.listdir("opennmt_data")
    for lang in langs:
        config = fill_template(lang)
        with open(os.path.join("config", lang + ".yaml"), "w") as cf:
            cf.write(config)
