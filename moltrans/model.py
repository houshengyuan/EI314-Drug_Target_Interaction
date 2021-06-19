import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Layer, BatchNormalization, Dense, Dropout, Embedding, Conv2D, ReLU, Softmax
from tensorflow import keras
import math
import copy


class NormLayer(keras.Model):
    def __init__(self, hidden_size):
        super(NormLayer, self).__init__(name='NormLayer')
        self.hidden_size = hidden_size
        self.dense = Dense(hidden_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.bn(x)
        return x


class Embeddings(keras.Model):
    def __init__(self, voc_size, hidden_size, max_pos_size, dropout_rate):
        super(Embeddings, self).__init__(name='Embeddings')
        self.word_emb = tf.keras.layers.Embedding(voc_size, hidden_size,embeddings_initializer='uniform')
        self.pos_emb = tf.keras.layers.Embedding(max_pos_size, hidden_size,embeddings_initializer='uniform')
        self.NormLayer = NormLayer(hidden_size)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        #print(inputs.shape)
        seq_len = inputs.shape[1]
        #inputs=tf.reshape(inputs,(-1,16))
        #print(seq_len)
        #exit(1)
        pos_idx = tf.range(seq_len)
        pos_idx=tf.expand_dims(pos_idx,axis=0)
        #print(pos_idx.shape)
        pos_idx=tf.reshape(pos_idx,(-1,seq_len))
        #print(pos_idx.shape)

        word_embedding = self.word_emb(pos_idx)
        pos_embedding = self.pos_emb(inputs)

        embedding = word_embedding + pos_embedding
        x = self.NormLayer(embedding)
        x = self.dropout(x)
        return x


class SelfAttention(keras.Model):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(SelfAttention, self).__init__(name='selfattention')
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Hidden size {} should be a multiple of attention number {}.".format(hidden_size, num_heads))
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = int(hidden_size / num_heads)
        self.total_head_size = self.num_heads * self.head_size

        self.query = Dense(self.total_head_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.key = Dense(self.total_head_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.value = Dense(self.total_head_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.dropout = Dropout(attention_dropout_rate)

    def to_scores(self, x):
        #print(x.shape)
        #rint(x.shape[-1])
        #new_x_shape = x.shape[-1] + (self.num_heads, self.head_size)
        #print(new_x_shape)
        #exit(1)
        x=tf.reshape(x,(-1,x.shape[1],self.num_heads,self.head_size))
        #print(x.shape)
        #x = x.view(*new_x_shape)
        x=tf.transpose(x,perm=(0,2,1,3))
        return x

    def call(self, hidden_states, training=None, mask=None):
        query_layer = self.query(hidden_states)
        #print(query_layer.shape)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.to_scores(query_layer)
        #print(query_layer.shape)  # (batch_size * num_attention_heads * seq_length * attention_head_size)
        key_layer = self.to_scores(key_layer)
        value_layer = self.to_scores(value_layer)
        #print(key_layer.shape)
        #print(query_layer.shape)

        scores = tf.matmul(query_layer, tf.transpose(key_layer,perm=(0,1,3,2)))
        # batch_size * num_attention_heads * seq_length * seq_length
        scores = scores / math.sqrt(self.head_size)

        scores = scores + mask

        attention_p = Softmax(axis=-1)(scores)

        attention_p = self.dropout(attention_p)  # ???

        #print(value_layer.shape)
        #print(attention_p.shape)
        # shape of value_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        # shape of first context_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        # shape of second context_layer: batch_size * seq_length * num_attention_heads * attention_head_size
        # context_layer 维度恢复到：batch_size * seq_length * hidden_size
        context_layer = tf.matmul(attention_p, value_layer)
        context_layer = tf.transpose(context_layer,perm=(0, 2, 1, 3))#.contiguous()
        #print(context_layer.shape)
        context_layer=tf.reshape(context_layer,(-1,context_layer.shape[1],self.total_head_size))
        #new_context_layer_shape = context_layer.size()[:-2] + (self.total_head_size,)
        #context_layer = context_layer.view(*new_context_layer_shape)
        #print(context_layer.shape)
        return context_layer


class AttentionOutput(keras.Model):
    def __init__(self, hidden_size, output_dropout_rate):
        super(AttentionOutput, self).__init__(name='output')
        self.hidden_size = hidden_size
        self.dropout_rate = output_dropout_rate
        self.dense = Dense(hidden_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.normlayer = NormLayer(hidden_size)
        self.dropout = Dropout(output_dropout_rate)

    def call(self, hidden_states, inputs, training=None, mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        out = self.normlayer(hidden_states + inputs)
        return out


class Attention(keras.Model):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate, output_dropout_rate):
        super(Attention, self).__init__(name='attention')
        self.attention = SelfAttention(hidden_size, num_heads, attention_dropout_rate)
        self.out_attention = AttentionOutput(hidden_size, output_dropout_rate)

    def call(self, inputs, training=None, mask=None):
        out = self.attention(inputs, mask=mask)
        attention_output = self.out_attention(out, inputs)
        return attention_output


class Encoder(keras.Model):
    def __init__(self, hidden_size, intermediate_size, num_heads, attention_dropout_rate, output_dropout_rate):
        super(Encoder, self).__init__(name='encoder')
        self.attention = Attention(hidden_size, num_heads, attention_dropout_rate, output_dropout_rate)
        self.inter_dense = Dense(intermediate_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.inter_relu = ReLU()

        self.out_dense = Dense(hidden_size,kernel_initializer='random_uniform',bias_initializer='zeros')
        self.normlayer = NormLayer(hidden_size)
        self.out_dropout = Dropout(output_dropout_rate)

    def call(self, inputs, training=None, mask=None):
        attention_output = self.attention(inputs, mask=mask)

        inter_output = self.inter_dense(attention_output)
        inter_output = self.inter_relu(inter_output)

        final_output = self.out_dense(inter_output)
        final_output = self.out_dropout(final_output)
        final_output = self.normlayer(final_output + attention_output)

        return final_output

class MultiEncoder(keras.Model):
    def __init__(self,n_layer, hidden_size, intermediate_size, num_heads, attention_dropout_rate, output_dropout_rate):
        super(MultiEncoder,self).__init__(name='multi_encoder')
        self.one_encoder=Encoder(hidden_size,intermediate_size,num_heads,attention_dropout_rate,output_dropout_rate)
        #self.encoders=[copy.deepcopy(one_encoder) for _ in range(n_layer)]

    def call(self, inputs, training=None, mask=None):
        #for one in self.encoders:
            #inputs=one(inputs,mask=mask)
        out=self.one_encoder(inputs,mask=mask)
        return out


class MolTrans(keras.Model):
    def __init__(self,**config):
        super(MolTrans, self).__init__(name='moltrans')
        self.max_drug_seq=config['max_drug_seq']
        self.max_protein_seq=config['max_protein_seq']
        self.dropout_rate=config['dropout_rate']
        self.embed_size=config['embed_size']

        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernel_dense_size = config['kernel_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_d = config['input_dim_d']
        self.input_dim_p = config['input_dim_p']
        self.gpus = 1
        self.n_layer = 2

        self.hidden_size = config['embed_size']
        self.inter_size = config['inter_size']
        self.num_heads = config['num_heads']
        self.attention_dropout_rate = config['attention_dropout_rate']
        self.output_dropout_rate = config['output_dropout_rate']

        self.flat_dim=config['flat_dim']

        self.emb_drug=Embeddings(self.input_dim_d,self.embed_size,self.max_drug_seq,self.dropout_rate)
        self.emb_protein=Embeddings(self.input_dim_p,self.embed_size,self.max_protein_seq,self.dropout_rate)

        self.drug_encoder=MultiEncoder(self.n_layer,self.hidden_size,self.inter_size,self.num_heads,self.attention_dropout_rate,self.output_dropout_rate)
        self.protein_encoder=MultiEncoder(self.n_layer,self.hidden_size,self.inter_size,self.num_heads,self.attention_dropout_rate,self.output_dropout_rate)
        self.cnn_layer=Conv2D(3,3,padding='valid',data_format='channels_first')

        self.decoder=keras.Sequential()
        self.decoder.add(Dense(512,activation='relu',input_shape=(self.flat_dim,),kernel_initializer='random_uniform',bias_initializer='zeros'),)
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dense(64,activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dense(32,activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        self.decoder.add(Dense(1,kernel_initializer='random_uniform',bias_initializer='zeros'))

    def call(self, d,p,d_mask,p_mask):
        ex_d_mask = tf.expand_dims(d_mask,axis=1)
        ex_d_mask=tf.expand_dims(ex_d_mask,axis=2)
        ex_p_mask = tf.expand_dims(p_mask,axis=1)
        ex_p_mask=tf.expand_dims(ex_p_mask,axis=2)

        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        drug_emb=self.emb_drug(d)
        protein_emb=self.emb_protein(p)

        drug_encoder=self.drug_encoder(tf.cast(drug_emb,tf.float32),mask=tf.cast(ex_d_mask,tf.float32))
        protein_encoder = self.protein_encoder(tf.cast(protein_emb,tf.float32), mask=tf.cast(ex_p_mask,tf.float32))


        drug_encoder=tf.expand_dims(drug_encoder,2)
        drug_aug = tf.tile(drug_encoder,[1,1,self.max_protein_seq,1])  # repeat along protein size
        protein_encoder=tf.expand_dims(protein_encoder,1)
        protein_aug = tf.tile(protein_encoder, [1,self.max_drug_seq, 1, 1])  # repeat along drug size

        interaction=drug_aug*protein_aug
        interaction=tf.reshape(interaction,(-1,self.embed_size,self.max_drug_seq,self.max_protein_seq))

        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        interaction=tf.reduce_sum(interaction,axis=1)
        interaction=tf.expand_dims(interaction,axis=1)

        interaction=Dropout(self.dropout_rate)(interaction)
        out=self.cnn_layer(interaction)
        out=tf.reshape(out,(-1,self.flat_dim))

        out=self.decoder(out)

        return out



