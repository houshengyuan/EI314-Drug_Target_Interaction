#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import collections
import copy
import math
import numpy as np
import re
import tensorflow as tf

params={
"hidden_size":768,
"num_hidden_layers":12,
"num_attention_heads":12,
"intermediate_size":3072,
"hidden_act":"gelu",
"hidden_dropout_prob":0.1,
"attention_probs_dropout_prob":0.1,
"max_position_embeddings":512,
"type_vocab_size":16
}
vocab_size=None


class BertModel(object):
    def __init__(self,config,is_training,input_ids,input_mask=None,token_type_ids=None,scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        input_shape = get_shape_list(input_ids)
        batch_size,seq_length= input_shape[0]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    word_embedding_name="word_embeddings")
                # Add positional embeddings and token type embeddings, then layer normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)
            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
                # Run the stacked transformer sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    do_return_all_layers=True)
            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(first_token_tensor,
                    config.hidden_size,activation=tf.tanh,
                    kernel_initializer=create_initializer(0.02))


class DTIBertModel(object):
    def __init__(self,config,is_training,input_ids,input_mask=None,scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        input_shape = get_shape_list(input_ids)#, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    word_embedding_name="word_embeddings")
                # Add positional embeddings and token type embeddings, then layer normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=False,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)
            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)
                # Run the stacked transformer. `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    do_return_all_layers=True)
            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(first_token_tensor,config.hidden_size,
                    activation=tf.tanh,kernel_initializer=create_initializer(0.02))


def gelu(x):
   cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
   return x * cdf


def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,vocab_size,embedding_size=128,
                     word_embedding_name="word_embeddings"):
    """Looks up words embeddings for id tensor.
    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      word_embedding_name: string. Name of the embedding table.
    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,num_inputs].
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])
    embedding_table = tf.get_variable(name=word_embedding_name,shape=[vocab_size, embedding_size],
        initializer=create_initializer(0.02))
    flat_input_ids = tf.reshape(input_ids, [-1])
    output = tf.gather(embedding_table, flat_input_ids)
    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output,input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,use_token_type=False,
                            token_type_ids=None,token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable for positional embeddings.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.
    Returns:
      float tensor with same shape as `input_tensor`.
    """
    batch_size, seq_length, width = get_shape_list(input_tensor)#, expected_rank=3)
    output = input_tensor
    if use_token_type:
        token_type_table = tf.get_variable(name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(0.02))
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,[batch_size, seq_length, width])
        output += token_type_embeddings
    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(0.02))
            # So full_position_embeddings is effectively an embedding table
            # for position [0, 1, 2, ...,max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],[seq_length, -1])
            num_dims = len(output.shape.as_list())
            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,position_broadcast_shape)
            output += position_embeddings
    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor)#, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_shape = get_shape_list(to_mask)#, expected_rank=2)
    to_seq_length = to_shape[1]
    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    return mask


def attention_layer(from_tensor,to_tensor,attention_mask=None,
                    num_attention_heads=1,size_per_head=512,query_act=None,
                    key_act=None,value_act=None,attention_probs_dropout_prob=0.0,do_return_2d_tensor=False,
                    batch_size=None,from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
    If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.
    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the attention probabilities.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.
    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).
    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor
    from_shape = get_shape_list(from_tensor)#, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor)#, expected_rank=[2, 3])
    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
    if len(from_shape) == 3:
        batch_size,from_seq_length,to_seq_length = from_shape
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")
    # Scalar dimensions referenced here:
    #   B = batch size
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    query_layer = tf.layers.dense(from_tensor_2d,num_attention_heads * size_per_head,
        activation=query_act,name="query",
        kernel_initializer=create_initializer(0.02))
    key_layer = tf.layers.dense(to_tensor_2d,num_attention_heads * size_per_head,
        activation=key_act,name="key",
        kernel_initializer=create_initializer(0.02))
    value_layer = tf.layers.dense(to_tensor_2d,num_attention_heads * size_per_head,
        activation=value_act,name="value",
        kernel_initializer=create_initializer(0.02))
    query_layer = transpose_for_scores(query_layer, batch_size,num_attention_heads, from_seq_length,size_per_head)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,to_seq_length, size_per_head)
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,1.0 / math.sqrt(float(size_per_head)))
    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder
    # Normalize the attention scores to probabilities.
    # attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    value_layer = tf.reshape(value_layer,[batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    if do_return_2d_tensor:
        context_layer = tf.reshape(context_layer,[batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(context_layer,[batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer


def transformer_model(input_tensor,attention_mask=None,
                      hidden_size=768,num_hidden_layers=12,
                      num_attention_heads=12,intermediate_size=3072,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,do_return_all_layers=False):
    """
    This is almost an exact implementation of the original Transformer encoder.
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.?????
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feedforward) layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
      do_return_all_layers: Whether to also return all layers or just the final layer.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final hidden layer of the Transformer.
    """
    assert hidden_size % num_attention_heads == 0
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor)#, expected_rank=3)
    batch_size, seq_length, input_width = input_shape
    assert input_width == hidden_size
    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)
    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(from_tensor=layer_input,
                        to_tensor=layer_input,attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        do_return_2d_tensor=True,batch_size=batch_size,
                        from_seq_length=seq_length,to_seq_length=seq_length)
                    attention_heads.append(attention_head)
                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)
                # Run a linear projection of `hidden_size` then add a residual with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(attention_output,hidden_size,kernel_initializer=create_initializer(0.02))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)
            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(attention_output,
                    intermediate_size,activation=gelu,
                    kernel_initializer=create_initializer(0.02))
            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(intermediate_output,hidden_size,
                    kernel_initializer=create_initializer(0.02))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:tensor: A tf.Tensor object to find the shape of.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    assert ndims >= 2
    if ndims == 2:
        return input_tensor
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate,global_step,num_train_steps,
        end_learning_rate=0.0,power=1.0,cycle=False)
    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate,
        weight_decay_rate=0.01,beta_1=0.9,
        beta_2=0.999,epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    if use_tpu:
       optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    grads = tf.gradients(loss, tvars)
    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""
    def __init__(self,learning_rate,weight_decay_rate=0.0,beta_1=0.9,beta_2=0.999,
                 epsilon=1e-6,exclude_from_weight_decay=None,name="AdamWeightDecayOptimizer"):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param),m.assign(next_m),v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name