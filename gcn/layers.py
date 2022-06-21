from inits import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
initializer='glorot_normal'
params_dict = {
    'kernel_initializer': initializer,
    'kernel_regularizer': l2(0.001),
    'activation':'relu',
}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']#稀疏退出的助手变量

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        
        
        from tensorflow.keras.layers import Reshape

        # self.input_dim = self.inputs.get_shape().as_list()[1]
        # x= tf.reshape(output,[4,8])
        m=output.get_shape().as_list()[1]
        n=int(m/4)
        x=Reshape((4,n))(output)
        # x= tf.reshape(output,[4,8])

        d1= x.get_shape().as_list()
        d1=d1[2]

        output1=Self_Attention(d1)(x)
        # m=output.get_shape().as_list()[1]
        # X=Transformer(m,m)(output)
        # from tensorflow.keras.layers import GlobalAveragePooling1D
        # output1= GlobalAveragePooling1D()(X)
        # # output1=MultiHeadAttention(output.shape.as_list(),3)
        output1=tf.layers.Flatten()(output1)
        # print("**************")
        output=output+output1
        # # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape.as_list()[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim**0.5)

        QK = K.softmax(QK)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)

    # def get_config(self):
    #     config = {
    #         'output_dim': self.output_dim
    #     }
    #     base_config = super(Self_Attention, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))




# -*- coding: utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras.callbacks import Callback

# from keras import backend as K
# from keras.engine.topology import Layer
# from deep_recommenders.keras.models.nlp import MultiHeadAttention

# class Transformer(Layer):

#     def __init__(self, vocab_size, model_dim, 
#             n_heads=8, encoder_stack=6, decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1, **kwargs):
#         self._vocab_size = vocab_size+1
#         self._model_dim = model_dim
#         self._n_heads = n_heads
#         self._encoder_stack = encoder_stack
#         self._decoder_stack = decoder_stack
#         self._feed_forward_size = feed_forward_size
#         self._dropout_rate = dropout_rate
#         super(Transformer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.embeddings = self.add_weight(
#             shape=(self._vocab_size, self._model_dim),
#             initializer='glorot_uniform',
#             trainable=True,
#             name="embeddings")
#         self.EncoderPositionEncoding = PositionEncoding(self._model_dim)
#         self.EncoderMultiHeadAttentions = [
#             MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
#             for _ in range(self._encoder_stack)
#         ]
#         self.EncoderLayerNorms0 = [
#             LayerNormalization()
#             for _ in range(self._encoder_stack)
#         ]
#         self.EncoderPositionWiseFeedForwards = [
#             PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
#             for _ in range(self._encoder_stack)
#         ]
#         self.EncoderLayerNorms1 = [
#             LayerNormalization()
#             for _ in range(self._encoder_stack)
#         ]
#         self.DecoderPositionEncoding = PositionEncoding(self._model_dim)
#         self.DecoderMultiHeadAttentions0 = [
#             MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
#             for _ in range(self._decoder_stack)
#         ]
#         self.DecoderLayerNorms0 = [
#             LayerNormalization()
#             for _ in range(self._decoder_stack)
#         ]
#         self.DecoderMultiHeadAttentions1 = [
#             MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
#             for _ in range(self._decoder_stack)
#         ]
#         self.DecoderLayerNorms1 = [
#             LayerNormalization()
#             for _ in range(self._decoder_stack)
#         ]
#         self.DecoderPositionWiseFeedForwards = [
#             PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
#             for _ in range(self._decoder_stack)
#         ]
#         self.DecoderLayerNorms2 = [
#             LayerNormalization()
#             for _ in range(self._decoder_stack)
#         ]
#         super(Transformer, self).build(input_shape)


#     def encoder(self, inputs):
#         if K.dtype(inputs) != 'int32':
#             inputs = K.cast(inputs, 'int32')

#         masks = K.equal(inputs, 0)
#         # Embeddings
#         embeddings = K.gather(self.embeddings, inputs)
#         embeddings *= self._model_dim ** 0.5 # Scale
#         # Position Encodings
#         position_encodings = PositionEncoding(self._model_dim)(embeddings)
#         # Embedings + Postion-encodings
#         encodings = embeddings + position_encodings
#         # Dropout
#         encodings = K.dropout(encodings, self._dropout_rate)

#         for i in range(self._encoder_stack):
#             # Multi-head-Attention
#             attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
#             attention_input = [encodings, encodings, encodings, masks]
#             attention_out = attention(attention_input)
#             # Add & Norm
#             attention_out += encodings
#             attention_out = LayerNormalization()(attention_out)
#             # Feed-Forward
#             ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
#             ff_out = ff(attention_out)
#             # Add & Norm
#             ff_out += attention_out
#             encodings = LayerNormalization()(ff_out)

#         # return encodings, masks
#         return encodings

#     def call(self, encoder_inputs, **kwargs):
#         encoder_encodings= self.encoder(encoder_inputs)
#         # encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
#         return encoder_encodings

#     def compute_output_shape(self, input_shape):
#         # return  (input_shape[0][0], input_shape[0][1], self._vocab_size)
#         return  (input_shape[0], input_shape[1], self._vocab_size)
    
# class PositionWiseFeedForward(Layer):
    
#     def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
#         self._model_dim = model_dim
#         self._inner_dim = inner_dim
#         self._trainable = trainable
#         super(PositionWiseFeedForward, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.weights_inner = self.add_weight(
#             shape=(int(input_shape[-1]), self._inner_dim),
#             initializer='glorot_uniform',
#             trainable=self._trainable,
#             name="weights_inner")
#         self.weights_out = self.add_weight(
#             shape=(int(self._inner_dim), self._model_dim),
#             initializer='glorot_uniform',
#             trainable=self._trainable,
#             name="weights_out")
#         self.bais_inner = self.add_weight(
#             shape=(int(self._inner_dim),),
#             initializer='uniform',
#             trainable=self._trainable,
#             name="bais_inner")
#         self.bais_out = self.add_weight(
#             shape=(int(self._model_dim),),
#             initializer='uniform',
#             trainable=self._trainable,
#             name="bais_out")
#         super(PositionWiseFeedForward, self).build(input_shape)

#     def call(self, inputs):
#         if K.dtype(inputs) != 'float32':
#             inputs = K.cast(inputs, 'float32')
#         inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
#         outputs = K.dot(inner_out, self.weights_out) + self.bais_out
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return self._model_dim
    
# class LayerNormalization(Layer):

#     def __init__(self, epsilon=1e-8, **kwargs):
#         self._epsilon = epsilon
#         super(LayerNormalization, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.beta = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer='zero',
#             name='beta')
#         self.gamma = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer='one',
#             name='gamma')
#         super(LayerNormalization, self).build(input_shape)

#     def call(self, inputs):
#         mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
#         normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
#         outputs = self.gamma * normalized + self.beta
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return input_shape

# class PositionEncoding(Layer):


#     def __init__(self, model_dim, **kwargs):
#         self._model_dim = model_dim
#         super(PositionEncoding, self).__init__(**kwargs)

#     def call(self, inputs, **kwargs):
#         seq_length = inputs.shape[1]
#         position_encodings = np.zeros((seq_length, self._model_dim))
#         for pos in range(seq_length):
#             for i in range(self._model_dim):
#                 position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
#         position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
#         position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
#         position_encodings = K.cast(position_encodings, 'float32')
#         return position_encodings

#     def compute_output_shape(self, input_shape):
#         return input_shape

# class MultiHeadAttention(Layer):

#     def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
#         self._n_heads = n_heads
#         self._head_dim = head_dim
#         self._dropout_rate = dropout_rate
#         self._masking = masking
#         self._future = future
#         self._trainable = trainable
#         super(MultiHeadAttention, self).__init__(**kwargs)

#     def build(self, input_shape):
    
#         self._weights_queries = self.add_weight(
#             shape=(int(input_shape[0][-1]), self._n_heads * self._head_dim),
#             initializer='glorot_uniform',
#             trainable=self._trainable,
#             name='weights_queries')
#         self._weights_keys = self.add_weight(
#             shape=(int(input_shape[1][-1]), self._n_heads * self._head_dim),
#             initializer='glorot_uniform',
#             trainable=self._trainable,
#             name='weights_keys')
#         self._weights_values = self.add_weight(
#             shape=(int(input_shape[2][-1]), self._n_heads * self._head_dim),
#             initializer='glorot_uniform',
#             trainable=self._trainable,
#             name='weights_values')
#         super(MultiHeadAttention, self).build(input_shape)


#     def call(self, inputs):
#         if self._masking:
#             assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
#             queries, keys, values, masks = inputs
#         else:
#             assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
#             queries, keys, values = inputs
        
#         queries_linear = K.dot(queries, self._weights_queries) 
#         keys_linear = K.dot(keys, self._weights_keys)
#         values_linear = K.dot(values, self._weights_values)

#         queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
#         keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
#         values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)
        
#         if self._masking:
#             att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
#         else:
#             att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
            
#         attention = ScaledDotProductAttention(
#             masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
#         att_out = attention(att_inputs)

#         outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return input_shape

# class ScaledDotProductAttention(Layer):

#     def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
#         self._masking = masking
#         self._future = future
#         self._dropout_rate = dropout_rate
#         self._masking_num = -2**32+1
#         super(ScaledDotProductAttention, self).__init__(**kwargs)

#     def mask(self, inputs, masks):
#         masks = K.cast(masks, 'float32')
#         masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
#         masks = K.expand_dims(masks, 1)
#         outputs = inputs + masks * self._masking_num
#         return outputs
    
#     def future_mask(self, inputs):
#         diag_vals = tf.ones_like(inputs[0, :, :])
#         tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  
#         future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
#         paddings = tf.ones_like(future_masks) * self._masking_num
#         outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
#         return outputs

#     def call(self, inputs):
#         if self._masking:
#             assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
#             queries, keys, values, masks = inputs
#         else:
#             assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
#             queries, keys, values = inputs

#         if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
#         if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
#         if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

#         matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
#         scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
#         if self._masking:
#             scaled_matmul = self.mask(scaled_matmul, masks) # Mask(opt.)

#         if self._future:
#             scaled_matmul = self.future_mask(scaled_matmul)

#         softmax_out = K.softmax(scaled_matmul) # SoftMax
#         # Dropout
#         out = K.dropout(softmax_out, self._dropout_rate)
        
#         outputs = K.batch_dot(out, values)

#         return outputs

#     def compute_output_shape(self, input_shape):
#         return input_shape

# class Add(Layer):

#     def __init__(self, **kwargs):
#         super(Add, self).__init__(**kwargs)

#     def call(self, inputs):
#         input_a, input_b = inputs
#         return input_a + input_b

#     def compute_output_shape(self, input_shape):
#         return input_shape[0]

