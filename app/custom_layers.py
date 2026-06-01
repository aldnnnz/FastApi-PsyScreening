import tensorflow as tf
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer untuk model NLP.

    Input shape:
        (batch_size, sequence_length, hidden_dim)

    Output shape:
        (batch_size, hidden_dim)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(
            tf.matmul(inputs, self.W)
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = weights * inputs

        context = tf.reduce_sum(
            context,
            axis=1
        )

        return context

    def get_config(self):
        config = super().get_config()
        return config