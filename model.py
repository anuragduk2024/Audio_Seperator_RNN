import tensorflow as tf 
import numpy as np 
import os
import shutil
from datetime import datetime

class SVSRNN(tf.keras.Model):
    def __init__(self, num_features, num_rnn_layer=4, num_hidden_units=[256, 256, 256, 256], tensorboard_directory='graphs/svsrnn', clear_tensorboard=True):
        super(SVSRNN, self).__init__()
        
        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        
        self.rnn_layers = [tf.keras.layers.GRU(units, return_sequences=True) for units in self.num_hidden_units]
        self.dense_src1 = tf.keras.layers.Dense(self.num_features, activation='relu', name='y_hat_src1')
        self.dense_src2 = tf.keras.layers.Dense(self.num_features, activation='relu', name='y_hat_src2')

        self.gamma = 0.001
        self.optimizer = tf.keras.optimizers.Adam()

        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors=True)
        self.writer = tf.summary.create_file_writer(tensorboard_directory)
        
    def call(self, inputs, training=False):
        x = inputs
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        y_hat_src1 = self.dense_src1(x)
        y_hat_src2 = self.dense_src2(x)
        
        mask_logits = tf.stack([y_hat_src1, y_hat_src2], axis=-1)
        mask = tf.nn.softmax(mask_logits, axis=-1)
    
        y_tilde_src1 = mask[..., 0] * inputs
        y_tilde_src2 = mask[..., 1] * inputs
        
        return y_tilde_src1, y_tilde_src2

    def si_snr(self, target, estimate, eps=1e-8):
        target = tf.reshape(target, [tf.shape(target)[0], -1])
        estimate = tf.reshape(estimate, [tf.shape(estimate)[0], -1])
        target_mean = tf.reduce_mean(target, axis=1, keepdims=True)
        estimate_mean = tf.reduce_mean(estimate, axis=1, keepdims=True)
        target_zm = target - target_mean
        estimate_zm = estimate - estimate_mean

        # Projection of estimate onto target
        s_target = tf.reduce_sum(estimate_zm * target_zm, axis=1, keepdims=True) * target_zm / (
            tf.reduce_sum(target_zm ** 2, axis=1, keepdims=True) + eps)
        e_noise = estimate_zm - s_target

        # Clamp numerator and denominator to avoid log(0) or division by zero
        s_target_energy = tf.reduce_sum(s_target ** 2, axis=1) + eps
        e_noise_energy = tf.reduce_sum(e_noise ** 2, axis=1) + eps

        si_snr = 10 * tf.math.log(s_target_energy / e_noise_energy) / tf.math.log(10.0)
        return si_snr

    def loss_fn(self, y_src1, y_src2, y_pred_src1, y_pred_src2):
        # SI-SNR is a maximization metric, so we minimize -SI-SNR
        si_snr1 = self.si_snr(y_src1, y_pred_src1)
        si_snr2 = self.si_snr(y_src2, y_pred_src2)
        loss = -(tf.reduce_mean(si_snr1) + tf.reduce_mean(si_snr2)) / 2.0
        return loss

    @tf.function
    def train_step(self, x, y1, y2):
        with tf.GradientTape() as tape:
            y_pred_src1, y_pred_src2 = self(x, training=True)
            loss = self.loss_fn(y1, y2, y_pred_src1, y_pred_src2)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, y_pred_src1, y_pred_src2

    def train(self, x, y1, y2, learning_rate):
        self.optimizer.learning_rate = learning_rate
        loss, y_pred_src1, y_pred_src2 = self.train_step(x, y1, y2)
        
        self.summary(x, y1, y2, y_pred_src1, y_pred_src2, loss)
        
        return loss.numpy()

    @tf.function
    def validate(self, x, y1, y2):
        y1_pred, y2_pred = self(x, training=False)
        validate_loss = self.loss_fn(y1, y2, y1_pred, y2_pred)
        return y1_pred, y2_pred, validate_loss

    def test(self, x):
        return self(x, training=False)

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not filename.endswith('.weights.h5'):
            filename = filename.rsplit('.', 1)[0] + '.weights.h5'
        self.save_weights(os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):
        self.load_weights(filepath)

    def summary(self, x, y1, y2, y_pred_src1, y_pred_src2, loss):
        with self.writer.as_default():
            tf.summary.scalar('loss', loss, step=self.gstep)
            tf.summary.histogram('x_mixed', x, step=self.gstep)
            tf.summary.histogram('y_src1', y1, step=self.gstep)
            tf.summary.histogram('y_src2', y2, step=self.gstep)
            tf.summary.histogram('y_pred_src1', y_pred_src1, step=self.gstep)
            tf.summary.histogram('y_pred_src2', y_pred_src2, step=self.gstep)
            self.writer.flush()
        
        self.gstep.assign_add(1)
