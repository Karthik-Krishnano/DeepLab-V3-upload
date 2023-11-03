
import os
from datetime import datetime
import tf_slim as slim
import tensorflow as tf
from feature_extractor import MobileNet, Resnet, Vgg16
from modules import atrous_spatial_pyramid_pooling
import tensorflow.compat.v1 as tfa
tfa.compat.v1.disable_eager_execution()

class DeepLab(object):

    def __init__(self, base_architecture, training=True, num_classes=21, ignore_label=255, batch_norm_momentum=0.9997, pre_trained_model=None, log_dir='data/logs/deeplab/'):

        self.is_training = tfa.placeholder(tf.bool, None, name='is_training')
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.inputs_shape = [None, None, None, 3]
        self.labels_shape = [None, None, None, 1]
        self.training = training
        self.inputs = tfa.placeholder(tf.float32, shape=self.inputs_shape, name='inputs')
        self.labels = tfa.placeholder(tf.uint8, shape=self.labels_shape, name='labels')

        self.target_height = tfa.placeholder(tf.int32, None, name='target_image_height')
        self.target_width = tfa.placeholder(tf.int32, None, name='target_image_width')

        self.weight_decay = tfa.placeholder(tf.float32, None, name='weight_decay')
        #self.regularizer = tf.compat.v1.estimator.layers.l2_regularizer(scale=self.weight_decay)
        self.batch_norm_momentum = batch_norm_momentum

        self.feature_map = self.backbone_initializer(base_architecture)
        if pre_trained_model:
            self.initialize_backbone_from_pretrained_weights(pre_trained_model)
        self.outputs = self.model_initializer()

        self.learning_rate = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
        self.loss = self.loss_initializer()

        self.optimizer = self.optimizer_initializer()

        # Initialize tensorflow session
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.training:
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            #self.writer = tf.summary.create_file_writer(self.log_dir, tf.compat.v1.get_default_graph())
            self.train_summaries, self.valid_summaries = self.summary()

    def backbone_initializer(self, base_architecture):

        with tf.compat.v1.variable_scope('backbone'):
            if base_architecture == 'vgg16':
                features = Vgg16(self.inputs, self.weight_decay, self.batch_norm_momentum)
            elif base_architecture.startswith('resnet'):
                n_layers = int(base_architecture.split('_')[-1])
                features = Resnet(n_layers, self.inputs, self.weight_decay, self.batch_norm_momentum, self.is_training)
            elif base_architecture.startswith('mobilenet'):
                depth_multiplier = float(base_architecture.split('_')[-1])
                features = MobileNet(depth_multiplier, self.inputs, self.weight_decay, self.batch_norm_momentum, self.is_training)
            else:
                raise ValueError('Unknown backbone architecture!')

        return features

    def model_initializer(self):

        pools = atrous_spatial_pyramid_pooling(inputs=self.feature_map, filters=256)#, regularizer=self.regularizer)
        logits = tf.compat.v1.layers.conv2d(inputs=pools, filters=self.num_classes, kernel_size=(1, 1), name='logits')
        outputs = tf.compat.v1.image.resize_bilinear(images=logits, size=(self.target_height, self.target_width), name='resized_outputs')

        return outputs

    def loss_initializer(self):

        labels_linear = tf.reshape(tensor=self.labels, shape=[-1])
        not_ignore_mask = tf.cast(tf.not_equal(labels_linear, self.ignore_label),dtype=tf.float32)
        # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
        # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
        onehot_labels = tf.one_hot(indices=labels_linear, depth=self.num_classes, on_value=1.0, off_value=0.0)

        loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights=not_ignore_mask)

        return loss

    def optimizer_initializer(self):

        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        return optimizer

    def summary(self):

        with tf.name_scope('loss'):
            train_loss_summary = tf.summary.scalar('train', self.loss)
            valid_loss_summary = tf.summary.scalar('valid', self.loss)

        return train_loss_summary, valid_loss_summary

    def train(self, inputs, labels, target_height, target_width, learning_rate, weight_decay):

        _, outputs, train_loss, summaries = self.sess.run([self.optimizer, self.outputs, self.loss, self.train_summaries], feed_dict={self.inputs: inputs, self.labels: labels, self.learning_rate: learning_rate, self.target_height: target_height, self.target_width: target_width, self.weight_decay: weight_decay, self.is_training: True})

        #self.writer.add_summary(summaries, self.train_step)
        self.train_step += 1

        return outputs, train_loss

    def validate(self, inputs, labels, target_height, target_width):

        outputs, valid_loss, summaries = self.sess.run([self.outputs, self.loss, self.valid_summaries], feed_dict={self.inputs: inputs, self.labels: labels, self.target_height: target_height, self.target_width: target_width, self.is_training: False})

        #self.writer.add_summary(summaries, self.train_step)

        return outputs, valid_loss

    def test(self, inputs, target_height, target_width):

        outputs = self.sess.run(self.outputs, feed_dict={self.inputs: inputs, self.target_height: target_height, self.target_width: target_width, self.is_training: False})

        return outputs

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

    def initialize_backbone_from_pretrained_weights(self, path_to_pretrained_weights):

        variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
        valid_prefix = 'backbone/'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.train.init_from_checkpoint(path_to_pretrained_weights, {v.name[len(valid_prefix):].split(':')[0]: v for v in variables_to_restore if v.name.startswith(valid_prefix)})

    def close(self):

        #if self.training:
        #    self.writer.close()
        self.sess.close()


if __name__ == '__main__':

    deeplab = DeepLab('resnet_101', pre_trained_model='data/models/pretrained/resnet_101/resnet_v2_101.ckpt')
    print('Graph compiled successfully.')
    deeplab.close()
