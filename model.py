import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse


def load_images(file_name_list, base_dir, use_augmentation=False):
    images = []

    for file_name in file_name_list:
        fullname = os.path.join(base_dir, file_name).replace("\\", "/")
        img = cv2.imread(fullname)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)

        if img is not None:
            img = np.array(img)

            n_img = (img - 128.0) / 128.0
            images.append(n_img)

            if use_augmentation is True:
                n_img = cv2.flip(img, 1)
                n_img = (n_img - 128.0) / 128.0
                images.append(n_img)

    return np.array(images)


def discriminator(x, activation='relu', scope='discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = dense_block_depth
        bottleneck_width = 8
        if use_patch is True:
            bottleneck_width = 16
        #num_iter = input_width // bottleneck_width
        #num_iter = int(np.sqrt(num_iter))
        num_iter = 3

        print('Discriminator Input: ' + str(x.get_shape().as_list()))
        l = layers.conv(x, scope='conv_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Discriminator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            for j in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='res_block_' + str(i) + '_' + str(j))
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        if use_patch is True:
            print('Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='patch_block_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')
            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[3, 3, 1], stride_dims=[1, 1],
                                non_linear_fn=None, bias=False)
            print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            #print('Discriminator Attention Block : ' + str(l.get_shape().as_list()))
            #l = layers.self_attention(l, block_depth, act_func=act_func)
            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=False,
                                              scope='at_block_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')

            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.fc(feature, 1, non_linear_fn=None, scope='flat')

    return feature, logit


def latent_discriminator(x, activation='relu', scope='latent_discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Latent Discriminator Input: ' + str(x.get_shape().as_list()))

        l = x
        l = layers.fc(l, l.get_shape().as_list()[-1] // 2, non_linear_fn=act_func, scope='flat1')
        print('Latent Discriminator layer 1: ' + str(l.get_shape().as_list()))
        # l = layers.layer_norm(l, scope='ln0')

        feature = layers.fc(l, l.get_shape().as_list()[-1] // 4, non_linear_fn=act_func, scope='flat2')
        print('Latent Discriminator Feature: ' + str(feature.get_shape().as_list()))
        logit = layers.fc(feature, 1, non_linear_fn=None, scope='final')

    return feature, logit


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        # loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(target, value)))))
    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'hinge':
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ls':
        return tf.reduce_mean((real - fake) ** 2)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        # loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l1')
    a_n = get_residual_loss(anchor, negative, 'l1')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return tf.reduce_mean(a_p / a_n)


def get_gradient_loss(img1, img2):
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    # loss = tf.abs(tf.subtract(v_a, v_b))
    loss = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b))) + tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss


def generate_sample_z(low, high, num_samples, sample_length, b_uniform=True):
    if b_uniform is True:
        z = np.random.uniform(low=low, high=high, size=[num_samples, sample_length])
    else:
        z = np.random.normal(low, high, size=[num_samples, sample_length])

    return z


def make_multi_modal_noise(num_mode=8):
    size = representation_dim // num_mode

    for i in range(batch_size):
        noise = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)

    for i in range(num_mode - 1):
        n = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)
        noise = tf.concat([noise, n], axis=1)

    return noise


def translator(x, activation='relu', scope='translator', norm='layer', use_upsample=False, b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        bottleneck_width = 64
        bottleneck_itr = 9
        num_iter = input_width // bottleneck_width
        num_iter = int(np.sqrt(num_iter))

        print('Translator Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        l = layers.conv(x, scope='conv_init', filter_dims=[7, 7, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Translator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        for i in range(bottleneck_itr):
            print('Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='bt_block_' + str(i))

        for i in range(num_iter):
            block_depth = block_depth // 2

            if use_upsample is True:
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                # l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                # l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))
                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

                for j in range(2):
                    l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                  act_func=act_func, norm=norm, b_train=b_train, use_dilation=False,
                                                  scope='block_' + str(i) + '_' + str(j))
            else:
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                  filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print('Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        l = layers.conv(l, scope='last', filter_dims=[7, 7, num_channel], stride_dims=[1, 1], non_linear_fn=tf.nn.tanh,
                        bias=False)

        print('Translator Final: ' + str(l.get_shape().as_list()))

    return l


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    G_scope = 'translator_X_to_Y'
    F_scope = 'translator_Y_to_X'
    DX_scope = 'discriminator_X'
    DY_scope = 'discriminator_Y'

    with tf.device('/device:CPU:0'):
        X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        X_FAKE_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_FAKE_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR = tf.placeholder(tf.float32, None)
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.device('/device:GPU:1'):
        fake_Y = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                            use_upsample=False)
        recon_X = translator(fake_Y, activation='swish', norm='instance', b_train=b_train, scope=F_scope,
                             use_upsample=False)
        fake_X = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=F_scope,
                            use_upsample=False)
        recon_Y = translator(fake_X, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                             use_upsample=False)

        if use_identity_loss is True:
            id_X = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=F_scope,
                              use_upsample=False)
            id_Y = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                              use_upsample=False)

    with tf.device('/device:GPU:0'):
        _, X_FAKE_IN_logit = discriminator(X_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                           scope=DX_scope, use_patch=True)
        _, Y_FAKE_IN_logit = discriminator(Y_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                           scope=DY_scope, use_patch=True)
        _, real_X_logit = discriminator(X_IN, activation='swish', norm='instance', b_train=b_train,
                                        scope=DX_scope, use_patch=True)
        _, fake_X_logit = discriminator(fake_X, activation='swish', norm='instance', b_train=b_train,
                                        scope=DX_scope, use_patch=True)
        _, real_Y_logit = discriminator(Y_IN, activation='swish', norm='instance', b_train=b_train,
                                        scope=DY_scope, use_patch=True)
        _, fake_Y_logit = discriminator(fake_Y, activation='swish', norm='instance', b_train=b_train,
                                        scope=DY_scope, use_patch=True)

    reconstruction_loss_Y = get_residual_loss(Y_IN, recon_Y, type='l1')
    reconstruction_loss_X = get_residual_loss(X_IN, recon_X, type='l1')
    cyclic_loss = reconstruction_loss_Y + reconstruction_loss_X
    alpha = 10.0
    cyclic_loss = alpha * cyclic_loss

    if gan_mode == 'ls':
        # LS GAN
        trans_loss_X2Y = tf.reduce_mean((fake_Y_logit - tf.ones_like(fake_Y_logit)) ** 2)
        trans_loss_Y2X = tf.reduce_mean((fake_X_logit - tf.ones_like(fake_X_logit)) ** 2)
        disc_loss_Y = get_discriminator_loss(real_Y_logit, tf.ones_like(real_Y_logit), type='ls') + \
                      get_discriminator_loss(Y_FAKE_IN_logit, tf.zeros_like(Y_FAKE_IN_logit), type='ls')
        disc_loss_X = get_discriminator_loss(real_X_logit, tf.ones_like(real_X_logit), type='ls') + \
                      get_discriminator_loss(X_FAKE_IN_logit, tf.zeros_like(X_FAKE_IN_logit), type='ls')
    else:
        # for WGAN
        trans_loss_X2Y = -tf.reduce_mean(fake_Y_logit)
        trans_loss_Y2X = -tf.reduce_mean(fake_X_logit)
        disc_loss_Y, _, _ = get_discriminator_loss(real_Y_logit, Y_FAKE_IN_logit, type='wgan')
        disc_loss_X, _, _ = get_discriminator_loss(real_X_logit, X_FAKE_IN_logit, type='wgan')

    if use_identity_loss is True:
        identity_loss_Y = alpha * (get_residual_loss(Y_IN, id_Y, type='l1'))
        identity_loss_X = alpha * (get_residual_loss(X_IN, id_X, type='l1'))
        identity_loss = 0.5 * (identity_loss_X + identity_loss_Y)
        total_trans_loss = trans_loss_X2Y + trans_loss_Y2X + cyclic_loss + identity_loss
    else:
        total_trans_loss = trans_loss_X2Y + trans_loss_Y2X + cyclic_loss

    beta = 0.5
    total_disc_loss = beta * disc_loss_Y + beta * disc_loss_X

    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)
    disc_X_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DX_scope)
    disc_vars = disc_Y_vars + disc_X_vars

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    trans_Y2X_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=F_scope)
    trans_vars = trans_X2Y_vars + trans_Y2X_vars

    if gan_mode == 'wgan':
        # Alert: Clip range is critical to WGAN.
        disc_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars]

    with tf.device('/device:GPU:0'):
        #disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(total_disc_loss, var_list=disc_vars)
        disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_disc_loss, var_list=disc_vars)

    with tf.device('/device:GPU:1'):
        #trans_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(total_trans_loss, var_list=trans_vars)
        trans_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_trans_loss, var_list=trans_vars)
    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trX_dir = os.path.join(train_data, 'X').replace("\\", "/")
        trY_dir = os.path.join(train_data, 'Y').replace("\\", "/")
        trX = os.listdir(trX_dir)
        trY = os.listdir(trY_dir)
        total_input_size = min(len(trX), len(trY))

        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        if file_batch_size == 0:
            file_batch_size = 1

        num_critic = 1
        if gan_mode == 'wgan':
            num_critic = 5

        image_pool = util.ImagePool(maxsize=50)
        learning_rate = 2e-4
        lr_decay_step = 100

        for e in range(num_epoch):
            trX = shuffle(trX)
            tfY = shuffle(trY)

            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0
            if e > lr_decay_step:
                learning_rate = learning_rate * (num_epoch - e)/(num_epoch - lr_decay_step)

            for start, end in training_batch:
                imgs_X = load_images(trX[start:end], base_dir=trX_dir, use_augmentation=False)
                if len(imgs_X[0].shape) != 3:
                    imgs_X = np.expand_dims(imgs_X, axis=3)
                imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False)
                if len(imgs_Y[0].shape) != 3:
                    imgs_Y = np.expand_dims(imgs_Y, axis=3)

                trans_X2Y, trans_Y2X = sess.run([fake_Y, fake_X], feed_dict={X_IN: imgs_X, Y_IN: imgs_Y, b_train: True})
                pool_X2Y, pool_Y2X = image_pool([trans_X2Y, trans_Y2X])

                _, d_loss = sess.run([disc_optimizer, total_disc_loss],
                                     feed_dict={X_IN: imgs_X, Y_IN: imgs_Y,
                                                X_FAKE_IN: pool_Y2X, Y_FAKE_IN: pool_X2Y, b_train: True, LR: learning_rate})

                if gan_mode == 'wgan':
                    _ = sess.run([disc_weight_clip])

                if itr % num_critic == 0:
                    _, t_loss, x2y_loss, y2x_loss = sess.run([trans_optimizer, total_trans_loss, trans_loss_X2Y, trans_loss_Y2X],
                                         feed_dict={Y_IN: imgs_Y, X_IN: imgs_X, b_train: True, LR: learning_rate})

                    print('epoch: ' + str(e) + ', d_loss: ' + str(d_loss) +
                          ', t_loss: ' + str(t_loss) + ', x2y: ' + str(x2y_loss) + ', y2x: ' + str(y2x_loss))
                    decoded_images_Y2X = np.squeeze(trans_Y2X)
                    decoded_images_X2Y = np.squeeze(trans_X2Y)
                    cv2.imwrite('imgs/Y2X_' + trY[start], (decoded_images_Y2X[0] * 128.0) + 128.0)
                    cv2.imwrite('imgs/X2Y_' + trX[start], (decoded_images_X2Y[0] * 128.0) + 128.0)
                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')
            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def train_one2one(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    G_scope = 'translator_X_to_Y'
    DX_scope = 'discriminator_X'
    DY_scope = 'discriminator_Y'

    with tf.device('/device:CPU:0'):
        X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        X_FAKE_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_FAKE_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR = tf.placeholder(tf.float32, None)
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.device('/device:GPU:1'):
        fake_Y = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                            use_upsample=False)
        recon_X = translator(fake_Y, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                             use_upsample=False)
        fake_X = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                            use_upsample=False)
        recon_Y = translator(fake_X, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                             use_upsample=False)

        if use_identity_loss is True:
            id_X = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                              use_upsample=False)
            id_Y = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                              use_upsample=False)

    with tf.device('/device:GPU:0'):
        _, X_FAKE_IN_logit = discriminator(X_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                           scope=DX_scope, use_patch=True)
        _, Y_FAKE_IN_logit = discriminator(Y_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                           scope=DY_scope, use_patch=True)
        _, real_X_logit = discriminator(X_IN, activation='swish', norm='instance', b_train=b_train,
                                        scope=DX_scope, use_patch=True)
        _, fake_X_logit = discriminator(fake_X, activation='swish', norm='instance', b_train=b_train,
                                        scope=DX_scope, use_patch=True)
        _, real_Y_logit = discriminator(Y_IN, activation='swish', norm='instance', b_train=b_train,
                                        scope=DY_scope, use_patch=True)
        _, fake_Y_logit = discriminator(fake_Y, activation='swish', norm='instance', b_train=b_train,
                                        scope=DY_scope, use_patch=True)

    reconstruction_loss_Y = get_residual_loss(Y_IN, recon_Y, type='l1') + get_gradient_loss(Y_IN, recon_Y)
    reconstruction_loss_X = get_residual_loss(X_IN, recon_X, type='l1') + get_gradient_loss(X_IN, recon_X)
    cyclic_loss = reconstruction_loss_Y + reconstruction_loss_X
    alpha = 10.0
    cyclic_loss = alpha * cyclic_loss

    if gan_mode == 'ls':
        # LS GAN
        trans_loss_X2Y = tf.reduce_mean((fake_Y_logit - tf.ones_like(fake_Y_logit)) ** 2) + alpha * reconstruction_loss_X
        trans_loss_Y2X = tf.reduce_mean((fake_X_logit - tf.ones_like(fake_X_logit)) ** 2) + alpha * reconstruction_loss_Y
        disc_loss_Y = get_discriminator_loss(real_Y_logit, tf.ones_like(real_Y_logit), type='ls') + \
                      get_discriminator_loss(Y_FAKE_IN_logit, tf.zeros_like(Y_FAKE_IN_logit), type='ls')
        disc_loss_X = get_discriminator_loss(real_X_logit, tf.ones_like(real_X_logit), type='ls') + \
                      get_discriminator_loss(X_FAKE_IN_logit, tf.zeros_like(X_FAKE_IN_logit), type='ls')
    else:
        # for WGAN
        trans_loss_X2Y = -tf.reduce_mean(fake_Y_logit)
        trans_loss_Y2X = -tf.reduce_mean(fake_X_logit)
        disc_loss_Y, _, _ = get_discriminator_loss(real_Y_logit, Y_FAKE_IN_logit, type='wgan')
        disc_loss_X, _, _ = get_discriminator_loss(real_X_logit, X_FAKE_IN_logit, type='wgan')

    if use_identity_loss is True:
        identity_loss_Y = alpha * (get_residual_loss(Y_IN, id_Y, type='l1') + get_gradient_loss(Y_IN, id_Y))
        identity_loss_X = alpha * (get_residual_loss(X_IN, id_X, type='l1') + get_gradient_loss(X_IN, id_X))
        identity_loss = 0.5 * (identity_loss_X + identity_loss_Y)
        total_trans_loss = trans_loss_X2Y + trans_loss_Y2X + cyclic_loss + identity_loss
    else:
        total_trans_loss = trans_loss_X2Y + trans_loss_Y2X + cyclic_loss

    beta = 0.5
    total_disc_loss = beta * disc_loss_Y + beta * disc_loss_X

    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)
    disc_X_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DX_scope)
    disc_vars = disc_Y_vars + disc_X_vars

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    trans_vars = trans_X2Y_vars

    if gan_mode == 'wgan':
        # Alert: Clip range is critical to WGAN.
        disc_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars]

    with tf.device('/device:GPU:0'):
        disc_optimizer_X = tf.train.AdamOptimizer(learning_rate=LR).minimize(disc_loss_X, var_list=disc_X_vars)
        disc_optimizer_Y = tf.train.AdamOptimizer(learning_rate=LR).minimize(disc_loss_Y, var_list=disc_Y_vars)

    with tf.device('/device:GPU:1'):
        trans_optimizer_X2Y = tf.train.AdamOptimizer(learning_rate=LR).minimize(trans_loss_X2Y, var_list=trans_vars)
        trans_optimizer_Y2X = tf.train.AdamOptimizer(learning_rate=LR).minimize(trans_loss_Y2X, var_list=trans_vars)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trX_dir = os.path.join(train_data, 'X').replace("\\", "/")
        trY_dir = os.path.join(train_data, 'Y').replace("\\", "/")
        trX = os.listdir(trX_dir)
        trY = os.listdir(trY_dir)
        total_input_size = min(len(trX), len(trY))

        num_augmentations = 2  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        if file_batch_size == 0:
            file_batch_size = 1

        num_critic = 1
        if gan_mode == 'wgan':
            num_critic = 5

        image_pool = util.ImagePool(maxsize=50)
        learning_rate = 2e-4
        lr_decay_step = 100

        for e in range(num_epoch):
            trX = shuffle(trX)
            tfY = shuffle(trY)

            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0
            if e > lr_decay_step:
                learning_rate = learning_rate * (num_epoch - e)/(num_epoch - lr_decay_step)

            for start, end in training_batch:
                imgs_X = load_images(trX[start:end], base_dir=trX_dir, use_augmentation=False)
                if len(imgs_X[0].shape) != 3:
                    imgs_X = np.expand_dims(imgs_X, axis=3)
                imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False)
                if len(imgs_Y[0].shape) != 3:
                    imgs_Y = np.expand_dims(imgs_Y, axis=3)

                trans_X2Y, trans_Y2X = sess.run([fake_Y, fake_X], feed_dict={X_IN: imgs_X, Y_IN: imgs_Y, b_train: True})
                pool_X2Y, pool_Y2X = image_pool([trans_X2Y, trans_Y2X])

                _, dx_loss = sess.run([disc_optimizer_X, disc_loss_X],
                                     feed_dict={X_IN: imgs_X, X_FAKE_IN: pool_Y2X, b_train: True, LR: learning_rate})
                _, dy_loss = sess.run([disc_optimizer_Y, disc_loss_Y],
                                     feed_dict={Y_IN: imgs_Y, Y_FAKE_IN: pool_X2Y, b_train: True, LR: learning_rate})

                if gan_mode == 'wgan':
                    _ = sess.run([disc_weight_clip])

                if itr % num_critic == 0:
                    _, tx2y_loss = sess.run([trans_optimizer_X2Y, trans_loss_X2Y],
                                         feed_dict={X_IN: imgs_X, b_train: True, LR: learning_rate})
                    _, ty2x_loss = sess.run([trans_optimizer_Y2X, trans_loss_Y2X],
                                         feed_dict={Y_IN: imgs_Y, b_train: True, LR: learning_rate})

                    print('epoch: ' + str(e) + ', d_loss: ' + str(dx_loss + dy_loss) +
                          ', t_loss: ' + str(tx2y_loss + ty2x_loss))
                    decoded_images_Y2X = np.squeeze(trans_Y2X)
                    decoded_images_X2Y = np.squeeze(trans_X2Y)
                    cv2.imwrite('imgs/Y2X_' + trY[start], (decoded_images_Y2X * 128.0) + 128.0)
                    cv2.imwrite('imgs/X2Y_' + trX[start], (decoded_images_X2Y * 128.0) + 128.0)
                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')
            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def test(model_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path

    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32
    batch_size = 2
    representation_dim = 128

    img_width = 256
    img_height = 256
    input_width = 256
    input_height = 256
    num_channel = 3

    test_size = 100
    num_epoch = 300
    gan_mode = 'ls'
    use_identity_loss = False

    if args.mode == 'train':
        train(model_path)
        # train_one2one(model_path)
    else:
        model_path = args.model_path
        test_data = args.test_data
        batch_size = 1
        test(model_path)
