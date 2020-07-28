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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for_face = True

        if for_face is False:
            # For SKC
            if use_augmentation is True:
                resized_img = cv2.resize(img, dsize=(512, 520), interpolation=cv2.INTER_CUBIC)
                resized_img_hd = cv2.resize(img, dsize=(1024, 1040), interpolation=cv2.INTER_CUBIC)
        else:
            # For face
            img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        if img is not None:
            img = np.array(img)

            if for_face is True:
                # For face
                n_img = (img - 128.0) / 128.0
                images.append(n_img)

                if use_augmentation is True:
                    n_img = cv2.flip(img, 1)
                    n_img = (img - 128.0) / 128.0
                    images.append(n_img)
            else:
                img = img[4:260, 0:256]

                # Center crop
                center_x = 256 // 2
                center_y = 256 // 2
                img = img[center_y-64:center_y+64, center_x-64:center_x+64]
                n_img = (img - 128.0) / 128.0
                images.append(n_img)

                if use_augmentation is True:
                    n_img = cv2.flip(img, 1)
                    n_img = (n_img - 128.0) / 128.0
                    images.append(n_img)

                    img = np.array(resized_img)
                    img = img[8:520, 0:512]
                    center_x = 512 // 2
                    center_y = 512 // 2
                    img = img[center_y-64:center_y+64, center_x-64:center_x+64]
                    n_img = (img - 128.0) / 128.0
                    images.append(n_img)

                    img = np.array(resized_img_hd)
                    img = img[16:1040, 0:1024]
                    center_x = 1024 // 2
                    center_y = 1024 // 2
                    img = img[center_y - 64:center_y + 64, center_x - 64:center_x + 64]
                    n_img = (img - 128.0) / 128.0
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
        num_iter = input_width // bottleneck_width
        num_iter = int(np.sqrt(num_iter))

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

            logit = layers.global_avg_pool(last_layer, output_length=1, use_bias=False,
                                             scope='gp_logit')
            print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            print('Discriminator Attention Block : ' + str(l.get_shape().as_list()))
            l = layers.self_attention(l, block_depth, act_func=act_func)
            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=False, scope='at_block_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False, scope='gp')

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
        #l = layers.layer_norm(l, scope='ln0')

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
        #loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
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


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))
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
    image_a = img1 #tf.expand_dims(img1, axis=0)
    image_b = img2 #tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    #loss = tf.abs(tf.subtract(v_a, v_b))
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

    for i in range(num_mode-1):
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

        bottleneck_width = 8
        bottleneck_itr = 4
        num_iter = input_width // bottleneck_width
        num_iter = int(np.sqrt(num_iter))

        print('Translator Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        l = layers.conv(x, scope='conv_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Translator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            #for j in range(1):
            #    l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
            #                                  norm=norm, b_train=b_train, scope='res_block_' + str(i) + '_' + str(j))
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        # [128, 128] -> [8, 8]
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

        if use_upsample is False:
            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                              act_func=act_func, norm=norm, b_train=b_train, use_dilation=False,
                                              scope='tr_block_' + str(i))

        l = layers.conv(l, scope='last', filter_dims=[1, 1, num_channel], stride_dims=[1, 1], non_linear_fn=tf.nn.tanh,
                        bias=False)

        print('Translator Final: ' + str(l.get_shape().as_list()))

    return l


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        G = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        F = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.device('/device:GPU:0'):
        fake_F = translator(G, activation='swish', norm='instance', b_train=b_train, scope='translator_G_to_F', use_upsample=False)
        fake_F_feature, fake_F_logit = discriminator(fake_F, activation='swish', norm='instance', b_train=b_train, scope='discriminator_F', use_patch=True)
        real_F_feature, real_F_logit = discriminator(F, activation='swish', norm='instance', b_train=b_train, scope='discriminator_F', use_patch=True)

    with tf.device('/device:GPU:1'):
        fake_G = translator(F, activation='swish', norm='instance', b_train=b_train, scope='translator_F_to_G', use_upsample=False)
        fake_G_feature, fake_G_logit = discriminator(fake_G, activation='swish', norm='instance', b_train=b_train, scope='discriminator_G', use_patch=True)
        real_G_feature, real_G_logit = discriminator(G, activation='swish', norm='instance', b_train=b_train, scope='discriminator_G', use_patch=True)

    with tf.device('/device:GPU:0'):
        recon_F = translator(fake_G, activation='swish', norm='instance', b_train=b_train, scope='translator_G_to_F',
                             use_upsample=False)
    with tf.device('/device:GPU:1'):
        recon_G = translator(fake_F, activation='swish', norm='instance', b_train=b_train, scope='translator_F_to_G',
                             use_upsample=False)

    reconstruction_loss_F = get_residual_loss(F, recon_F, type='l1')
    reconstruction_loss_G = get_residual_loss(G, recon_G, type='l1')
    cyclic_loss = reconstruction_loss_F + reconstruction_loss_G
    alpha = 10.0
    cyclic_loss = alpha * cyclic_loss

    disc_loss_F, _, _ = get_discriminator_loss(real_F_logit, fake_F_logit, type='wgan')
    disc_loss_G, _, _ = get_discriminator_loss(real_G_logit, fake_G_logit, type='wgan')

    trans_loss_G2F = -tf.reduce_mean(fake_F_logit)
    trans_loss_F2G = -tf.reduce_mean(fake_G_logit)

    disc_F_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_F')
    disc_G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_G')
    disc_vars = disc_F_vars + disc_G_vars

    trans_G2F_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='translator_G_to_F')
    trans_F2G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='translator_F_to_G')

    # Alert: Clip range is critical to WGAN.
    disc_weight_clip = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in disc_vars]

    disc_G_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss_G, var_list=disc_G_vars)
    disc_F_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss_F, var_list=disc_F_vars)
    trans_G2F_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(trans_loss_G2F, var_list=trans_G2F_vars)
    trans_F2G_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(trans_loss_F2G, var_list=trans_F2G_vars)

    with tf.device('/device:GPU:1'):
        cyclic_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(cyclic_loss)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trG_dir = os.path.join(train_data, 'G').replace("\\", "/")
        trF_dir = os.path.join(train_data, 'F').replace("\\", "/")
        trG = os.listdir(trG_dir)
        trF = os.listdir(trF_dir)

        num_augmentations = 2  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations
        num_critic = 5

        for e in range(num_epoch):
            trG = shuffle(trG)
            trF = shuffle(trF)

            training_batch = zip(range(0, len(trG), file_batch_size),  range(file_batch_size, len(trG)+1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs_G = load_images(trG[start:end], base_dir=trG_dir, use_augmentation=True)
                imgs_G = np.expand_dims(imgs_G, axis=3)
                imgs_F = load_images(trF[start:end], base_dir=trF_dir, use_augmentation=True)
                imgs_F = np.expand_dims(imgs_F, axis=3)

                _, d_loss_G = sess.run([disc_G_optimizer, disc_loss_G],
                                        feed_dict={G: imgs_G, F: imgs_F, b_train: True})

                _, d_loss_F = sess.run([disc_F_optimizer, disc_loss_F],
                                        feed_dict={G: imgs_G, F: imgs_F, b_train: True})

                _ = sess.run([disc_weight_clip])

                if itr % num_critic == 0:
                    _, t_loss_G = sess.run([trans_G2F_optimizer, trans_loss_G2F], feed_dict={G: imgs_G,  b_train: True})
                    _, t_loss_F = sess.run([trans_F2G_optimizer, trans_loss_F2G], feed_dict={F: imgs_F,  b_train: True})
                    _, c_loss = sess.run([cyclic_optimizer, cyclic_loss], feed_dict={G: imgs_G, F: imgs_F, b_train: True})

                    print('epoch: ' + str(e) + ', d_loss_G: ' + str(d_loss_G) + ', d_loss_F: ' + str(d_loss_F) +
                          ', t_loss_G: ' + str(t_loss_G) + ', t_loss_F: ' + str(t_loss_F) +
                          ', c_loss: ' + str(c_loss))

                    decoded_images_F, decoded_images_G = sess.run([fake_F, fake_G], feed_dict={G: imgs_G, F: imgs_F, b_train: True})
                    decoded_images_F = np.squeeze(decoded_images_F)
                    decoded_images_G = np.squeeze(decoded_images_G)
                    cv2.imwrite('imgs/' + trG[start], (decoded_images_G[3] * 128.0) + 128.0)
                    cv2.imwrite('imgs/' + trF[start], (decoded_images_F[3] * 128.0) + 128.0)

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
    batch_size = 16
    representation_dim = 128

    img_width = 256
    img_height = 256
    input_width = 128
    input_height = 128
    num_channel = 1

    test_size = 100
    num_epoch = 30000

    if args.mode == 'train':
        train(model_path)
    else:
        model_path = args.model_path
        test_data = args.test_data
        batch_size = 1
        test(model_path)
