import vgg19 as vgg19
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface


def gram(matrix):
    gram_shift = -1.
    dim_0, dim_1, dim_2 = matrix.get_shape().as_list()
    matrix = tf.transpose(matrix, [2, 0, 1])
    matrix = tf.reshape(matrix, [dim_2, dim_0 * dim_1])
    return tf.matmul(matrix + gram_shift,
                     tf.matrix_transpose(matrix) + gram_shift)  # / matrix.get_shape().num_elements()


def style_loss(layers, style_weights):
    s_loss = 0
    for layer in layers:
        dim_0, dim_1, dim_2, dim_3 = layers[layer].get_shape().as_list()
        M = dim_1 * dim_2
        N = dim_3
        style_layer_gram = gram(layers[layer][1])
        generated_layer_gram = gram(layers[layer][2])
        s_loss += (style_weights[layer] * (tf.reduce_mean((style_layer_gram - generated_layer_gram) ** 2.))) / (
            4 * M ** 2 * N ** 2)
    return s_loss


def style_loss1(layers, style_weights):
    return 0

def content_loss1(layers, content_weights):
    c_loss = 0
    for layer in layers:
        c_loss += content_weights[layer] * (tf.reduce_mean((layers[layer][0] - layers[layer][2]) ** 2.))
    return c_loss


def set_style_weights(layers):
    style_weights = {}
    num_layers = len(layers)
    for layer in layers:
        dim_0, dim_1, dim_2, dim_3 = layers[layer].get_shape().as_list()
        M = dim_1 * dim_2
        N = dim_3
        style_layer_gram = gram(layers[layer][1])
        generated_layer_gram = gram(layers[layer][2])
        style_weights[layer] = (1e2 / num_layers) / ((tf.reduce_mean(
            (style_layer_gram - generated_layer_gram) ** 2.)) / (4 * M ** 2 * N ** 2))
    return style_weights


def content_loss(content_layer, generated_layer):
    return tf.reduce_mean((content_layer - generated_layer) ** 2) / 2


def noise_loss(sandwich_tf):
    noise_rows = tf.reduce_mean((sandwich_tf[2, 1:, :, :] - sandwich_tf[2, :-1, :, :]) ** 2)
    noise_cols = tf.reduce_mean((sandwich_tf[2, :, 1:, :] - sandwich_tf[2, :, :-1, :]) ** 2)
    return noise_rows + noise_cols
    # return tf.image.total_variation(sandwich_tf[2])/sandwich_tf[2].get_shape().num_elements()


def tensor_transfer(sandwich, style_weights_mode=1, content_coef=5e-3, noise_coef=1e0, mask=False):
    g = tf.Graph()
    with g.as_default():
        content_image_tf = tf.expand_dims(tf.constant(sandwich[0], tf.float32), axis=0)
        style_image_tf = tf.expand_dims(tf.constant(sandwich[1], tf.float32), axis=0)

        try:
            assert mask.dtype == 'float32'
            premask_img = tf.clip_by_value(
                tf.Variable(sandwich[2], dtype=tf.float32, name='gen_image'), 0.01, 0.99)
            generated_image_tf = tf.expand_dims(mask * premask_img, axis=0)
            use_mask = True
        except:
            generated_image_tf = tf.expand_dims(
                tf.clip_by_value(tf.Variable(sandwich[2], dtype=tf.float32, name='gen_image'), 0.01, 0.99), axis=0)
            use_mask = False

        sandwich_tf = tf.concat((content_image_tf, style_image_tf, generated_image_tf), axis=0)

        vgg = vgg19.Vgg19()
        vgg.build(sandwich_tf)

        layers = {'conv1_1': vgg.conv1_1,  # ,'conv1_2':vgg.conv1_2,
                  'conv2_1': vgg.conv2_1,  # ,'conv2_2':vgg.conv2_2,
                  'conv3_1': vgg.conv3_1,  # 'conv3_2': vgg.conv3_2,'conv3_3': vgg.conv3_3,
                  'conv4_1': vgg.conv4_1,  # 'conv4_2': vgg.conv4_2,'conv4_3': vgg.conv4_3,
                  'conv5_1': vgg.conv5_1}  # ,'conv5_2': vgg.conv5_2,'conv5_3': vgg.conv5_3}

        content_layer = vgg.conv4_2[0]
        generated_layer = vgg.conv4_2[2]

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # ==========================================
            if style_weights_mode == 0:
                style_weights = {'conv1_1': 0., 'conv2_1': 0., 'conv3_1': 0., 'conv4_1': 0., 'conv5_1': 0.}
            if style_weights_mode == 1:
                style_weights = sess.run(set_style_weights(layers))
            if style_weights_mode == 2:
                style_weights = {'conv1_1': 0.2, 'conv2_1': 0.2, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
            if style_weights_mode == 3:
                style_weights = {'conv1_1': .2, 'conv2_1': .4, 'conv3_1': .8, 'conv4_1': 1.6, 'conv5_1': 3.2}
            # ==========================================
            content_weights = {'conv1_1': 0.1, 'conv2_1': 0.2, 'conv3_1': 0.3, 'conv4_1': 0.4, 'conv5_1': 0.5}

            loss = style_loss(layers, style_weights) + \
                   noise_coef * noise_loss(sandwich_tf) + \
                   content_coef * content_loss1(layers, content_weights)

            optimizer = ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 20})

            print('from', loss.eval())
            print("noise loss", noise_loss(sandwich_tf).eval())
            print('content_loss', content_loss(content_layer, generated_layer).eval())
            optimizer.minimize(sess)
            # =========================================
            sandwich = sandwich_tf.eval()
            if use_mask:
                image = premask_img.eval()
                sandwich[2] = image
                # =========================================
    del vgg
    return sandwich
