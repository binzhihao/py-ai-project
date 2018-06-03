from functools import partial
import tensorflow as tf
import numpy as np
import data.mnist.loader as loader

OPTIMIZER = 'adam'

if __name__ == '__main__':

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    learning_rate = 0.01
    n_epochs = 40
    batch_size = 200
    batch_norm_momentum = 0.9

    def shuffle_batch(X, y, size):
        rnd_idx_ = np.random.permutation(len(X))
        n_batches_ = len(X) // size
        for batch_idx_ in np.array_split(rnd_idx_, n_batches_):
            X_batch_, y_batch_ = X[batch_idx_], y[batch_idx_]
            yield X_batch_, y_batch_

    # prepare data
    X_train, y_train = loader.load_mnist(kind='train')
    X_train = X_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)

    X_test, y_test = loader.load_mnist(kind='test')
    X_test = X_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int32)

    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    # placeholder
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=None, name='y')
    # 给Batch norm加一个placeholder
    training = tf.placeholder_with_default(False, shape=(), name='training')

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=batch_norm_momentum)
        my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)

        hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
        bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
        hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
        logists_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
        logists = my_batch_norm_layer(logists_before_bn)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits= logists)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('train'):
        if OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logists, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # create variables initializer
    init = tf.global_variables_initializer()
    # create a saver
    saver = tf.train.Saver()

    # 注意：由于我们使用的是 tf.layers.batch_normalization() 而不是 tf.contrib.layers.batch_norm()（如本书所述），
    # 所以我们需要明确运行批量规范化所需的额外更新操作（sess.run([ training_op，extra_update_ops], ...)。
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            X_batch = None
            y_batch = None
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
