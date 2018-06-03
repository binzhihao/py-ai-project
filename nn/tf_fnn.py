import tensorflow as tf
import numpy as np
import data.mnist.loader as loader

if __name__ == '__main__':

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    learning_rate = 0.01
    n_epochs = 40
    batch_size = 50

    def neuron_layer(X, n_neurons, name, activation=None):
        """
        :param X: input
        :param n_neurons: number of neurons in this layer
        :param name: name of this layer
        :param activation: activation function
        :return:
        """
        with tf.name_scope(name):
            n_inputs_ = int(X.get_shape()[1])

            # 截断的正态分布(truncated_normal)随机初始化权重，缓解梯度消失和梯度爆炸的策略之一
            # 这里使用2 / np.sqrt(n_inputs_)作为方差
            stddev_ = 2 / np.sqrt(n_inputs_)
            init_ = tf.truncated_normal((n_inputs_, n_neurons), stddev=stddev_)

            W = tf.Variable(init_, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b

            if activation is not None:
                return activation(Z)
            else:
                return Z

    def neuron_layer_dense(X, n_neurons, name, activation=None):
        # 新增的方法
        if activation is None:
            return tf.layers.dense(X, n_neurons, name=name)
        else:
            return tf.layers.dense(X, n_neurons, name=name, activation=activation)

    def shuffle_batch(X, y, size):
        rnd_idx_ = np.random.permutation(len(X))
        n_batches_ = len(X) // size
        for batch_idx_ in np.array_split(rnd_idx_, n_batches_):
            X_batch_, y_batch_ = X[batch_idx_], y[batch_idx_]
            yield X_batch_, y_batch_

    def leaky_relu(z, name=None):
        return tf.maximum(0.01 * z, z, name=name)

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
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, name="y")

    with tf.name_scope("fnn"):
        # hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        # hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        # logits = neuron_layer(hidden2, n_outputs, name="outputs")

        # 一般地，效果上 ELU > leaky ReLU > ReLU > tanh > sigmoid
        # 前三个都是非饱和激活函数，缓解梯度消失和梯度爆炸的策略之一
        # tf 定义了 tf.nn.elu, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, 但没有定义 leaky ReLU
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # create variables initializer
    init = tf.global_variables_initializer()
    # create a saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # init
        init.run()
        # train
        for epoch in range(n_epochs):
            X_batch = None
            y_batch = None
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
        # save
        # save_path = saver.save(sess, "./my_model_final.ckpt")
