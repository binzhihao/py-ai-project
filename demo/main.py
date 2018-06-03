import tensorflow as tf

if __name__ == '__main__':
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f1 = x * x * y + y + 2
    f2 = x + y

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        result1, result2 = sess.run([f1, f2])
        print(result1)
        print(result2)
