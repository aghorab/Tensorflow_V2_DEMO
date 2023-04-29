# import tensorflow as tf;
# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))
tf.config.set_visible_devices([], 'GPU')  # disable GPU
print(tf.__version__)


a = 2
b = 3
c = tf.add(a, b, name='Add')
print(c)  # TF 2.0 supports eager execution which means you don't have to explicitly




# create a session and run the code in it.

tf.compat.v1.disable_eager_execution()
a = tf.constant(2, name='A')
b = tf.constant(3, name='B')

c = tf.add(a, b, name='Add')

# sess = tf.Session() # not allowed in tf v2, you should use
sess = tf.compat.v1.Session()
print(sess.run(c))
sess.close()


m1 = tf.compat.v1.get_variable('m1', initializer=tf.constant([[0, 1], [2, 3]]))
m2 = tf.compat.v1.get_variable('m2', initializer=tf.constant([[3, 4], [5, 6]]))
sum_matrix = tf.add(m1, m2, name="Add")

# Add an Op to initialize variables
init_op = tf.compat.v1.global_variables_initializer()
# launch the graph in a session
with tf.compat.v1.Session() as sess:
    # run the variable initializer
    sess.run(init_op)
    # now we can run the desired operation
    print(sess.run(sum_matrix))
