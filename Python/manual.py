import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from keras import backend as K
import scipy.stats as sps


d_degree = 12

q_level_loss = np.arange(0.01, 1, 0.01)
B = np.zeros((d_degree+1, 99))
for d in range(d_degree+1):
    B[d, :] = sps.binom.pmf(d, d_degree, q_level_loss)

def qt_loss(y_true, y_pred):
    # Quantiles calculated via basis and increments
    B_tensor = K.constant(B, shape=(d_degree+1, 99), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_pred = tf.reshape(y_pred, (-1, y_true.shape[1], d_degree+1))

    q = K.dot(K.cumsum(y_pred, axis=2), B_tensor)
    y_true = tf.expand_dims(y_true, axis=2)

    # Calculate CRPS
    err = tf.subtract(y_true, q)

    e1 = err * tf.constant(q_level_loss, shape=(1, 99), dtype=tf.float32)
    e2 = err * tf.constant(q_level_loss - 1, shape=(1, 99), dtype=tf.float32)

    scores = tf.maximum(e1, e2)
    scores = tf.reduce_mean(scores, axis=2)
    scores = tf.reduce_mean(scores, axis=1)
    return scores

#y_true_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 24))
#y_pred_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 24 * (d_degree+1)))

#with tf.compat.v1.Session() as sess:
#    # Testdaten generieren
#    y_true_data = np.random.rand(32, 24)
#    y_pred_data = np.random.rand(32, 24 * (d_degree+1))

    # Führen Sie die Loss-Funktion in der Session aus
#    loss_value = sess.run(qt_loss(y_true_placeholder, y_pred_placeholder),
#                          feed_dict={y_true_placeholder: y_true_data, y_pred_placeholder: y_pred_data})

#    print("Loss Value:", loss_value)

qt_loss(tf.Variable(np.random.rand(32, 24), dtype=tf.float32), tf.Variable(np.random.rand(32, 24 * (d_degree+1)), dtype=tf.float32))