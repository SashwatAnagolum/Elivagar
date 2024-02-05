import numpy as np
import tensorflow as tf
import pennylane as qml

def mae_loss(pred, label):
    return tf.abs(pred - label)


def mse_loss(pred, label):
    return tf.abs(pred - label) ** 2


def mse_vec_loss(pred, label):
    diff_vec = tf.math.subtract(label, pred)
    return tf.reduce_sum(tf.math.multiply(diff_vec, diff_vec))

def log_loss_prob(label, probs):
    return -0.5 * ((1 + label) * tf.math.log(probs[0]) + (1 - label) * tf.math.log(probs[1]))


def log_loss_exp(label, exp):
    prob_0 = (exp + 1) / 2
    return -0.5 * ((1 + label) * tf.math.log(prob_0) + (1 - label) * tf.math.log(1 - prob_0))


def zero_prob_loss(pred, label):
    return pred[0]


def hinge_loss(pred, label):
    return 1 - pred * label


def cat_ce_loss(pred, label):
    probs = tf.math.divide(tf.math.exp(pred), tf.math.reduce_sum(tf.math.exp(pred)))
    ret = -1 * tf.math.reduce_sum(tf.math.multiply(tf.math.log(probs), label))

    return ret 


def cat_ce_loss_4(pred, label):
    pool = tf.math.unsorted_segment_mean(pred, tf.constant([0, 0, 1, 1]), num_segments=2)

    return cat_ce_loss(label, pool)


def compute_qnn_acc(circ, data_x, data_y, params):
    acc = 0
    preds = []

    for i in range(len(data_x)):
        pred = circ(data_x[i], params)[0]
        preds.append(pred)

    acc = np.sum(np.multiply(preds, data_y) > 0)

    return acc / len(data_x)


def train_qnn(circ, data_x, data_y, data_test_x, data_test_y, weights_shape, steps, lr, batch_size, loss_fn, verbosity=1, 
              loss_window=5, init_params=None, acc_thres=1.1, shuffle=True, print_loss=0):
    num_samples = len(data_x)
    num_samples_test = len(data_test_x)
    
    if shuffle:  
        ordering = np.random.permutation(num_samples)
        x = data_x[ordering]
        y = data_y[ordering]
    else:
        x = data_x
        y = data_y
        
    x = tf.constant(x)
    y = tf.constant(y)
    data_test_x = tf.constant(data_test_x)
    data_test_y = tf.constant(data_test_y)
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    weights = tf.Variable(init_params) if init_params is not None else tf.Variable(2 * np.pi * np.random.sample(weights_shape),
                                                                                   dtype=tf.float64)
    init_params = np.copy(weights.numpy())
    
    losses = [1e+10]
    accs = []
    acc_passed = False
    num_batches = 0
    grads_shape = tuple([1] + list(weights_shape))
    best_params = np.copy(weights.numpy())
    params_list = [init_params]
    val_acc = None

    for i in range(steps):
        if ((i + 1) * batch_size) % num_samples < (i * batch_size) % num_samples:
            batch_indices = [k for k in range((i * batch_size) % num_samples, num_samples)] + [
                k for k in range(batch_size - num_samples + (i * batch_size) % num_samples)]
        else:
            batch_indices = [k for k in range((i * batch_size) % num_samples, ((i + 1) * batch_size) % num_samples)]
        
        batch_loss = 0
        grads = tf.zeros(grads_shape, dtype=tf.float64)
        
        for j in range(batch_size):
            with tf.GradientTape() as tape:
                preds = circ(x[batch_indices[j]], weights)
                loss = loss_fn(y[batch_indices[j]], preds)

            sample_grads = tape.gradient(loss, [weights])
        
            grads += sample_grads

            batch_loss += loss               
        
        losses.append(batch_loss.numpy().item())
#         print(losses)
        if batch_loss < np.min(losses):
            best_params = weights.numpy().copy()
        
        opt.apply_gradients(zip(grads / batch_size, [weights]))
        
        params_list.append(np.copy(weights.numpy()))
        
        if verbosity and not i % verbosity and not i == 0:
            val_acc = compute_qnn_acc(circ, data_test_x, data_test_y, weights)
            accs.append(val_acc)
            
            print('Step {} | Sliding Loss Window : {} | Accuracy: {}'.format(
                i + 1, np.mean(losses[1:][-loss_window:]), val_acc))
        elif print_loss and not i % print_loss:
            print('Step {} | Sliding Loss Window : {}'.format(
                i + 1, np.mean(losses[1:][-loss_window:])))
            
        if val_acc and val_acc >= acc_thres:
            acc_passed = True
            num_batches = i
            break
        
    return losses[1:], acc_passed, accs, num_batches, best_params, params_list