import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

class data_container():
    def __init__(self, X, y):
        self.X = X.astype(np.float32) # Pandas datatype
        self.y = y.astype(np.float32) # Pandas datatype

    def create_random_train_test(self, test_size=0.25, seed=None):
        # Method for created random training, testing data
        if seed:
            X_train, X_test, y_train, y_test = \
                    train_test_split(self.X, self.y,
                            test_size=test_size, random_state=seed)
        else:
            X_train, X_test, y_train, y_test = \
                    train_test_split(self.X, self.y,
                            test_size=test_size)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_train_test_by_end_date(self, train_end_date):
        self.X_train = self.X.loc[:train_end_date]
        self.X_test = self.X.loc[train_end_date:]
        self.y_train = self.y.loc[:train_end_date]
        self.y_test = self.y.loc[train_end_date:]

    def one_hot_encode_y(self):
        ''' Intended to be used to predict the direction of returns
        movements, so y is expected to be returns. This will
        one hot encode the sign of y, producing an n83 array
        (down, no change, up). Only intended to be used for one asset.'''
        assert self.y.shape[1] < 2
        enc = OneHotEncoder()
        class_creator = lambda x: np.sign(x).values + 1
        y_train_sign = class_creator(self.y_train)
        y_test_sign = class_creator(self.y_test)
        enc.fit(y_train_sign)
        self.y_train_class = enc.transform(y_train_sign)
        self.y_test_class = enc.transform(y_test_sign)

def tf_classifier(data, hidden_layer_size):
    pass

def tf_regressor(data, hidden_layer_size, hidden_drop_keep_probs, learning_rate=0.01):
    num_outs = data.y.shape[1]
    num_ins = data.x.shape[1]

    weights_list = []
    bias_list = []

    # Build the graph
    w_0 = tf.Variable(tf.truncated_normal([num_ins, hidden_layer_size[0]], \
            stddev = 1.0/np.sqrt(num_ins), name="Weight0"))
    weights_list.append(w_0)

    b_0 = tf.Variable(tf.truncaded_normal([hidden_layer_size[0]], \
            stddev=1.0/np.sqrt(hidden_layer_size[0]), name="Bias0"))
    bias_list.append(b_0)

    for w in range(1, len(hidden_layer_size)):
        this_in = hidden_layer_size[w-1]
        this_out = hidden_layer_size[w]
        w_n = tf.Variable(tf.truncated_normal([this_in, this_out], \
                stddev = 1.0/np.sqrt(this_in), name="Weight"+str(w)))
        weights_list.append(w_n)

        b_n = tf.Variable(tf.truncaded_normal([this_in,], \
                stddev=1.0/np.sqrt(this_out), name="Bias"+str(w)))
        bias_list.append(b_n)

    # now for ourput layer
    W_final = tf.Variable(tf.truncated_normal([hidden_layer_size[-1], num_outs], \
            stddev = 1.0/np.sqrt(num_ins), name="WeightFinal"))
    weights_list.append(W_final)

    b_final = tf.Variable(tf.truncaded_normal([num_outs,], \
            stddev=1.0/np.sqrt(num_outs), name="BiasFinal"))
    bias_list.append(b_final)

    # now perform operations
    assert len(hidden_layer_size) == len(hidden_drop_keep_probs)

    X = tf.placeholder("float")
    y = tf.placeholder("float")

    hidden = tf.nn.relu(tf.add(tf.matmul(X, weights_list[0]), bias_list[0]))
    hidden = tf.nn.dropout(hidden, keep_prob=hidden_drop_keep_probs[0])

    for w in range(1, len(hidden_layer_size)):
        hidden = tf.nn.relu(tf.add(tf.matmul(hidden, weights_list[w]), bias_list[w]))
        hidden = tf.nn.dropout(hidden, keep_prob=hidden_drop_keep_probs[w])

    # now output
    y_model = tf.nn.relu(tf.add(tf.matmul(hidden, weights_list[-1]), bias_list[-1]))

    loss = tf.reduce_mean(tf.square(y - y_model) * 1e4)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.initialize_all_variables()
