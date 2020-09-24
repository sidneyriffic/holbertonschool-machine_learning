#!/usr/bin/env python3
"""Forecast BTC prices 1 hour out"""


import numpy as np
from tensorflow.data import Dataset
import tensorflow as tf
keras = tf.keras
tf.enable_eager_execution()

"""
Data columns: gain/loss ratio, btc volume
Data labels: real dollar values (not really used for training)
Data prices: real prices we'll be multiplying our output by to reduce
effect of stable price changing over time changing our prediction.
Data ratios: ratios between last input and prediction prices, for training.
"""
data = np.load("data.npy")
datalabels = np.load("datalabels.npy")
dataprices = np.load("dataprices.npy")
dataratios = np.load("dataratios.npy")
data_len = data.shape[0]
data_features = data.shape[1]
print(data[0])
print(data[1500])
print(dataprices[0], dataratios[0])
print("features:", data_features)
"""
Slice off last 3 months on bitstamp for test and validation.
We'll use two months for validation and one for testing.
It is the most recent data and will be most analagous to making
future predictions given we're not collecting more data.
Since we have millions of data points and most of them are not recent
(thus possibly not taking into account current trends) taking this
relatively small slice should be fine.
"""
combine = 5  # should match param from preprocess
window = int(1440 / combine)
print(data.shape[0])
data = Dataset.from_tensor_slices(data).window(window, 1, combine, True)
data = data.flat_map(lambda x: x.batch(window, drop_remainder=True))
print(data)
labels = Dataset.from_tensor_slices(datalabels)
prices = Dataset.from_tensor_slices(dataprices)
ratios = Dataset.from_tensor_slices(dataratios)
ins = Dataset.zip((data, prices))
outs = Dataset.zip((ratios, prices))
data = Dataset.zip((ins, outs))
"""
Choose a window stride number for training that does not have a common factor
with minutes in a day (1440) so we can pinstripe through every day while going
through the whole timescale, but also still pick many per day. I picked 29.
Repeat is done before our pinstriping window so it will roll over the end and
pinstripe the year as well as long as it not a factor in our data size either
(which it is not for the bitstamp set nor the slightly truncated one due to
the first 1440 size window).

We'll split off the last month for testing and the two months before that for
validation.
"""
val_test_count = 8760 * 3
val_start = int(data_len - val_test_count)
win_shift = 29
assert(1440 % win_shift)
print("taking {} for training".format(val_start))
train = data.take(val_start).repeat()
train = train.window(1, win_shift, 1)
train = train.flat_map(lambda x, y: Dataset.zip((x, y)))
print(data_len)
print("skipping {} for validation".format(val_start + 1440 / combine))
val = data.skip(int(val_start + 1440 / combine))
test = data.skip(int(data_len - 8760 + 1440 / combine))

print(train)

batch_size = 32
batch_per_datalen = int(val_start / batch_size / win_shift)

inputdata = keras.Input(shape=(window, data_features))
inputprice = keras.Input(shape=(1,))
rnn = keras.layers.GRU(100)(inputdata)
dense = keras.layers.Dense(800)(rnn)
dense = keras.layers.Dense(400)(dense)
dense = keras.layers.Dense(200)(dense)
dense = keras.layers.Dense(100)(dense)
dense = keras.layers.Dense(50)(dense)
dense = keras.layers.Dense(25)(dense)
dense = keras.layers.Dense(1)(dense)

pred_ratio = keras.layers.multiply([dense, tf.constant([60.0 / combine])],
                                   name='ratio')
pricescale = keras.layers.multiply([pred_ratio, inputprice], name='price')
model = keras.Model([inputdata, inputprice], [pred_ratio, pricescale])


def no_loss(y_true, y_pred):
    """
    Using this to dummy out price loss for training. TF dataset kind of
    inflexible about this.
    """
    return 0.0


if True:
    name = 'ratio'
    weights_file = './ratioloss'
    losses = ['mse', no_loss]
    metrics = {'price': 'mse'}
else:
    name = 'price'
    weights_file = './priceloss'
    losses = [no_loss, 'mse']
    metrics = {'tf_op_layer_mul': 'mse'}
model.summary()
optimizer = keras.optimizers.Adam()
model.compile(optimizer, losses, metrics)
train = train.batch(batch_size).prefetch(2)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iter(train),
                           optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, './', max_to_keep=1,
                                     checkpoint_name=name)
ckpt.restore(manager.latest_checkpoint)
print(ckpt.step)


class CkptWithTFDS(keras.callbacks.Callback):
    """
    Create a checkpoint that includes Tensorflow Dataset iterator
    state so we can pick up where we left off next time and preserve
    pinstriping the data differently every epoch.
    """
    def __init__(self, ckpt):
        self.manager = manager

    def on_train_batch_begin(self, batch, logs=None):
        """We'll call our checkpoint every 10 batches."""
        if not batch % 10:
            manager.save()


if 0:
    checkpoint = CkptWithTFDS(ckpt)
    print("Fitting")

    model.fit(train, epochs=int(batch_per_datalen / 50 * 29 + 1),
              steps_per_epoch=50, callbacks=[checkpoint])
    manager.save()


if 1:
    print("Doing validation")
    # Just go until we run out of data
    val = val.batch(batch_size).prefetch(2)
    model.evaluate(val)
