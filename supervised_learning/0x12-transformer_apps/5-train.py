#!/usr/bin/env python3
"""Train transformer network"""


import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, dm, warmup_steps=4000):
    super(LRSchedule, self).__init__()
    
    self.dm = dm
    self.dm = tf.cast(self.dm, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train transformer network"""
    print("start")
    dataset = Dataset(batch_size, max_len)
    in_vocab = dataset.tokenizer_pt.vocab_size + 2
    out_vocab = dataset.tokenizer_en.vocab_size + 2
    transformer = Transformer(N, dm, h, hidden, in_vocab, out_vocab,
                              max_len, max_len)
    optimizer = tf.keras.optimizers.Adam(LRSchedule(dm), beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy
    loss_object = loss_object(from_logits=True, reduction='none')
    def loss_f(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
  
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    train_loss = tf.keras.metrics.Mean(name='loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy
    train_acc = train_acc(name='accuracy')
    #tf.compat.v1.enable_eager_execution()
    #for i in dataset.data_train.take(1):
    #    print(i)

    for epoch in range(epochs):
        batch = 0
        for (input, target) in dataset.data_train:
            target_input = target[:, :-1]
            target_real = target[:, 1:]
            enc_mask, look_mask, dec_mask = create_masks(input, target_input)
            with tf.GradientTape() as tape:
                predictions = transformer(input, target_input, True, enc_mask,
                                          look_mask, dec_mask)
                loss = loss_f(target_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_acc(target_real, predictions)

            if not batch % 10:
                print ('Epoch {}, Batch {}: Loss {:} Accuracy {:}'.format(
                       epoch + 1, batch, train_loss.result(), train_acc.result()))
            batch += 1
        batch -= 1
        print('Epoch {}: Loss {:} Accuracy {:}'.format(
                  epoch + 1, batch, train_loss.result(), train_acc.result()))
    return transformer
