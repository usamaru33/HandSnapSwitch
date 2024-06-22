import tensorflow as tf

# 学習させる(for)
def execution(num, model, optimizer, loss_fn, accuracy_fn, train_x, train_t):
    for i in range(1, num + 1):
        with tf.GradientTape() as tape:
            predictions = model(train_x, training=True)
            loss = loss_fn(train_t, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if i % 100 == 0:
            accuracy = accuracy_fn(train_t, predictions)
            print('...学習中...')
            print(f'Step: {i}, Loss: {loss.numpy()}, Accuracy: {accuracy.numpy()}')

def learning_algorithm(data_len):
    num_units1 = 200
    num_units2 = 200
    num_units3 = 200

    # モデルの定義
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(data_len,)),
        tf.keras.layers.Dense(num_units1, activation='tanh'),
        tf.keras.layers.Dense(num_units2, activation='tanh'),
        tf.keras.layers.Dense(num_units3, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 損失関数の定義
    def loss_fn(y_true, y_pred):
        return -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) +
                              (1 - y_true) * tf.math.log(1 - tf.clip_by_value(y_pred, 1e-10, 1.0)))

    # 精度関数の定義
    def accuracy_fn(y_true, y_pred):
        # データ型を明示的に一致させる
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        correct_prediction = tf.equal(tf.sign(y_pred - 0.5), tf.sign(y_true - 0.5))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # オプティマイザーの定義
    optimizer = tf.keras.optimizers.SGD(0.001)

    return model, optimizer, loss_fn, accuracy_fn