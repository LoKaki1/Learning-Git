import tensorflow as tf
from tensorflow import keras
from ClassifierStocksAI import build_model_for_multiple_prediction, generate_fit_and_prepare_data
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
TICKER = 'TSLA'
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
checkpoint_path = "training_1/cp.ckpt"


def create_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def secondary_main(ticker='NIO'):
    new_model = tf.keras.models.load_model(f'saved_model/{ticker}_model')
    # Check its architecture
    new_model.summary()
    epochs, units, prediction_days, prediction_day, \
        scalar, scaled_data, \
        x_train, y_train = generate_fit_and_prepare_data(ticker,
                                                         None, None, None, None, None, None, True)
    model_inputs = scaled_data
    real_data = [model_inputs[len(model_inputs) -
                              prediction_days * 4: len(model_inputs) + prediction_day, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    t = new_model.predict(real_data)
    print(t)
    print((p := scalar.inverse_transform(np.array([t]).reshape(-1, 1))), len(p))


if __name__ == '__main__':
    secondary_main()
