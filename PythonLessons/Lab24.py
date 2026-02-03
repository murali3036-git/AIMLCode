import tensorflow as tf
import numpy as np
x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_train = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=(1,), activation='linear'), # Hidden Layer
    tf.keras.layers.Dense(units=1)                                         # Output Layer
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=500, verbose=0)

pred = model.predict(np.array([[10.0]], dtype=np.float32), verbose=0)
print("Prediction for x=10:", pred[0][0])