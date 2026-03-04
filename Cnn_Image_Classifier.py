import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image 
import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)

#Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshape for CNN
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

#One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

cnn_model = Sequential()

cnn_model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnn_model.add(MaxPooling2D((2,2)))

cnn_model.add(Conv2D(64,(3,3),activation='relu'))
cnn_model.add(MaxPooling2D((2,2)))

cnn_model.add(Flatten())

cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(10,activation='softmax'))

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train CNN
history_cnn = cnn_model.fit(
    datagen.flow(X_train,y_train,batch_size=64),
    epochs=10,
    validation_data=(X_test,y_test)
)

import os

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Accuracy
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])

plt.title("CNN Accuracy")
plt.legend(["Train","Validation"])

plt.savefig("plots/cnn_accuracy.png")
plt.show()

# Loss
plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])

plt.title("CNN Loss")

plt.savefig("plots/cnn_loss.png")
plt.show()

# Using MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Resize images
import tensorflow as tf

X_train_resized = tf.image.resize(X_train,(96,96))
X_test_resized = tf.image.resize(X_test,(96,96))

# Load Pretrained Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

# Freeze layers:
for layer in base_model.layers:
    layer.trainable=False

    x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)

model_transfer = Model(inputs=base_model.input, outputs=predictions)

model_transfer.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

import tensorflow as tf

# Resize images for MobileNetV2
X_train_resized = tf.image.resize(X_train, (96,96))
X_test_resized = tf.image.resize(X_test, (96,96))

# Ensure the last dimension is 1 (grayscale) before converting to RGB
if len(X_train_resized.shape) == 3:
    X_train_resized = tf.expand_dims(X_train_resized, axis=-1)
if len(X_test_resized.shape) == 3:
    X_test_resized = tf.expand_dims(X_test_resized, axis=-1)

# Replicate the single channel to three channels for the MobileNetV2 input
X_train_resized_rgb = tf.image.grayscale_to_rgb(X_train_resized)
X_test_resized_rgb = tf.image.grayscale_to_rgb(X_test_resized)

# Explicitly build the model to avoid ValueError during tf.function tracing
_ = model_transfer(X_train_resized_rgb[:1])

history_transfer = model_transfer.fit(
    X_train_resized_rgb,
    y_train,
    epochs=5,
    validation_data=(X_test_resized_rgb,y_test)
)

# Predictions:
y_pred = cnn_model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

y_true = np.argmax(y_test,axis=1)


# Confusion Matrix
cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm,annot=True,cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
plt.show()

# Classification Report
print(classification_report(y_true,y_pred))