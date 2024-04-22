import pathlib
import sys

import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Version: ", sys.version)
print("Tensorflow version: ", tf.version.VERSION)

# Get dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

# Count the available pictures as ".JPG files
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Images: ', image_count)

# Get all roses
roses = list(data_dir.glob('roses/*'))
# Open the first 2 rose pictures
PIL.Image.open(str(roses[0]))
PIL.Image.open(str(roses[1]))

# Get all tulips
tulips = list(data_dir.glob('tulips/*'))
# Open the first 2 tulip pictures
PIL.Image.open(str(tulips[0]))
PIL.Image.open(str(tulips[1]))

# Set model variables
batch_size = 32
img_height = 180
img_width = 180

# Setup train data
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Setup validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Print the names of the classes
class_names = train_ds.class_names
print(class_names)

# add data augmentation to increase the amount of available data to train with
# Randomflip will mirror the image, Randomrotate will rotate the image and Randomzoom wil zoom in on the image
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

# Visualize/Plot the pictures
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print("Image Batch shape: ", image_batch.shape)
    print("Label Batch shape: ", labels_batch.shape)
    break

# Configure for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Create the model with sequential
model = tf.keras.Sequential([
    # data_augmentation,  # apply data augmentation
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Dropout(0.2),  # apply dropout
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, name="outputs")
])

# Optimize the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train the model for 15 epochs
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize the accuracy and loss results of the training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Create the plots
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('plot.png')
plt.show()

# Save the model to "data" folder
model.save('data/new_model.keras')
print("Model completed and saved to files")
model.summary()

# TODO: Improve the model since it sucks and always returns the same class(which can be very much wrong) with low confidence
#
# print("Commencing image test")
# # Apply an unknown image to the model and see the results
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#
# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
# new_model = keras.models.load_model('data/new_model.keras')
# new_model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# new_model.summary()
#
# predictions = new_model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
