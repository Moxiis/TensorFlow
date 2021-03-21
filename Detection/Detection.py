import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

TRAINING_DIRECTORY = "C:\\Users\\Moxis\\Downloads\\BelgiumTSC_Training\\Training"
TESTING_DIRECTORY = "C:\\Users\\Moxis\\Downloads\\BelgiumTSC_Testing\\Testing"
SIZE = 100
EPOCHS = 5
LEARNING_RATE = 0.01

def Data_count(directory, format="jpg"):
	#ploting how many particular format files are in folders
	labels = []
	files = []
	count = -1
	for x in os.listdir(directory):
		count += 1
		if x.isdigit():
			labels.append(x[3:5])
			files.append(0)
			for y in os.listdir(os.path.join(directory, x)):
				if y.endswith(format):
					files[count] += 1
	assert len(labels) == len(files)

	plt.bar(labels, files)
	plt.xlabel("Label name")
	plt.ylabel("Images count")
	plt.xticks(rotation=90, fontsize=6)
	plt.show()

def Show_images(directory, format="jpg"):
	#showing every img in folder
	for folder in os.listdir(directory):
		if os.path.isdir(os.path.join(directory, folder)):
			for files in os.listdir(os.path.join(directory, folder)):
				if files.endswith(format):
					print(os.path.join(directory, folder, files))
					img = cv2.imread(os.path.join(directory, folder, files))
					cv2.imshow(files, img)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

def Images_pixels(directory, format="jpg"):
	#Ploting how many pixels have every file in folder
	pixels = []
	count = 0
	for folder in os.listdir(directory):
		if os.path.isdir(os.path.join(directory, folder)):
			for files in os.listdir(os.path.join(directory, folder)):
				if files.endswith(format):
					width, height = Image.open(os.path.join(directory, folder, files)).size
					pixels.append(width * height)
					count += 1
	pixels.sort(reverse=True)
	plt.figure(figsize=(14,8))
	plt.bar(np.arange(0,count), pixels)
	plt.xlabel("Image")
	plt.ylabel("Pixels")
	plt.show()

def File_converter(directory, old_format, new_format):
	#converting files from one format to another
	for folder in os.listdir(directory):
		if os.path.isdir(os.path.join(directory, folder)):
			for file in os.listdir(os.path.join(directory, folder)):
				if file.endswith(old_format):
					im = Image.open(os.path.join(directory, folder, file))
					im.save(os.path.join(directory, folder, file) + "." + new_format)
					os.remove(os.path.join(directory, folder, file))

training_ds = tf.keras.preprocessing.image_dataset_from_directory(TRAINING_DIRECTORY, labels="inferred", label_mode="int", batch_size=8, image_size=(SIZE, SIZE), shuffle=True, seed=1337)
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(TESTING_DIRECTORY, labels="inferred", label_mode="int", batch_size=8, image_size=(SIZE, SIZE), shuffle=False)
class_length = len(training_ds.class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
training_prefetcher = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_prefetcher = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(SIZE, SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(SIZE, SIZE, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(class_length, activation='softmax')
])

model.build(input_shape=(None, SIZE, SIZE, 3))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

t1 = time.time()
history = model.fit(
	training_prefetcher,
	epochs=EPOCHS,
	verbose=1,
	validation_data=testing_prefetcher
)
t2 = time.time()
print(t2-t1)