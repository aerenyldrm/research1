import tensorflow as t
from tensorflow.lite.python.lite_constants import KERAS
from tensorflow.python.keras import layers, models
import keras
import matplotlib as plot
import os
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.layers.normalization import normalization

# 1 load dataset
train_directory = r"C:\Users\aeren\Downloads\jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\train"
test_directory =r"C:\Users\aeren\Downloads\jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\test"

# 2 set image and batch size
image_size = (640, 640)
batch_size = 32

# 3 construct train and validation datasets
train_dataset = keras.utils.image_dataset_from_directory(
    train_directory,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = keras.utils.image_dataset_from_directory(
    test_directory,
    image_size=image_size,
    batch_size=batch_size
)

# 4 check class names
class_names = sorted(entry.name for entry in os.scandir(train_directory) if entry.is_dir())
print("ClASS NAME:", class_names)


