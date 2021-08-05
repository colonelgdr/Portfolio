import os 
import tensorflow as tf
from tensorflow import keras as Keras

os.environ["TF_CPP_LOG_LEVEL"] = "2"

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Image information and path to dataset 
FRAME_SIZE = (224, 224, 3)
BATCH_SIZE = 32
DATASET_PATH = "C:\\Users\\gdrpc\\Google Drive\\University\\Capstone\\Lechuga"

# Get folder 
dataDir = Keras.utils.get_file(    
    fname = DATASET_PATH, 
    origin=None,
    )
# Create training set
trainDataset = Keras.preprocessing.image_dataset_from_directory(
    dataDir, 
    validation_split = 0.7, 
    subset = "training",
    seed = 123,
    image_size = (FRAME_SIZE[0], FRAME_SIZE[1]), 
    batch_size = BATCH_SIZE
    )
# Create validation set 
validationDataset = Keras.preprocessing.image_dataset_from_directory(
    dataDir,
    validation_split = 0.3,
    subset = "validation",
    seed = 123, 
    image_size = (FRAME_SIZE[0], FRAME_SIZE[1]),
    batch_size = BATCH_SIZE
    )
# Data Augmentation 
datagen = Keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=4,
    zoom_range=(0.4, 0.5),
    horizontal_flip=True,
    vertical_flip=False,
    data_format='channels_last',
    validation_split=0.7,
    dtype=tf.float32
    )
trainGen = datagen.flow_from_directory(
    dataDir, 
    target_size=(224, 224),
    class_mode='sparse', 
    shuffle=True, 
    subset='training',
    seed=123
    )
validationGen = datagen.flow_from_directory(
    dataDir, 
    target_size=(224, 224),
    class_mode='sparse', 
    shuffle=True, 
    subset='validation',
    seed=123
    )
# Model declaration an definition 
model = Keras.applications.DenseNet121(
    include_top=True, weights=None, input_tensor=None,
    input_shape=FRAME_SIZE, pooling=None, classes=4)
model.compile(
    optimizer="adam",
    loss=[Keras.losses.SparseCategoricalCrossentropy(from_logits=True)], 
    metrics='accuracy', 
    loss_weights=None, 
    weighted_metrics=None,
    run_eagerly=None, 
    steps_per_execution=None
    )
# Model Training
model.fit(
    x=trainGen, 
    #y=validationGen,
    epochs=10, 
    steps_per_epoch=2, 
    verbose=2
    )
model.save("C:/Users/gdrpc/Google Drive/University/Capstone/Neural Networks/LettuceNet_224_v2.h5")

# Convert to tflite model 
