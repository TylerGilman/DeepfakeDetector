import einops
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
import pathlib
from Video_handling import *
sns.set()



train_path = pathlib.Path('UCF101_subset/train')
#val_path = # set path to validation videos
test_path = pathlib.Path('UCF101_subset/test')





n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))


# Batch the data
train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_path, n_frames, training=True),
                                          output_signature = output_signature)


train_ds = train_ds.batch(batch_size)

# UNCOMMENT IF USING VALIDATION SET
"""
val_ds = tf.data.Dataset.from_generator(FrameGenerator(val_path, n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)
"""
test_ds = tf.data.Dataset.from_generator(FrameGenerator(test_path, n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)


height = 224
width = 224


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):

        super().__init__()
        self.seq = keras.Sequential([  
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                            kernel_size=(1, kernel_size[1], kernel_size[2]),
                            padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters, 
                            kernel_size=(kernel_size[0], 1, 1),
                            padding=padding)
            ])

    def call(self, x):
        return self.seq(x)

class ResidualMain(keras.layers.Layer):
    """
        Residual block of the model with convolution, layer normalization, and the
        activation function, ReLU.
    """
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, 
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    """
        Project certain dimensions of the tensor as the data is passed through different 
        sized filters and downsampled. 
    """
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)
    

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b = batch size, t = time, h = height, 
    # w = width, and c = number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos
  



##### BUILDING THE MODEL


input_shape = (None, 10, height, width, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(height // 2, width // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(height // 4, width // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(height // 8, width // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(height // 16, width // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))
x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(10)(x)

model = keras.Model(input, x)


#frames, label = next(iter(train_ds))

#model.build(frames)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
              metrics = ['accuracy'])



info = model.fit(x = train_ds,
                    epochs = 1, 
                    #validation_data = val_ds
)



def plot_history(info):
  """
    Plotting training loss and accuracy curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(info.history['loss'], label = 'train')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(info.history['loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train']) 

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(info.history['accuracy'],  label = 'train')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train'])
  plt.subplots_adjust(wspace=0.4,hspace=0.4)
  plt.show()

plot_history(info)


# evaluate on test set
test_metrics = model.evaluate(test_ds, return_dict=True)
print(test_metrics)