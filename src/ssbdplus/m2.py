import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

from tensorflow import keras
from tensorflow.python.ops.array_ops import concat_v2_eager_fallback


IMAGE_DIMS = (256, 256)


def get_movenet_data(path):
  cap = cv.VideoCapture(path)
  if not cap.isOpened():
    print("Error")
    return []

  frames = []
  vid_keypts = []

  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      img1 = frame.copy()
      img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 512, 682)
      frames.append(np.squeeze(img1))
      
      # reshape image
      img = frame.copy()
      img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
      input_image = tf.cast(img, dtype=tf.float32)

      # Input/Output
      input_details = estimator.get_input_details()
      output_details = estimator.get_output_details()

      # predict
      estimator.set_tensor(input_details[0]['index'], np.array(input_image))
      estimator.invoke()

      keypts = estimator.get_tensor(output_details[0]['index'])
      vid_keypts.append(keypts)

    else:
      break

  cap.release()
  return vid_keypts, frames



def frame_with_max_change(keypts):
  max_diff = 0
  max_loc = 0
  for frame in range(len(keypts) - 1):
    shaped1 = np.squeeze(keypts[frame])
    shaped2 = np.squeeze(keypts[frame + 1])

    diff_mag = np.linalg.norm(shaped2 - shaped1)

    if max_diff < diff_mag:
      max_diff = diff_mag
      max_loc = frame

  return max_diff, max_loc



def processed(keypts):
  proc_keypts = []

  for keypt in keypts:
    keypt = np.squeeze(keypt)[:, :2].flatten()
    # print(keypt.shape)
    proc_keypts.append(keypt)

  return proc_keypts



def get_best_frames(video_dir, test = False):
  best_frames = []
  set_label = np.array([1.0, 0.0, 0.0], dtype = np.float32)
  if 'headbanging' in video_dir and not test:
    set_label = np.array([0.0, 1.0, 0.0], dtype = np.float32)
  elif 'spinning' in video_dir and not test:
    set_label = np.array([0.0, 0.0, 1.0], dtype = np.float32)
  y = []
  all_keypts = []
  for video in os.listdir(video_dir):
    if 'noclass' in video:
      continue

    keypts, frames = get_movenet_data(video_dir + '/' + video)
    all_keypts.append(processed(keypts))

    max_diff, max_loc = frame_with_max_change(keypts)
    best_frames.append(frames[max_loc + 1])
    if not test:
      y.append(set_label)

    else:
      if 'armflapping' in video:
        y.append(np.array([1.0, 0.0, 0.0], dtype = np.float32))
      elif 'headbanging' in video:
        y.append(np.array([0.0, 1.0, 0.0], dtype = np.float32))
      elif 'spinning' in video:
        y.append(np.array([0.0, 0.0, 1.0], dtype = np.float32))

  return (best_frames, all_keypts), y



def create_model():
  base_model = tf.keras.applications.Xception(
    weights = 'imagenet',
    input_shape = (512, 682, 3),
    include_top = False,
    pooling = 'avg'
  )

  base_model.trainable = False

  frame_input = tf.keras.Input(shape = (512, 682, 3))
  joint_input = tf.keras.Input(shape = (40, 34,))

  x = base_model(frame_input, training = False)
  y = tf.keras.layers.LSTM(units = 4)(joint_input)
  combined = tf.keras.layers.Concatenate()([x, y])

  z = tf.keras.layers.UnitNormalization()(combined)
  z = tf.keras.layers.GaussianDropout(rate = 0.5, seed = 42)(z)
  z = tf.keras.layers.Dense(3)(z)
  z = tf.keras.layers.Softmax()(z)

  return tf.keras.Model(inputs = [frame_input, joint_input], outputs = z)



def train_model(model):
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1, amsgrad = True),
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False),
              metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  mcp_save = tf.keras.callbacks.ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')

  model.fit(x = [train_frames, train_keypts], y = train_y,
            epochs = 100, validation_data = ([test_frames, test_keypts], test_y),
            callbacks=[mcp_save], batch_size = 64, shuffle = True)



def load_model(model_pth):
  return tf.keras.models.load_model(model_pth)



def identify_action(model, test_frames, test_keypts):
  return model.predict([test_frames, test_keypts])

