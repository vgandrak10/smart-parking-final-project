#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import cv2 as cv
import pathlib
import tensorflow as tf
from absl import flags

flags.DEFINE_string('ip_addr', "0.0.0.0", 'System IP Address')

FLAGS = flags.FLAGS

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = '/home/sai/Desktop/vihari_workspace/exported_model' # download_model(MODEL_NAME, MODEL_DATE)


LABEL_FILENAME = 'occupied-and-vacant-parking-spac_label_map.pbtxt'
PATH_TO_LABELS = '/home/sai/Desktop/vihari_workspace/parking_detection_dataset.v1-416x416.tfrecord/test/occupied-and-vacant-parking-spac_label_map.pbtxt' # download_labels(LABEL_FILENAME)

# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings



#########################################################
# Objct Detection Function
########################################################

def my_object_detection(image_np):

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # print("print result: ", detections)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)


    return image_np_with_detections



##########################################################
# MQTT Code
##########################################################

import base64
import cv2 as cv
import numpy as np
import paho.mqtt.client as mqtt


from flask import Flask, render_template, Response
app = Flask(__name__)

MQTT_BROKER = "192.168.0.31" #"IP Address of the Broker"
MQTT_RECEIVE = "home/server"

frame = np.zeros((240, 320, 3), np.uint8)



# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_RECEIVE)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global frame
    # Decoding the message
    img = base64.b64decode(msg.payload)
    # converting into numpy array from buffer
    npimg = np.frombuffer(img, dtype=np.uint8)
    # Decode to Original Frame
    mqtt_frame = cv.imdecode(npimg, 1)

    # Call Object detection on this frame
    frame = my_object_detection(mqtt_frame)

    # For flask
    # ret, buffer = cv.imencode('.jpg', frame)
    # frame = buffer.tobytes() 
    # yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



################################
# Flask Functions

def gen_frames():  # generate frame by frame from camera
    global frame
    
    while True:
        
        if type(frame) == type(np.zeros((240, 320, 3))) :
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes() 
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    print("Rferesh 1")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(on_message, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    print("refresh2")
    return render_template('index.html')

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("at 1")
client.connect(MQTT_BROKER)
print("at 2")
# Starting thread which will receive the frames
client.loop_start()

# app.run(host="192.168.0.31", port=5000, debug=True)
app.run(host=FLAGS.ip_addr, port=5000, debug=True)

# Stop the Thread
client.loop_stop()