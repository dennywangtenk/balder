# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# #
# In[ ]:
## v4

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2
import datetime
from timeit import default_timer as timer


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


SCRIPT_NAME = 'Test Object Detection Test AI V4'

print(SCRIPT_NAME)


from utils import label_map_util

from utils import visualization_utils as vis_util

# # Model preparation

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/denny/run_ai/test_ai/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/denny/run_ai/test_ai/label_map.pbtxt'

NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection
def run_inference_for_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []

            for image in images:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                output_dict_array.append(output_dict)

    return output_dict_array



FILE_INPUT = '/home/denny/run_ai/test_ai/test.mp4'

cap = cv2.VideoCapture(FILE_INPUT)

FILE_OUTPUT = '/home/denny/run_ai/test_ai/output_test.mp4'

BATCH_SIZE = 100

# Define the codec and create VideoWriter object
ret, frame = cap.read()

print('OpenCV version:',cv2.__version__,'ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])

FPS= 25.0
FrameSize=(frame.shape[1], frame.shape[0]) # MUST set or not thing happen !!!!
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT, fourcc, FPS, FrameSize)

frameCount = 1
images = []

while (cap.isOpened() and frameCount <= 10000):

    ret, image_np = cap.read()

    if ret == True:
        frameCount = frameCount + 1

        if frameCount < 300: ##skip a few seconds
            continue

        ##cv2.imshow('Input: ', image_np)
        images.append(image_np)

        if frameCount % BATCH_SIZE == 0:
             now = datetime.datetime.now()
             print(str(now) + " : count is : " + str(frameCount))
             start = timer()
             output_dict_array = run_inference_for_images(images,detection_graph)
             end = timer()
             avg = (end - start) / len(images)

             print("TF inferencing took: "+str(end - start) +" for ["+str(len(images))+"] images, average["+str(avg)+"]")

            ## print("output array has:" + str(len(output_dict_array)))

             for idx in range(len(output_dict_array)):
                 output_dict = output_dict_array[idx]
                 image_np_org = images[idx]
                 vis_util.visualize_boxes_and_labels_on_image_array(
                     image_np_org,
                     output_dict['detection_boxes'],
                     output_dict['detection_classes'],
                     output_dict['detection_scores'],
                     category_index,
                     instance_masks=output_dict.get('detection_masks'),
                     use_normalized_coordinates=True,
                     line_thickness=6)

                 out.write(image_np_org)
                 ##cv2.imshow('object image', image_np_org)

             del output_dict_array[:]
             del images[:]

       
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
