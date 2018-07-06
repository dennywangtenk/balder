# coding: utf-8

# #
# In[ ]:
## v1

import datetime
import os
import sys
from timeit import default_timer as timer
import logging
import numpy as np
import tensorflow as tf
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

SCRIPT_NAME = 'Test Object Detection Round PK V1'

print(SCRIPT_NAME)

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# # Model preparation
##Initial Log
root = logging.getLogger()
root.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_GRAPH = '/home/denny/run_ai/test_ai/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/denny/run_ai/test_ai/label_map.pbtxt'

NUM_CLASSES = 2

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
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

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[ ]:
input_images = []
input_image_filenames = []
output_dir = '/home/denny/run_ai/test_ai/output'
img_dir = '/home/denny/run_ai/test_ai/images'
valid_images = [".jpg", ".gif", ".png"]

for f in os.listdir(img_dir):
    fn = os.path.splitext(f)[0]
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    input_image_filenames.append(fn)
    logging.debug("Loading {}...".format(fn))

    image_filename = os.path.join(img_dir, f)
    image = Image.open(image_filename)
    image_np = load_image_into_numpy_array(image)
    input_images.append(image_np)
    logging.debug("  Image:{} loaded...".format(fn))

now = datetime.datetime.now()
start = timer()
output_dict_array = run_inference_for_images(input_images, detection_graph)
end = timer()
avg = (end - start) / len(input_images)

print("===TF inference took: " + str(end - start) + " for [" + str(len(input_images)) + "] images, average[" + str(
    avg) + "]===")

print("output array has:" + str(len(output_dict_array)))

for idx in range(len(output_dict_array)):
    output_dict = output_dict_array[idx]
    image_np_org = input_images[idx]
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_org,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=3)

    img_out = Image.fromarray(image_np_org, 'RGB')
    img_out.save(os.path.join(output_dir, 'output_image_{}.jpg'.format(input_image_filenames[idx])))

