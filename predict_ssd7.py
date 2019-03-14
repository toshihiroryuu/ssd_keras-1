from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

# Set the image size.
img_height = 300 
img_width = 480 
img_channels = 3 
subtract_mean = 127.5  
divide_by_stddev = 127.5 
n_classes=5
scales = [0.08, 0.16, 0.32, 0.64, 0.96] 
aspect_ratios = [0.5, 1.0, 2.0] 
two_boxes_for_ar1 = True 
steps = None 
offsets = None 
limit_boxes = False 
variances = [1.0, 1.0, 1.0, 1.0] 
coords = 'centroids' 
normalize_coords = False


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = '/home/ssd_keras/ssd7.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

#K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.


# We'll only load one image in this example.
img_path = '/home/Pictures/1478020975215232733.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

y_pred_decoded = decode_y2(y_pred,
                          confidence_thresh=0.5,
                          iou_threshold=0.4,
                          top_k='all',
                          input_coords='centroids',
                          normalize_coords=False,
                          img_height=None,
                          img_width=None)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('    class    conf  xmin     ymin   xmax    ymax')
print(y_pred_decoded[0])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes


plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

for box in y_pred_decoded[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[-4] 
    ymin = box[-3] 
    xmax = box[-2] 
    ymax = box[-1] 
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

plt.show()
