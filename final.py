import os

import matplotlib.pyplot as plt
import numpy as np
from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from lime import lime_image as lime
from skimage.segmentation import mark_boundaries

model = inc_net.InceptionV3()


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, 0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


images = transform_img_fn([os.path.join('.', 'A.png')])
# additional manipulation to brighten up image
plt.imshow(images[0] / 2 + 0.5)
plt.title("Test Image")
plt.show()

preds = model.predict(images)
decoded_predictions = decode_predictions(preds)[0]
for x in decoded_predictions:
    print(x)

top_result = decoded_predictions[0][1]
second_result = decoded_predictions[4][1]

explanation = lime.LimeImageExplainer(kernel_width=.25).explain_instance(images[0].astype(
    'double'), model.predict, top_labels=5, hide_color=0, num_samples=1000, num_features=100000)

# explanation for the first classification
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
plt.title("Pixels that strongly affected classification of '" + top_result + "'")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=True)
plt.title("Heat map for classification of  '" + top_result + "'")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

# explanation for the second classification
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[4], positive_only=True, num_features=10, hide_rest=True)
plt.title("Pixels that strongly affected classification of: '" +
          second_result + "'")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[4], positive_only=False, num_features=10, hide_rest=True)
plt.title("Heat map for classification of '" + second_result + "'")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()
