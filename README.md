# keras_to_OpenCV_tensorflow
 This is a tool for transforming a model trained with Keras into a Tensorflow Protocol Buffers (.pb). And optimize it for using it with the module Dnn of OpenCV.

## How to use

Save your trained keras model to a .json. This will store the arquitecture of your model but not the weigths. You can use the following code for that:

```python
model_json = my_model.to_json()
with open('keras_model.json', "w") as json_file:
	json_file.write(model_json)

```

Also we will need to store the weigths, for this we will have to store in a .h5 file. Keras offer us a simpel way of doing this:`my_model.save_weights('keras_model.h5')`

Once we have both the model arquitecture (as a .json) and the wiegths (as a .h5), we will run the folliwing code on terminal: (Try `$ python keras_to_tensorflow_pb.py -h` for other input arguments.)


```
$ python keras_to_tensorflow_pb.py --input_model keras_model.json --input_weigths keras_model.h5 --output_dir result/ --output_name tensorflow_model.pb
```

This will generate a .pb version of our model that is optimized for inference that will be able to be use with OpenCV Dnn module. 

You can implement the model with OpenCV python as follows:

```python
import cv2

# load model
net = cv.dnn.readNetFromTensorflow(<path_to_.pb_file>)

# load a sample image
image = cv2.imread(<path_to_sample_image>)

# this will resize your sample image to input size that is required by your model (W, H) 
# and can perform a mean substraction of (mean0, mean1, mean2) for normalzing it.
blob = cv2.dnn.blobFromImage(image, 1, (W, H), (mean0, mean1, mean2))

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
preds = net.forward()

# sort the indexes of the probabilities in descending order (higher first) and get the first 
idx = np.argsort(preds[0])[0]

# print the prediction label and probability
print "Label: {}, {:.2f}%".format(idx, preds[0][idx] * 100)
```


## Required dependencies

- Keras
- Tensorflow
- h5py
- argparse


