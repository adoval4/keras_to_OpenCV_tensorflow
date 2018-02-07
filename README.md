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

Once we have both the model arquitecture (as a .json) and the wiegths (as a .h5), we will run the folliwing code on terminal:


```
$ python keras_to_tensorflow_pb.py --input_model keras_model.json --input_weigths keras_model.h5 --output_dir result/ --output_name tensorflow_model.pb
```

Try `$ python keras_to_tensorflow_pb.py -h` for other input arguments.

## Required dependencies

- Keras
- Tensorflow
- h5py
- argparse


