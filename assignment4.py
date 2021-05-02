import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.models import Model

# constants
CLASSES = 1000
HEIGHT = 224
WIDTH = 224
EPOCHS = 1
INPUT_SHAPE = (HEIGHT, WIDTH, 3)

# image data generators
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.3)
train_data = '/Users/alureon/imagenette2-160/train'
val_data = "/Users/alureon/imagenette2-160/val"
train_gen = data_gen.flow_from_directory(
    train_data,
    target_size=(HEIGHT, WIDTH),
    batch_size=20,
    subset='training'
)
val_gen = data_gen.flow_from_directory(
    train_data,
    target_size=(HEIGHT, WIDTH),
    batch_size=20,
    subset='validation'
)
test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    val_data,
    target_size=(HEIGHT, WIDTH),
    shuffle=False,
    batch_size=20
)


# adds a senet block after a given input layer - returns a tensor
def add_senet_layers(input_layer):
    if isinstance(input_layer, tf.keras.layers.InputLayer):
        ll = input_layer.output_shape[0][3]
    else:
        ll = input_layer.output_shape[3]
    x = tf.keras.layers.GlobalAveragePooling2D()(input_layer.output)
    x = tf.keras.layers.Reshape((1, 1, ll))(x)
    x = tf.keras.layers.Dense(ll // 16, activation='relu')(x)
    x = tf.keras.layers.Dense(ll, activation='sigmoid')(x)
    return tf.keras.layers.multiply([input_layer.output, x])


# utility method from StackOverflow for acquiring layer index by name
def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


vgg = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=INPUT_SHAPE,
    pooling=None,
    classes=CLASSES,
    classifier_activation="softmax",
)
vgg.summary()
vgg.trainable = False

# modify output layer
default_output = vgg.output
z = tf.keras.layers.Flatten()(default_output)
z = tf.keras.layers.Dense(256, activation='relu')(z)
z = tf.keras.layers.Dense(10, activation='softmax')(z)
vgg = tf.keras.Model(vgg.input, z)

# initialize history arrays
train_loss_history = []
val_loss_history = []

# Get loss metrics with no attention module
vgg.compile(loss='binary_crossentropy', metrics=['acc'])
vgg.trainable = False
history = vgg.fit(train_gen,
                  epochs=EPOCHS,
                  validation_data=val_gen)
vgg.summary()
train_loss_history.append(history.history['loss'])
val_loss_history.append(history.history['val_loss'])

layers_to_insert_senet_after = ['input_1',
                                'block1_pool',
                                'block2_pool',
                                'block3_pool',
                                'block4_pool',
                                'block5_pool']


def insert_new_senet_block(model: Model, layer: Layer) -> Model:
    print("INSERTING " + layer)

    vgg_layers = [i for i in model.layers]
    target_index = getLayerIndexByName(model, layer)
    target_layer = model.get_layer(index=target_index)
    x = tf.keras.Input(shape=INPUT_SHAPE)

    for i in range(1, target_index - 1):
        if isinstance(model.get_layer(index=i), tf.keras.layers.Multiply):
            x = vgg_layers[i]([model.get_layer(index=i-5).output, x])
        else:
            x = vgg_layers[i](x)

    if (isinstance(x, KerasTensor)):
        x = add_senet_layers(target_layer)
    else:
        x = add_senet_layers(target_layer)(x)

    for i in range(target_index+1, len(model.layers)):
        x = vgg_layers[i](x)

    model = tf.keras.Model(inputs=model.input, outputs=x)
    model.trainable = False
    return model


for layer in layers_to_insert_senet_after:
    vgg = insert_new_senet_block(vgg, layer)
    vgg.compile(loss='binary_crossentropy', metrics=['acc'])
    vgg.trainable = False
    history = vgg.fit(train_gen,
                      epochs=EPOCHS,
                      validation_data=val_gen)
    vgg.summary()
    train_loss_history.append(history.history['loss'])
    val_loss_history.append(history.history['val_loss'])

plt.title('Loss with SeNet Module in different positions')
plt.ylabel('Loss')
plt.xlabel('Attention Module Positions')
plt.plot(train_loss_history)
plt.plot(val_loss_history)
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.show()
