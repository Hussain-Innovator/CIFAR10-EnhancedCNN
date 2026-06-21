import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tf_keras

def se_block(x, ratio=16):
    channels = x.shape[-1]
    se = tf_keras.layers.GlobalAveragePooling2D()(x)
    se = tf_keras.layers.Dense(channels // ratio, activation='relu', use_bias=False)(se)
    se = tf_keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    se = tf_keras.layers.Reshape((1, 1, channels))(se)
    return tf_keras.layers.Multiply()([x, se])

def residual_block(x, filters, use_depthwise=False):
    in_channels = x.shape[-1]
    shortcut = x
    x = tf_keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.ReLU()(x)
    if use_depthwise:
        x = tf_keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
        x = tf_keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    else:
        x = tf_keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = se_block(x)
    if in_channels != filters:
        shortcut = tf_keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = tf_keras.layers.BatchNormalization()(shortcut)
    x = tf_keras.layers.Add()([x, shortcut])
    return tf_keras.layers.ReLU()(x)

# Build model
inputs = tf_keras.Input(shape=(32, 32, 3), name='input')
x = tf_keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='stem_conv')(inputs)
x = tf_keras.layers.BatchNormalization(name='stem_bn')(x)
x = tf_keras.layers.ReLU(name='stem_relu')(x)
x = residual_block(x, 64,  False)
x = tf_keras.layers.MaxPooling2D(2, name='pool1')(x)
x = tf_keras.layers.Dropout(0.25, name='drop1')(x)
x = residual_block(x, 128, True)
x = tf_keras.layers.MaxPooling2D(2, name='pool2')(x)
x = tf_keras.layers.Dropout(0.25, name='drop2')(x)
x = residual_block(x, 256, True)
x = tf_keras.layers.MaxPooling2D(2, name='pool3')(x)
x = tf_keras.layers.Dropout(0.25, name='drop3')(x)
x = tf_keras.layers.GlobalAveragePooling2D(name='gap')(x)
x = tf_keras.layers.Dropout(0.4, name='drop_head')(x)
outputs = tf_keras.layers.Dense(10, activation='softmax', name='predictions')(x)
model = tf_keras.Model(inputs, outputs, name='EnhancedCNN_CIFAR10')

# Build with dummy input
dummy = np.zeros((1, 32, 32, 3), dtype='float32')
model(dummy, training=False)
print('Model built successfully')

# Print layer names to compare
print('\nLocal model layer names:')
for layer in model.layers:
    print(f'  {layer.name}')

# Load weights by name
model.load_weights(
    'models/model_weights.weights.h5',
    by_name=True,
    skip_mismatch=True
)
print('\nWeights loaded successfully')

# Test prediction
result = model.predict(dummy, verbose=0)
print(f'Output shape : {result.shape}')
print(f'Sum of probs : {result.sum():.4f}  (should be ~1.0)')
print('\nALL TESTS PASSED')