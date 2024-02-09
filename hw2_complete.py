### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plte

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, MaxPooling2D, Flatten, Dens

## 

### build model
def build_model1():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model2():
    model = Sequential([
        # First Conv2D layer with 32 filters remains the same as in build_model1
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        
        # Replace the next Conv2D layer with 64 filters with a SeparableConv2D and 1x1 Conv2D
        SeparableConv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(64, kernel_size=(1, 1), activation='relu'),

        # Replace the next Conv2D layer with 128 filters with a SeparableConv2D and 1x1 Conv2D
        SeparableConv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(128, kernel_size=(1, 1), activation='relu'),

        # For the remaining Conv2D layers with 128 filters, replace with SeparableConv2D layers
        # Since there is no striding in these layers, we do not apply stride in the SeparableConv2D
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(128, kernel_size=(1, 1), activation='relu'),

        # Repeat the pattern for each additional pair of Conv2D+BatchNorm layers
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(128, kernel_size=(1, 1), activation='relu'),

        SeparableConv2D(128, kernel_size=(3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(128, kernel_size=(1, 1), activation='relu'),

        SeparableConv2D(128, kernel_size=(3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Conv2D(128, kernel_size=(1, 1), activation='relu'),

        # MaxPooling, Flatten, and Dense layers remain the same
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')  # Assuming 10 classes in CIFAR-10
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))

    # Initial Convolution block
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # Stage 1: Convolutional block with residual connection
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # Preparing for next residual connection
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)

    # Stage 2: Another block, increasing filters
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # Final layers after all blocks
    x = MaxPooling2D((4, 4))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Increased dropout before the final layer
    outputs = Dense(10, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name="model3")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



def build_model50k():
    inputs = Input(shape=(32, 32, 3))

    # Convolutional block 1
    x = SeparableConv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)

    # Convolutional block 2
    x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(23.9, activation='relu')(x)  # Reduce the number of units here
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()

  


  
