### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

## 

def build_model1():
    model = Sequential([
        # Convolutional layer block 1
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        
        # Convolutional layer block 2
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        
        # Convolutional layer block 3
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        
        # Four more blocks of Conv2D+BatchNorm without striding
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        
        # MaxPooling
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        
        # Flattening the output of the conv layers to feed into the dense layer
        Flatten(),
        
        # Dense layer block
        Dense(128, activation='relu'),
        BatchNormalization(),
        
        # Output layer
        Dense(10, activation='softmax')  # Assuming 10 classes in CIFAR-10
    ])
    
    return model

def build_model2():
  model = None # Add code to define model 1.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
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
  
  
