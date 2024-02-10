### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras import Model, layers, Input

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
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False),
        Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False),
        Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False),
        Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False),
        Conv2D(127, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False),
        Conv2D(127, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        # Adjusted dense layer size to increase parameters
        Dense(264, activation='relu'),  # Adjusting the number of units here
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))
    
    residual = layers.Conv2D(32, (3, 3), strides=(2, 2), name='C1', activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(residual)
    c1 = layers.Dropout(0.5)(c1)

    c2 = layers.Conv2D(64, (3, 3), strides=(2, 2), name='C2', activation='relu', padding='same')(c1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.5)(c2)
    
    c3 = layers.Conv2D(128, (3, 3), strides=(2, 2), name='C3', activation='relu', padding='same')(c2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.5)(c3)

    s1 = layers.Conv2D(128, (1, 1), strides=(4, 4), name="S1")(residual)
    s1 = layers.Add()([s1, c3])

    c4 = layers.Conv2D(128, (3, 3), name='C4', activation='relu', padding='same')(s1)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.5)(c4)

    c5 = layers.Conv2D(128, (3, 3), name='C5', activation='relu', padding='same')(c4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.5)(c5)

    s2 = layers.Add()([s1, c5])

    c6 = layers.Conv2D(128, (3, 3), name='C6', activation='relu', padding='same')(s2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.5)(c6)

    c7 = layers.Conv2D(128, (3, 3), name='C7', activation='relu', padding='same')(c6)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.5)(c7)

    s3 = layers.Add()([s2, c7])
    
    pool = layers.MaxPooling2D((4, 4), strides=(4, 4))(s3)
    flatten = layers.Flatten()(pool)
    
    dense = layers.Dense(128, activation='relu')(flatten)
    dense = layers.BatchNormalization()(dense)

    output = layers.Dense(10, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=output)
    return model




def build_model50k():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(12, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':
    ########################################
    ## Add code here to Load the CIFAR-10 data set
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Split the training set into a smaller training set and a validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    ########################################
    ## Build and train model 1
    model1 = build_model1()
    model1.summary()

    # Train the model
    history = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc}')

    # Optional: Plot training and validation accuracy over epochs to check for overfitting
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
  
if __name__ == '__main__':
    ########################################
    ## Load the CIFAR-10 data set
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Split the training set into a smaller training set and a validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    ########################################
    ## Build and train model 2
    model2 = build_model2()
    model2.summary()

    # Train the model
    history_model2 = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
    test_loss_model2, test_acc_model2 = model2.evaluate(test_images, test_labels, verbose=2)
    print(f'Model 2 test accuracy: {test_acc_model2}')

    # Optional: Plot training and validation accuracy over epochs to check for overfitting
    plt.plot(history_model2.history['accuracy'], label='accuracy')
    plt.plot(history_model2.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    ########################################
    ## Load the CIFAR-10 data set
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Split the training set into a smaller training set and a validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    ########################################
    ## Build and train model 3
    model3 = build_model3()
    model3.summary()

    # Train the model
    history_model3 = model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

    # Evaluate the model on the test set
    test_loss_model3, test_acc_model3 = model3.evaluate(test_images, test_labels, verbose=2)
    print(f'Model 3 test accuracy: {test_acc_model3}')

    # Plot training and validation accuracy over epochs to check for overfitting
    plt.plot(history_model3.history['accuracy'], label='Training Accuracy')
    plt.plot(history_model3.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    # Plot training and validation loss as well
    plt.plot(history_model3.history['loss'], label='Training Loss')
    plt.plot(history_model3.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
  
if __name__ == '__main__':
    model = build_model50k()
    model.summary()

    # Check the number of parameters
    if model.count_params() > 50000:
        print("Model exceeds 50,000 parameters.")
    else:
        # Load and preprocess the CIFAR-10 data set
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

        # Compute quantities required for featurewise normalization
        datagen.fit(train_images)

        # Train the model
        history = model.fit(
            datagen.flow(train_images, train_labels, batch_size=32),
            steps_per_epoch=len(train_images) // 32,  # number of batches to yield from the generator per epoch
            epochs=50, 
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=2
        )

        # Load the best model saved by ModelCheckpoint
        best_model = tf.keras.models.load_model('best_model.h5')

        # Evaluate the best model
        test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
        print(f"Best model test accuracy: {test_acc}")

        # Plot the training and validation accuracy
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot the training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()