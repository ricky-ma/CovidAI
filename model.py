from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator


# Building the model
def build_model(x_train):
    inputs = Input(shape=x_train.shape[1:])

    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(inputs)
    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


# Compile the model
def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# Augment image data
def augment_data():
    datagen = ImageDataGenerator(
      rotation_range=10,
      zoom_range=0.1,
      width_shift_range=0.1,
      height_shift_range=0.1
    )
    return datagen


# Fit model
def fit(model, x_train, y_train, datagen):
    epochs = 3
    batch_size = 32

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size)
    return history
