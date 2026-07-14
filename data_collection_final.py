import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------- CONFIGURATION -------------------
train_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\train"
val_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\val"
image_size = (64, 64)  # You can increase to 128 for better accuracy
batch_size = 32
epochs = 20

# ------------------- DATA GENERATORS -------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ------------------- BUILD CNN MODEL -------------------
num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------- TRAIN THE MODEL -------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ------------------- SAVE THE MODEL -------------------
model_save_path = r"E:\test_data_2.0\sign2text_cnn_model.h5"
model.save(model_save_path)
print(f"\n✅ Model saved at: {model_save_path}")
