import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ------------------- CONFIGURATION -------------------
train_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\train"
val_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\val"
image_size = (128, 128)
batch_size = 32
phase1_epochs = 10   # Frozen base
phase2_epochs = 5    # Fine-tune last 20 layers

# ------------------- DATA AUGMENTATION -------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
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

# ------------------- BUILD MODEL (MobileNetV2) -------------------
num_classes = len(train_generator.class_indices)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Phase 1: freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------- CALLBACKS -------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
checkpoint_phase1 = ModelCheckpoint(
    filepath=r"E:\test_data_2.0\best_model_phase1_skeleton.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ------------------- PHASE 1 TRAINING -------------------
print("\n🚀 Phase 1: Training with frozen base...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=phase1_epochs,
    callbacks=[early_stop, lr_reducer, checkpoint_phase1]
)

# ------------------- PHASE 2: FINE-TUNING -------------------
print("\n📥 Loading best Phase 1 model for fine-tuning...")
model = load_model(r"E:\test_data_2.0\best_model_phase1_skeleton.h5")

# Unfreeze last 20 layers
base_model = model.layers[0]
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_phase2 = ModelCheckpoint(
    filepath=r"E:\test_data_2.0\best_model_finetune_skeleton.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\n🔧 Phase 2: Fine-tuning last 20 layers...")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=phase2_epochs,
    callbacks=[early_stop, lr_reducer, checkpoint_phase2]
)

# ------------------- SAVE FINAL MODEL -------------------
final_model_path = r"E:\test_data_2.0\sign2text_mobilenetv2_skeleton_model.h5"
model.save(final_model_path)
print(f"\n✅ Skeleton-trained model saved at: {final_model_path}")
