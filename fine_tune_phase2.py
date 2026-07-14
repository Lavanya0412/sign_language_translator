import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------- CONFIGURATION -------------------
train_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\train"
val_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split\val"
image_size = (128, 128)
batch_size = 32
fine_tune_epochs = 5  # Adjust if needed

# ------------------- DATA GENERATORS -------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
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

# ------------------- LOAD PHASE 1 MODEL -------------------
print("\n📥 Loading best Phase 1 model for fine-tuning...")
model = load_model(r"E:\test_data_2.0\best_model_phase1_skeleton.h5")

# ------------------- PHASE 2: FINE-TUNING -------------------
print("\n🔧 Fine-tuning last 20 layers of the model...")

# Freeze all layers except the last 20
for layer in model.layers[:-20]:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

# Recompile with a very low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------- CALLBACKS -------------------
early_stop = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1
)

checkpoint_phase2 = ModelCheckpoint(
    filepath=r"E:\test_data_2.0\best_model_finetune_skeleton.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ------------------- FINE-TUNING -------------------
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=fine_tune_epochs,
    callbacks=[early_stop, lr_reducer, checkpoint_phase2]
)

# ------------------- SAVE FINAL MODEL -------------------
final_model_path = r"E:\test_data_2.0\sign2text_mobilenetv2_skeleton_model.h5"
model.save(final_model_path)
print(f"\n✅ Fine-tuned skeleton model saved at: {final_model_path}")
