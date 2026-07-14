from tensorflow.keras.models import load_model

model_path = "E:/test_data_2.0/sign2text_mobilenetv2_skeleton_model.h5"
model = load_model(model_path)
print("Model loaded successfully!")
