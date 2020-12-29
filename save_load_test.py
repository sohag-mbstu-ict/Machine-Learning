from tensorflow.keras.models import load_model
model = load_model("bank_loan.h5")
model.summary()