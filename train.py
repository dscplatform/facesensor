from data import build_dataset
from model import build_model
from keras.callbacks import EarlyStopping
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters
version = 1
batch_size = 10
epochs = 16

# Build dataset and model
X, y = build_dataset()
model = build_model()

# Early stop
early_stopping_monitor = EarlyStopping(patience=3)

# Fit the model
history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping_monitor])

# Export the model
print("Training complete, saving model")
model.save(("export/mdl_v%d.h5")%(version))
