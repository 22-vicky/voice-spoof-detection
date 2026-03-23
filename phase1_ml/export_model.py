import joblib
import pickle

# Load model using joblib (because it was saved with joblib)
model = joblib.load("voice_spoof_model.pkl")

# Save again using pickle (Django-friendly)
with open("clean_voice_spoof_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Clean model saved as clean_voice_spoof_model.pkl")
