import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ===============================
# PATHS (DO NOT CHANGE)
# ===============================
DATASET_DIR = r"LA\ASVspoof2019_LA_train\flac"
PROTOCOL_FILE = r"LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"


print("Current directory:", os.getcwd())
print("Dataset exists:", os.path.exists(DATASET_DIR))
print("Protocol exists:", os.path.exists(PROTOCOL_FILE))

if not os.path.exists(DATASET_DIR) or not os.path.exists(PROTOCOL_FILE):
    raise FileNotFoundError("Dataset or protocol file not found. Check folder structure.")


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


# ===============================
# READ PROTOCOL FILE
# ===============================
print("Reading protocol file...")

with open(PROTOCOL_FILE, "r") as f:
    lines = f.readlines()

print("Total protocol entries:", len(lines))


# ===============================
# LOAD DATA (LIMITED)
# ===============================
X = []
y = []

MAX_SAMPLES = 3000   # ✅ IMPORTANT: keeps training fast
usable = 0

for line in lines:
    if usable >= MAX_SAMPLES:
        break

    parts = line.strip().split()
    file_id = parts[1]           # LA_T_******
    label = parts[-1]            # bonafide / spoof

    audio_path = os.path.join(DATASET_DIR, file_id + ".flac")

    if not os.path.exists(audio_path):
        continue

    try:
        features = extract_features(audio_path)
        X.append(features)
        y.append(1 if label == "bonafide" else 0)
        usable += 1

        if usable % 500 == 0:
            print(f"Processed {usable} samples...")

    except Exception as e:
        continue


print("Usable samples:", usable)


# ===============================
# TRAIN TEST SPLIT
# ===============================
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# TRAIN MODEL
# ===============================
print("Training model...")

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)


# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)


# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "voice_spoof_model.pkl")
print("Model saved as voice_spoof_model.pkl")
