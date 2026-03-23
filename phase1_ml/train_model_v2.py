import librosa
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV_PATH = "train_list.csv"
AUDIO_DIR = "LA/ASVspoof2019_LA_train/flac"

df = pd.read_csv(CSV_PATH)

# 🔍 Detect correct column name automatically
if "audio_path" in df.columns:
    audio_col = "audio_path"
elif "filename" in df.columns:
    audio_col = "filename"
elif "file" in df.columns:
    audio_col = "file"
elif "path" in df.columns:
    audio_col = "path"
else:
    raise Exception(f"❌ No audio column found. Columns are: {df.columns}")

label_col = "label"

X, y = [], []

print("🔄 Extracting MFCC features...")

for i, row in df.iterrows():
    filename = os.path.basename(str(row[audio_col]))
    label = int(row[label_col])

    audio_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(audio_path):
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        X.append(mfcc_mean)
        y.append(label)

    except Exception:
        continue

    if i % 2000 == 0 and i > 0:
        print(f"Processed {i} samples...")

X = np.array(X)
y = np.array(y)

print("✅ Feature extraction completed")
print("Total usable samples:", len(X))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🧠 Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Spoof"]))

# Save model
joblib.dump(model, "voice_spoof_model_v2.pkl")
print("💾 Model saved as voice_spoof_model_v2.pkl")
