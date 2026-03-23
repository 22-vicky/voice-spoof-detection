# import os
# import numpy as np
# import librosa
# import joblib   # ✅ IMPORTANT

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# # =====================================================
# # LOAD MODEL (JOBLIB — FIXES \x08 ERROR)
# # =====================================================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "voice_spoof_model_v2.pkl")

# print("🔍 Loading model from:", MODEL_PATH)

# model = None
# try:
#     model = joblib.load(MODEL_PATH)
#     print("✅ Model loaded successfully with joblib")
# except Exception as e:
#     print("❌ Model load failed:", e)
#     model = None


# # =====================================================
# # FEATURE EXTRACTION — MUST MATCH TRAINING
# # MFCC(20) → mean → shape (20,)
# # =====================================================

# def extract_features(audio_path):
#     try:
#         y, sr = librosa.load(audio_path, sr=16000)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
#         return np.mean(mfcc, axis=1)
#     except Exception as e:
#         print("❌ Feature extraction error:", e)
#         return None


# # =====================================================
# # PREDICT API
# # POST /predict/
# # =====================================================

# @csrf_exempt
# def predict(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "POST request required"}, status=400)

#     if model is None:
#         return JsonResponse({"error": "Model not loaded"}, status=500)

#     if "audio" not in request.FILES:
#         return JsonResponse({"error": "No audio file provided"}, status=400)

#     audio_file = request.FILES["audio"]

#     temp_dir = os.path.join(BASE_DIR, "temp")
#     os.makedirs(temp_dir, exist_ok=True)
#     temp_path = os.path.join(temp_dir, audio_file.name)

#     try:
#         with open(temp_path, "wb+") as f:
#             for chunk in audio_file.chunks():
#                 f.write(chunk)

#         features = extract_features(temp_path)
#         if features is None:
#             return JsonResponse({"error": "Feature extraction failed"}, status=500)

#         features = features.reshape(1, -1)

#         prediction = model.predict(features)[0]
#         confidence = model.predict_proba(features).max()

#         label = "Spoof" if prediction == 1 else "Genuine"

#         return JsonResponse({
#             "prediction": label,
#             "confidence": round(float(confidence), 4)
#         })

#     except Exception as e:
#         print("❌ Prediction error:", e)
#         return JsonResponse({"error": "Prediction failed"}, status=500)

#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)








import os
import uuid
import numpy as np
import librosa
import joblib

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render


# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_spoof_model_v2.pkl")

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None


# =====================================================
# FEATURE EXTRACTION
# Supports .wav and .flac
# =====================================================

def extract_features(audio_path):
    try:
        # Resample everything to 16kHz (same as training)
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract 20 MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Take mean across time axis → shape (20,)
        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean

    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None


# =====================================================
# PREDICT API
# POST /predict/
# =====================================================

@csrf_exempt
def predict(request):

    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    if model is None:
        return JsonResponse({"error": "Model not loaded"}, status=500)

    if "audio" not in request.FILES:
        return JsonResponse({"error": "No audio file provided"}, status=400)

    audio_file = request.FILES["audio"]

    # Create temp directory
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Keep original extension (.wav or .flac)
    original_ext = os.path.splitext(audio_file.name)[1].lower()
    filename = str(uuid.uuid4()) + original_ext
    temp_path = os.path.join(temp_dir, filename)

    try:
        # Save uploaded file
        with open(temp_path, "wb+") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Extract features
        features = extract_features(temp_path)

        if features is None:
            return JsonResponse(
                {"error": "Feature extraction failed. Please upload .wav or .flac file."},
                status=500
            )

        # Reshape for model
        features = features.reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        label = "Spoof" if prediction == 1 else "Genuine"

        return JsonResponse({
            "prediction": label,
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return JsonResponse({"error": "Prediction failed"}, status=500)

    finally:
        # Clean temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# LANDING PAGE
# =====================================================

def landing(request):
    return render(request, "landing.html")


# =====================================================
# REAL-TIME PAGE
# =====================================================

def realtime(request):
    return render(request, "realtime.html")


# =====================================================
# ANALYSIS PAGE
# =====================================================

def analysis_page(request):
    return render(request, "index.html")
