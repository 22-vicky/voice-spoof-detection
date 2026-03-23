import os
import csv

protocol_file = r"LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
audio_dir = r"LA\ASVspoof2019_LA_train\flac"
output_csv = "train_list.csv"

rows = []
missing = 0

with open(protocol_file, "r") as f:
    for line in f:
        parts = line.strip().split()

        file_id = parts[1]          # LA_T_xxxxxx
        label = parts[4]            # bonafide / spoof

        audio_path = os.path.join(audio_dir, file_id + ".flac")

        if not os.path.exists(audio_path):
            missing += 1
            continue

        y = 0 if label == "bonafide" else 1  # 0=real, 1=spoof
        rows.append([audio_path, y])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    writer.writerows(rows)

print("✅ Dataset index created!")
print(f"Total samples: {len(rows)}")
print(f"Missing files: {missing}")
