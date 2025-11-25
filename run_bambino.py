# Import Bambino
from bambino import Bambino

# Startup Framework w/ Record Images, Form Ground Truths and Desired Model
record_image_path = ["data/forms/im1.jpg"]
ground_truth_csv = "data/ground_truths/ground_truth_img4.csv"
model_path = "models/Oct19LV_500E_64B_W-AUG/detect/train2/weights/best.pt"

# Overlap Percent Threshold
overlap_threshold = 15

# Run Bambino
bambi = Bambino(record_images=record_image_path, ground_truth_csv=ground_truth_csv, model_path=model_path)
bambi.run(overlap_threshold) # Returns out.html