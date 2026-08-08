import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

# ==========================================
# 1. 3D CNN Model Architecture Definition
#    (Must match the training architecture)
# ==========================================
class VideoClassifier3D(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoClassifier3D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Expects: (Batch, Time, Channel, Height, Width) -> (B, T, C, H, W)
        # Permute to: (Batch, Channel, Time, Height, Width) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==========================================
# 2. Preprocessing & Uniform Frame Sampling
# ==========================================
def preprocess_video(video_path, T=16):
    """
    Loads a video, samples T frames uniformly, and applies validation-style scaling.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Error: Could not read any frames from the video at '{video_path}'.")
        
    # Apply identical uniform frame-sampling logic
    if total_frames >= T:
        indices = np.linspace(0, total_frames - 1, T, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        padding = [total_frames - 1] * (T - total_frames)
        indices = np.concatenate([indices, padding])
        
    sampled_frames = [frames[i] for i in indices]
    
    # Process sampled frames using standard validation transformations (Center Crop)
    transformed_frames = []
    for f in sampled_frames:
        img = Image.fromarray(f)
        img = TF.resize(img, (128, 128))
        img = TF.center_crop(img, (112, 112))
        
        # Convert to tensor and apply ImageNet normalization
        tensor_img = TF.to_tensor(img)
        tensor_img = TF.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformed_frames.append(tensor_img)
        
    # Stack list of tensors into a clip tensor with shape (1, T, C, H, W)
    clip_tensor = torch.stack(transformed_frames).unsqueeze(0)
    return clip_tensor


# ==========================================
# 3. Model Loading & Inference Execution
# ==========================================
def run_single_inference(video_path, model_path="D:/Cellula/Week 5/commit/best_video_classifier.pth", T=16):
    """
    Runs model inference on a single video file.
    """
    # Detect processing hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model checkpoint exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in the current directory.")
        return
        
    # Initialize and load model
    model = VideoClassifier3D(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess target video feed
    try:
        print(f"Reading and preprocessing video: '{video_path}'...")
        input_tensor = preprocess_video(video_path, T=T).to(device)
    except Exception as e:
        print(e)
        return
        
    # Predict action sequence
    print("Running model inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply Softmax to get raw probability scores
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class_idx = torch.max(probabilities, dim=0)
        
    # Mapping output prediction back to text class labels
    classes_map = {0: "Normal Shopping Behavior (Non-Shoplifting)", 1: "Shoplifting/Theft Act"}
    prediction_label = classes_map[predicted_class_idx.item()]
    confidence_percentage = confidence.item() * 100
    
    # Print clean formatted results
    print("\n==========================================")
    print("           SINGLE FEED INFERENCE RESULTS   ")
    print("==========================================")
    print(f"Target Video Path : {video_path}")
    print(f"Prediction Output : {prediction_label}")
    print(f"Confidence Level  : {confidence_percentage:.2f}%")
    print(f"Raw Probabilities :")
    print(f"  - Normal Behavior : {probabilities[0].item():.4f}")
    print(f"  - Theft Behavior  : {probabilities[1].item():.4f}")
    print("==========================================\n")


# ==========================================
# Script Execution Target
# ==========================================
if __name__ == "__main__":
    # Path to the specific video you want to test
    target_video = r"D:\Cellula\Week 5\Shop DataSet\shoplifting\shoplifting-70.mp4"
    
    # Execute inference
    run_single_inference(
        video_path=target_video, 
        model_path="D:/Cellula/Week 5/commit/best_video_classifier.pth", 
        T=16
    )