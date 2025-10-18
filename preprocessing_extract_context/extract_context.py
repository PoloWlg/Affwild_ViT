import cv2
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import time
import os 

# --- 1. Load the LLaVA-Next model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16)
model.to(device)

# --- 2. Function to process a frame and generate description ---
def describe_frame(frame):
    # Convert frame (OpenCV is BGR) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Describe briefly the background"},
          {"type": "image"},
        ],
    },
]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=frame_rgb, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=500)

    decoded = processor.decode(output[0], skip_special_tokens=True)
    decoded = decoded.split("ASSISTANT:")[1].strip()
    
    return decoded

# --- 3. Load video and extract frames ---
def process_video(video_path, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    contexts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx == 0:
          tic = time.time()
        if frame_idx % frame_skip == 0:
            print(f"Processing frame {frame_idx}...")
            context = describe_frame(frame)
            contexts.append((frame_idx, context))
        
        if frame_idx == 10:
          toc = time.time()
          print(f"Frame {frame_idx} processed in {toc - tic:.2f} seconds.")
        frame_idx += 1
      
        
    cap.release()
    return contexts

# --- 4. Example usage ---


video_file = "/projets/AS84330/Datasets/Abaw6_EXPR/raw_videos/462.avi"
# frame_contexts = process_video(video_file, frame_skip=1)  # e.g., one frame every 30 frames



# Print descriptions
# for idx, context in frame_contexts:
#     print(f"\nFrame {idx} context:\n{context}")
    

