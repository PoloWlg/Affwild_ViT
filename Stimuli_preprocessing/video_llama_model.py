import torch
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)



def video_llama(video_path, instruction, fps, max_frames):
    conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
            {"type": "text", "text":instruction},
        ]
    },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=4096)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

import os 
from tqdm import tqdm
def process_video(instruction, fps, max_frames):
    save_root_path = '/home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/Stimuli_preprocessing/video_stimuli/video_based'
    videos_path = '/projets/AS84330/Datasets/Abaw6/raw_videos'
    videos_list = os.listdir(videos_path)
    for video_name in tqdm(videos_list, desc="Processing videos", total=len(videos_list)):
        video_path = os.path.join(videos_path, video_name)
        video_name = video_name.split('.')[0]
        result = video_llama(video_path, instruction, fps=fps, max_frames=max_frames)
        # Save in a text file
        save_path = os.path.join(save_root_path, f"{video_name}_description.txt")
        with open(save_path, "w") as f:
            f.write(result)

if __name__ == "__main__":
    video_path = '/home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/Stimuli_preprocessing/video_visualisation/282.mp4'
    instruction = "Describe in details the movie this person is watching at. \
    What type of movie is it. \
    Describe the mood of the movie. \
    Describe what happen in the movie \
\
"
    fps = 1
    max_frames = 16
    process_video(instruction, fps, max_frames)