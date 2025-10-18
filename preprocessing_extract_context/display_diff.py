from utils import get_all_videos_and_annotations_path, count_frame_video
import cv2


data = {
    "video_name": [],
    "num_frames_video": [],
    "num_frames_annotation": [],
    "difference": [],
}

annotation_video_path = get_all_videos_and_annotations_path()

for annotation_path, video_path in annotation_video_path:
    video_name = video_path.split("/")[-1].split(".")[0]
    
    print(f"Video: {video_path}")
    print(f"Annotation: {annotation_path}")
    
    # Process video
    num_frames_video = count_frame_video(video_path)
    
    # Open annotation file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
        annotations_removed = [annotations for x in annotations if x != '-1\n']
        annotations_removed_len = len(annotations_removed) - 1

    difference = num_frames_video - annotations_removed_len
    
    data["video_name"].append(video_name)
    data["num_frames_video"].append(num_frames_video)
    data["num_frames_annotation"].append(annotations_removed_len)
    data["difference"].append(difference)
    
    
# Save data to a csv file
import pandas as pd
df = pd.DataFrame(data)
df.to_csv("video_annotation_difference.csv", index=False)