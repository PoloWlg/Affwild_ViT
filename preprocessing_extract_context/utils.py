import os 
import cv2




RAW_VIDEOS_PATH = '/projets/AS84330/Datasets/Abaw6_EXPR/raw_videos'

def get_videos_and_annotations_path(path):
    __list_annotations = []
    for annotation in os.listdir(path):
        annotation_path = os.path.join(path, annotation)
        
        if not (os.path.exists(annotation_path)):
            raise('annotation: ', annotation_path,  'does not exist')
        
        video_name = annotation.split('.')[0]
        video_name = video_name.split('_left')[0]
        video_name = video_name.split('_right')[0]
        
        video_path = os.path.join(RAW_VIDEOS_PATH, video_name + '.mp4')
        
        if os.path.exists(video_path):
            __list_annotations.append((annotation_path, video_path))
            continue
        else:
            video_path = os.path.join(RAW_VIDEOS_PATH, video_name + '.avi')
            if os.path.exists(video_path):
                __list_annotations.append((annotation_path, video_path))
                continue
            else:
                raise ('video: ', video_name,  'does not exist')
            
    return __list_annotations

def get_all_videos_and_annotations_path():
    """
    Get video and annotation path for affwild2 dataset 
    """
    
    annotations_path = '/projets/AS84330/Datasets/Abaw6_EXPR/6th_ABAW_Annotations/EXPR'
    
    
    train_path = os.path.join(annotations_path, 'train')
    validation_path = os.path.join(annotations_path, 'validate')
    train_annotations = get_videos_and_annotations_path(train_path)
    val_annotations = get_videos_and_annotations_path(validation_path)
    
    train_val_annotations = train_annotations + val_annotations
            
    return train_val_annotations


def count_frame_video(video_path):
    """Count the number of frames in a video file."""
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
      
        
    cap.release()
    return frame_idx

if __name__ == "__main__":
    list_annotations = get_all_videos_and_annotations_path()
    print(list_annotations)