import cv2
import numpy as np
from scipy.special import softmax
from moviepy.editor import VideoFileClip, AudioFileClip



def video_visualization(video_path, annotation_path, prediction_path, video_name):
    EMOTION_CAT = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"]

    # Load annotations
    with open(annotation_path, "r") as f:
        annotations = [line.strip() for line in f.readlines()]

    with open(prediction_path, "r") as f:
        predictions = [line.strip() for line in f.readlines()]
    # Open video
    
    cap = cv2.VideoCapture(video_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {num_frames} frames at {fps} fps")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'/home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/video_visualisation/{video_name}.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps,
                        (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= (num_frames - 100):
            break
        

        groud_truth = annotations[frame_idx]
        groud_truth = EMOTION_CAT[int(groud_truth)]
        
        prediction = predictions[frame_idx] 
        prediction = np.fromstring(predictions[frame_idx] , sep=',')
        if prediction.sum() == -8:
            groud_truth = -1
            prediction_cat = "No Face Detected"
            prediction = -1
            prediction_top4 = -1
            prediction_proposed = -1
            prediction_cat_proposed = -1
            prediction_top4_proposed = -1
            
        else:
            prediction_soft = softmax(prediction)
            top4_idx = np.argsort(prediction_soft)[-4:][::-1]
            
            prediction_top4 = prediction[top4_idx]
            prediction_cat = [EMOTION_CAT[i] for i in top4_idx]

            weights = [0.39, 0.55, 0.55, 0.62, 0.49, 0.58, 0.60, 0.48]
            temperature = 0.1
            
            prediction_proposed = prediction * weights
            prediction_proposed = softmax(prediction_proposed/temperature)
            
            top4_idx_proposed = np.argsort(prediction_proposed)[-4:][::-1]
            prediction_top4_proposed = prediction_proposed[top4_idx_proposed]
            prediction_cat_proposed = [EMOTION_CAT[i] for i in top4_idx_proposed]
            
        if groud_truth == prediction_cat[0]:
            color_prediction = (0, 255, 0)  # Green for correct
        else:
            color_prediction = (0, 0, 255)  # Red for incorrect
        # Draw annotation text
        cv2.putText(frame, f'GROUND TRUTH: {groud_truth}', (5, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f'FACE PREDICTION: {prediction_cat}', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_prediction, 1, cv2.LINE_AA)
        
        cv2.putText(frame, f'CONFIDENCE: {prediction_top4}', (5, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_prediction, 1, cv2.LINE_AA)
        
        cv2.putText(frame, f'PROPOSED CAT : {prediction_cat_proposed}', (5, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_prediction, 1, cv2.LINE_AA)
        
        cv2.putText(frame, f'PROPOSED CONFIDENCE : {prediction_top4_proposed}', (5, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_prediction, 1, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    # clip = VideoFileClip(video_path)
    # clip.audio.write_audiofile("audio_tmp.wav")
    # video = VideoFileClip(output_path)
    # audio = AudioFileClip("audio_tmp.wav")

    # final_duration = min(video.duration, audio.duration)

    # video = video.subclip(0, final_duration)
    # audio = audio.subclip(0, final_duration)
    
    # final = video.set_audio(audio)
    # final.write_videofile(output_path, codec="mpeg4", audio_codec="aac")
    print("Saved video_with_annotations.mp4 ✅")
    
    
if __name__ == "__main__":
    
    
    
    import os 
    import pickle
    with open('/home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/weights_saved/test_Proposed__fold0_valence_seed4/validate_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    videos = {}
    for video in loaded_dict:
        videos[video] = loaded_dict[video]['f1_score']['surprise']
    sorted_videos = sorted(videos.items(), key=lambda x: x[1])
    print("Videos sorted by F1 score:")
    for video, score in sorted_videos:
        print(f"{video}: {score}")
    
    for video_name in loaded_dict:
        video_name = 'video73'
        print(video_name)
        video_path = f'/projets/AS84330/Datasets/Abaw6/raw_videos/{video_name}.mp4'
        if not os.path.exists(video_path):
            video_path = f'/projets/AS84330/Datasets/Abaw6/raw_videos/{video_name}.avi'
        
        annotation_path = f'/projets/AS84330/Datasets/Abaw6/6th_ABAW_Annotations/EXPR_Recognition_Challenge/Validation_Set/{video_name}.txt'
        prediction_path = f'/home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/weights_saved/test_Proposed__fold0_valence_seed4/outputs_predictions/Validation_Set/{video_name}_output.txt'
        video_visualization(video_path, annotation_path, prediction_path, video_name)
        break