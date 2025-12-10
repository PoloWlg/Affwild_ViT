import cv2
import numpy as np

# cap = cv2.VideoCapture("/projets/AS84330/Datasets2/Abaw6_EXPR_contextual/raw_videos/131-30-1920x1080.mp4")
cap = cv2.VideoCapture("/home/ens/AS84330/Stimuli/Affwild/Stimuli_official/Affwild_ViT/My_video_test.mp4")

# Get frame width, height, FPS
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

def find_intersection(horizontal_lines, vertical_lines):
    intersections = []

    # Example: after detecting lines
    # horizontal_lines.append((y1, x1_temp, x2_temp))
    # vertical_lines.append((x1, y1_temp, y2_temp))

    for h in horizontal_lines:
        y = h[0]
        x_h_start = h[1]
        x_h_end = h[2]
        
        for v in vertical_lines:
            x = v[0]
            y_v_start = v[1]
            y_v_end = v[2]
            
            # Check if intersection is within both line segments
            if x_h_start <= x <= x_h_end and y_v_start <= y <= y_v_end:
                intersections.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # red dot at intersection
    return intersections

i = 0
while True:
    print(f"Processing frame {i}")
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Horizontal: y difference small
            if abs(y2 - y1) < 1 and abs(x2 - x1) > 100:
                x1_temp = x1 - 10000
                x2_temp = x2 + 10000
                cv2.line(frame, (x1_temp, y1), (x2_temp, y2), (0, 255, 0), 2)  # green
                horizontal_lines.append((y1, x1_temp, x2_temp))
            
            # Vertical: x difference small
            elif abs(x2 - x1) < 1 and abs(y2 - y1) > 100:
                y1_temp = y1 - 10000
                y2_temp = y2 + 10000
                cv2.line(frame, (x1, y1_temp), (x2, y2_temp), (255, 0, 0), 2)  # blue
                vertical_lines.append((x1, y1_temp, y2_temp))
    
        find_intersection(horizontal_lines, vertical_lines)  # Example call; adapt as needed
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()