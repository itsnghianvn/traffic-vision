import cv2
import time 

video = cv2.VideoCapture("data/videos/traffic.mp4")

# Validate video
if not video.isOpened():
    raise RuntimeError("Error opening video stream or file")

# Read video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    raise ValueError("FPS must be greater than 0")

duration = frame_count / fps

# Set output video properties
output_width = width // 2
output_height = height // 2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "data/output/output.mp4", 
    fourcc, 
    fps, 
    (output_width, output_height)
)

print("Starting video processing...")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Frame count: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# Process video frames
frame_index = 0
start_time = time.time()
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_index += 1
    # Resize frame
    frame = cv2.resize(frame, (output_width, output_height))
    # Draw rectangle
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 3)
    # Draw frame number
    cv2.putText(frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Process frame
    out.write(frame)

    if frame_index % 30 == 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        processing_fps = frame_index / elapsed_time
        print(f"Processed {frame_index}/{frame_count} frames. Current processing FPS: {processing_fps:.2f}")
        
video.release()
out.release()
end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds.")
