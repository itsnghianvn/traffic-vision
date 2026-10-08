from ultralytics import YOLO
import cv2
import math

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("data/videos/traffic.mp4")

if not video.isOpened():
    raise ValueError("Could not open video file.")

track_history = {}

# ROI
roi_x1 = 100
roi_y1 = 250
roi_x2 = 1800
roi_y2 = 1800

# Movement threshold in pixels
MOVEMENT_THRESHOLD = 5

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]
    tracking_frame = result.plot()

    roi_count = 0
    stopped_count = 0

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()

        for track_id, box in zip(track_ids, boxes):

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_position = (center_x, center_y)

            movement_status = "UNKNOWN"

            # Calculate movement
            if track_id in track_history:

                previous_x, previous_y = track_history[track_id]

                displacement = math.sqrt(
                    (center_x - previous_x) ** 2
                    + (center_y - previous_y) ** 2
                )

                if displacement < MOVEMENT_THRESHOLD:
                    movement_status = "STOPPED"
                else:
                    movement_status = "MOVING"

            # Check whether vehicle is inside ROI
            if (
                roi_x1 <= center_x <= roi_x2
                and roi_y1 <= center_y <= roi_y2
            ):
                roi_count += 1

                if movement_status == "STOPPED":
                    stopped_count += 1

            # Update track history
            track_history[track_id] = current_position

    # Calculate stopped ratio
    if roi_count > 0:
        stopped_ratio = stopped_count / roi_count
    else:
        stopped_ratio = 0

    # Determine congestion level
    if stopped_ratio < 0.3:
        congestion_level = "LOW"

    elif stopped_ratio < 0.6:
        congestion_level = "MEDIUM"

    else:
        congestion_level = "HIGH"

    # Draw ROI
    cv2.rectangle(
        tracking_frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 0),
        3
    )

    # Display analytics
    cv2.putText(
        tracking_frame,
        f"Vehicles: {roi_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Stopped: {stopped_count}",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Stopped Ratio: {stopped_ratio:.2f}",
        (50, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Congestion: {congestion_level}",
        (50, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Traffic Congestion", tracking_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()