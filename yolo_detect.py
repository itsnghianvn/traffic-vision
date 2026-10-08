from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
video = cv2.VideoCapture("data/videos/traffic.mp4")

if not video.isOpened():
    raise ValueError("Could not open video file.")

# Store previous center_y for each track
track_history = {}

# Store IDs that have already been counted
counted_ids = set()

# Traffic statistics
total_count = 0
vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

direction_counts = {
    "UP": 0,
    "DOWN": 0
}

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

    height, width = frame.shape[:2]
    line_y = int(height * 0.7)

    # Draw counting line
    cv2.line(
        tracking_frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 0),
        3
    )

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()

        for track_id, class_id, box in zip(
            track_ids,
            class_ids,
            boxes
        ):
            x1, y1, x2, y2 = box

            center_y = (y1 + y2) // 2

            class_name = result.names[class_id]

            if track_id in track_history:

                previous_y = track_history[track_id]

                direction = None

                # Above → Below
                if previous_y < line_y <= center_y:
                    direction = "DOWN"

                # Below → Above
                elif previous_y > line_y >= center_y:
                    direction = "UP"

                # Count only once per object
                if direction and track_id not in counted_ids:

                    counted_ids.add(track_id)

                    total_count += 1

                    # Count vehicle type
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1

                    # Count direction
                    direction_counts[direction] += 1

                    print(
                        f"ID {track_id} | "
                        f"{class_name} | "
                        f"{direction}"
                    )

            # Update tracking history
            track_history[track_id] = center_y

    # Display analytics
    cv2.putText(
        tracking_frame,
        f"Total: {total_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        tracking_frame,
        f"Cars: {vehicle_counts['car']}",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Motorcycles: {vehicle_counts['motorcycle']}",
        (50, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Buses: {vehicle_counts['bus']}",
        (50, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"Trucks: {vehicle_counts['truck']}",
        (50, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"UP: {direction_counts['UP']}",
        (width - 250, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        tracking_frame,
        f"DOWN: {direction_counts['DOWN']}",
        (width - 250, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Traffic Analytics", tracking_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()