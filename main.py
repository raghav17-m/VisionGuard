import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

try:
    from playsound import playsound
    SOUND_ON = True
except:
    SOUND_ON = False

# ================= CONFIG =================
VIDEO_INPUT = "traffic.mp4"      # Change to 0 for webcam
VIDEO_OUTPUT = "output.mp4"
ALERT_SOUND = "alert.wav"

CONF_THRESH = 0.4

RESIZE_W = 640
RESIZE_H = 480

# ================= COLORS =================
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)

# ================= LOAD MODEL =================
print("[INFO] Loading YOLOv8...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_INPUT)

fps_src = cap.get(cv2.CAP_PROP_FPS)
if fps_src <= 0:
    fps_src = 25

out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_src,
    (RESIZE_W, RESIZE_H)
)

total_persons = 0
no_helmet_count = 0

last_alert = {}
frame_num = 0
prev_time = time.time()


def play_beep(track_id, frame_id):
    if not SOUND_ON:
        return

    if frame_id - last_alert.get(track_id, -999) > 30:
        last_alert[track_id] = frame_id

        threading.Thread(
            target=playsound,
            args=(ALERT_SOUND,),
            daemon=True
        ).start()


def draw_dashboard(frame, total, violations, fps):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (295, 115),
        DARK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    cv2.putText(
        frame,
        "HELMETGUARD | LIVE",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1
    )

    cv2.putText(
        frame,
        f"Persons detected : {total}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1
    )

    color = RED if violations > 0 else (200, 200, 200)

    cv2.putText(
        frame,
        f"No-helmet alerts : {violations}",
        (18, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1
    )

    cv2.putText(
        frame,
        f"FPS : {fps:.0f}",
        (18, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (130, 130, 130),
        1
    )


print("[INFO] Running. Press Q to stop.")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_num += 1

    frame = cv2.resize(
        frame,
        (RESIZE_W, RESIZE_H)
    )

    fps_live = 1 / (time.time() - prev_time + 1e-9)
    prev_time = time.time()

    results = model(frame, verbose=False)[0]

    motorbikes = []
    persons = []

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        if cls == 3:
            motorbikes.append((x1, y1, x2, y2))

        if cls == 0:
            persons.append(
                (x1, y1, x2, y2,
                 id((x1, y1, x2, y2)))
            )

    # Draw motorbikes
    for x1, y1, x2, y2 in motorbikes:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            ORANGE,
            2
        )

        cv2.putText(
            frame,
            "Motorbike",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ORANGE,
            1
        )

    # Person processing
    for x1, y1, x2, y2, pid in persons:

        total_persons += 1

        person_h = y2 - y1

        head_y2 = y1 + int(person_h * 0.30)

        head_roi = frame[y1:head_y2, x1:x2]

        on_bike = any(
            x1 < (bx1 + bx2) // 2 < x2 and y2 > by1
            for bx1, by1, bx2, by2 in motorbikes
        )

        no_helmet = False

        if on_bike and head_roi.size > 0:

            gray = cv2.cvtColor(
                head_roi,
                cv2.COLOR_BGR2GRAY
            )

            variance = float(gray.var())

            no_helmet = variance > 800

        if no_helmet:

            no_helmet_count += 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                RED,
                3
            )

            cv2.rectangle(
                frame,
                (x1, y1 - 22),
                (x1 + 140, y1),
                RED,
                -1
            )

            cv2.putText(
                frame,
                "NO HELMET",
                (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (RESIZE_W, RESIZE_H),
                RED,
                12
            )

            play_beep(pid, frame_num)

        else:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                GREEN,
                2
            )

            if on_bike:
                cv2.putText(
                    frame,
                    "Helmet OK",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    GREEN,
                    1
                )

    draw_dashboard(
        frame,
        total_persons,
        no_helmet_count,
        fps_live
    )

    out.write(frame)

    cv2.imshow(
        "HelmetGuard (Q to quit)",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done. Persons: {total_persons}")
print(f"No-helmet alerts: {no_helmet_count}")
print(f"Output saved: {VIDEO_OUTPUT}")