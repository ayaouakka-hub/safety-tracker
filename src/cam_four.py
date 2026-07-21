import json
import cv2
import torch
import numpy as np
import winsound
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cpu')
yolo_model = YOLO("models/yolov8n.pt")
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

data = torch.load("models/embeddings.pt", map_location=device)
known_embeddings, known_names = data['embeddings'], data['names']

THRESHOLD = 0.9
TIMEOUT_ABSENCE = 5.0
CLEANUP_TIMEOUT = 30.0
MAX_DISPLAY = 5
RECOGNIZE_EVERY = 10

CAMERA_NAME = "Departement Four - Cam Phone" 

tracked_persons = {}

LOG_FILE = "docs/log_localisation.csv"
os.makedirs("docs", exist_ok=True)
log_exists = os.path.isfile(LOG_FILE)
log_file = open(LOG_FILE, "a", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
if not log_exists:
    log_writer.writerow(["timestamp", "id", "nom", "departement", "zone", "event"])

def log_event(pid, name, zone, event):
    log_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pid, name, CAMERA_NAME, zone, event])
    log_file.flush()

cap = cv2.VideoCapture("http://192.168.1.14:8080/video") 

ret, first_frame = cap.read()
if not ret:
    print("❌ ERREUR: Ma qdertsh n-t-connecta l'Camera d'Telephone. T2ked mn l-IP w WiFi!")
    exit()

first_frame = cv2.resize(first_frame, (640, 480))
h_frame, w_frame = first_frame.shape[:2]

zone_interdite = np.array([
    [int(w_frame * 0.55), int(h_frame * 0.35)],
    [int(w_frame * 0.85), int(h_frame * 0.35)],
    [int(w_frame * 0.85), int(h_frame * 0.75)],
    [int(w_frame * 0.55), int(h_frame * 0.75)],
], np.int32).reshape((-1, 1, 2))

zone_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
cv2.fillPoly(zone_mask, [zone_interdite], 255)

intrus_alarm_active = False
zone_alarm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))
    now = time.time()

    cv2.polylines(frame, [zone_interdite], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "ZONE INTERDITE", (zone_interdite[0][0][0], zone_interdite[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    results = yolo_model.track(frame, classes=[0], device='cpu', persist=True,
                                tracker="bytetrack_custom.yaml", verbose=False)

    current_ids = set()
    unknown_present = False
    known_in_zone = False

    for r in results:
        if r.boxes.id is None:
            continue
        for box, track_id in zip(r.boxes, r.boxes.id):
            pid = int(track_id.item())
            current_ids.add(pid)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w_frame, x2), min(h_frame, y2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            person_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
            cv2.rectangle(person_mask, (x1c, y1c), (x2c, y2c), 255, -1)
            overlap = cv2.bitwise_and(zone_mask, person_mask)
            is_in_zone = cv2.countNonZero(overlap) > 0
            zone_label = "ZONE_INTERDITE" if is_in_zone else "ZONE_NORMALE"

            state = tracked_persons.get(pid, {
                "name": "INCONNU", "zone": zone_label, "last_seen": now,
                "frame_count": 0, "alerted_missing": False
            })

            state["frame_count"] += 1
            if state["frame_count"] % RECOGNIZE_EVERY == 1 or state["name"] == "INCONNU":
                person_crop = frame[y1c:y2c, x1c:x2c]
                
                if person_crop.size > 0:
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face = mtcnn(rgb_crop)
                    
                    if face is not None:
                        emb = resnet(face.unsqueeze(0).to(device))
                        for name, k_emb in zip(known_names, known_embeddings):
                            if (emb - k_emb).norm().item() < THRESHOLD:
                                state["name"] = name 
                                break

            state["zone"] = zone_label
            state["last_seen"] = now
            state["alerted_missing"] = False
            state["x"] = cx 
            state["y"] = cy
            tracked_persons[pid] = state

            identity = state["name"]
            color = (0, 255, 0) if identity != "INCONNU" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID{pid}:{identity}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            if identity == "INCONNU":
                unknown_present = True
                cv2.putText(frame, "!!! PERSONNE NON IDENTIFIEE !!!", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if not intrus_alarm_active:
                    log_event(pid, identity, zone_label, "INTRUS_DETECTE")

            if is_in_zone and identity != "INCONNU":
                known_in_zone = True
                cv2.putText(frame, f"!!! {identity} EN ZONE INTERDITE !!!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
                if not zone_alarm_active:
                    log_event(pid, identity, zone_label, "ZONE_INTERDITE")

    if unknown_present:
        if not intrus_alarm_active:
            winsound.PlaySound("data/alerte.wav", winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            intrus_alarm_active = True
            zone_alarm_active = False
    elif known_in_zone:
        if not zone_alarm_active:
            winsound.PlaySound("data/autorise.wav", winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            zone_alarm_active = True
            intrus_alarm_active = False
    else:
        if intrus_alarm_active or zone_alarm_active:
            winsound.PlaySound(None, winsound.SND_PURGE)
            intrus_alarm_active = False
            zone_alarm_active = False

    to_delete = [pid for pid, s in tracked_persons.items()
                 if pid not in current_ids and (now - s["last_seen"]) > CLEANUP_TIMEOUT]
    for pid in to_delete:
        del tracked_persons[pid]

    missing = [(pid, s) for pid, s in tracked_persons.items()
               if pid not in current_ids and (now - s["last_seen"]) > TIMEOUT_ABSENCE]
    missing.sort(key=lambda x: now - x[1]["last_seen"])

    y_offset = 110
    for pid, state in missing[:MAX_DISPLAY]:
        msg = f"ID{pid} ({state['name']}) NON LOCALISE - dernier lieu: {CAMERA_NAME} il y a {int(now - state['last_seen'])}s"
        cv2.putText(frame, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_offset += 30
        if not state["alerted_missing"]:
            log_event(pid, state["name"], state["zone"], "NON_LOCALISE")
            state["alerted_missing"] = True
            
    try:
        export_data = {}
        for pid_export, s_export in tracked_persons.items():
            unique_name = f"{s_export['name']} (ID:{pid_export})"
            export_data[unique_name] = {
                "x": s_export.get("x", 0),
                "y": s_export.get("y", 0),
                "time": datetime.fromtimestamp(s_export["last_seen"]).strftime("%H:%M:%S"),
                "status": "DANGER" if s_export["zone"] == "ZONE_INTERDITE" else "SAFE",
                "departement": CAMERA_NAME 
            }
        with open("data/last_seen_four.json", "w") as f:
            json.dump(export_data, f)
    except Exception as e:
        pass

    cv2.imshow(f"Camera de Surveillance - {CAMERA_NAME}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

winsound.PlaySound(None, winsound.SND_PURGE)
cap.release()
cv2.destroyAllWindows()
log_file.close()