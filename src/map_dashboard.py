import cv2
import json
import numpy as np
import time

print("🗺️ Lancement du Dashboard Holcim - Tracking Multi-Cameras...")

map_width, map_height = 800, 600

while True:
    floor_plan = np.ones((map_height, map_width, 3), dtype=np.uint8) * 40
    
    cv2.rectangle(floor_plan, (440, 210), (680, 450), (0, 0, 150), -1) 
    cv2.putText(floor_plan, "ZONE INTERDITE", (470, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(floor_plan, "HOLCIM - Dashboard Temps Reel (Derniere Position)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    tracking_data_global = {}
    
    try:
        with open("data/last_seen_broyeur.json", "r") as f1:
            tracking_data_global.update(json.load(f1))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
        
    try:
        with open("data/last_seen_four.json", "r") as f2:
            tracking_data_global.update(json.load(f2))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if not tracking_data_global:
        cv2.putText(floor_plan, "En attente des donnees des cameras...", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        for person, info in tracking_data_global.items():
            map_x = int(info['x'] * (map_width / 640))
            map_y = int(info['y'] * (map_height / 480))
            
            last_time = info['time']
            status = info['status']
            departement = info.get('departement', 'Departement Inconnu')
            
            color = (0, 0, 255) if "DANGER" in status else (0, 255, 0)
            if person == "INCONNU": color = (0, 165, 255)
            
            cv2.circle(floor_plan, (map_x, map_y), 15, color, -1)
            cv2.circle(floor_plan, (map_x, map_y), 15, (255, 255, 255), 2)
            
            cv2.putText(floor_plan, f"{person}", (map_x - 20, map_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(floor_plan, f"Vue a: {last_time}", (map_x - 40, map_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(floor_plan, f"Lieu: {departement}", (map_x - 40, map_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    cv2.imshow("Map Holcim - Tracking Multi-Cameras", floor_plan)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()