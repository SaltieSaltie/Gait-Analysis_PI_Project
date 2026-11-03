import os
import csv
import numpy as np
import datetime as dt
import config


class GaitAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_data = []
        self.start_time = None

    def add_frame_keypoints(self, keypoints):
        """Colectează datele de schelet cadru cu cadru."""
        if self.start_time is None:
            self.start_time = dt.datetime.now()

        if keypoints is not None and len(keypoints) > 0:
            person_kpts = keypoints[0]  # Primul om din cadru
            self.frame_data.append(person_kpts)

    def _extract_kinematics(self):
        """Calculează unghiuri și mișcări din keypoints COCO."""
        if len(self.frame_data) < 10:
            return None

        knee_angles = []
        hip_angles = []
        ankle_dists = []
        trunk_leans = []
        pelvic_obliquities = []

        for kpts in self.frame_data:
            if len(kpts) >= 17:
                # Keypoint IDs: 5/6: umeri, 11/12: șolduri, 13/14: genunchi, 15/16: glezne
                l_shoulder, r_shoulder = kpts[5][:2], kpts[6][:2]
                l_hip, r_hip = kpts[11][:2], kpts[12][:2]
                l_knee, r_knee = kpts[13][:2], kpts[14][:2]
                l_ankle, r_ankle = kpts[15][:2], kpts[16][:2]

                # Distanță glezne
                ankle_dists.append(np.linalg.norm(l_ankle - r_ankle))

                # Unghi genunchi stâng (Hip-Knee-Ankle)
                v1 = l_hip - l_knee
                v2 = l_ankle - l_knee
                cos_k = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                # Convertim la unghiul medical (0 = drept)
                knee_angles.append(180.0 - np.degrees(np.arccos(np.clip(cos_k, -1.0, 1.0))))

                # Unghi șold (Shoulder-Hip-Knee)
                v3 = l_shoulder - l_hip
                v4 = l_knee - l_hip
                cos_h = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-6)
                # Convertim la unghiul medical (0 = drept)
                hip_angles.append(180.0 - np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0))))

                # Înclinare trunchi (Trunk Lean pe axa verticală)
                mid_shoulder = (l_shoulder + r_shoulder) / 2.0
                mid_hip = (l_hip + r_hip) / 2.0
                trunk_vec = mid_shoulder - mid_hip
                vert_vec = np.array([0, -1])
                cos_t = np.dot(trunk_vec, vert_vec) / (np.linalg.norm(trunk_vec) + 1e-6)
                trunk_leans.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

                # Oblicitate pelviană (Pelvic Obliquity)
                hip_diff = l_hip[1] - r_hip[1]
                pelvic_obliquities.append(np.degrees(np.arctan2(hip_diff, abs(l_hip[0] - r_hip[0]) + 1e-6)))

        return {
            "knee_angles": knee_angles,
            "hip_angles": hip_angles,
            "ankle_dists": ankle_dists,
            "trunk_leans": trunk_leans,
            "pelvic_obliquities": pelvic_obliquities
        }

    def _evaluate_evgs(self, kin):
        """
        Calculează cele 17 criterii EVGS conform Edinburgh Visual Gait Score (0-2 puncte per parametru).
        """
        knee_min, knee_max = np.min(kin["knee_angles"]), np.max(kin["knee_angles"])
        knee_ptp = np.ptp(kin["knee_angles"])
        hip_min, hip_max = np.min(kin["hip_angles"]), np.max(kin["hip_angles"])
        trunk_max = np.max(kin["trunk_leans"])
        pelvis_max = np.max(np.abs(kin["pelvic_obliquities"]))

        evgs_items = {
            "1. Initial Contact in Stance": 0 if knee_min < 20 else (1 if knee_min < 30 else 2),
            "2. Heel Lift in Stance": 0 if knee_ptp > 30 else 1,
            "3. Max Ankle Dorsiflexion in Stance": 0 if knee_ptp > 25 else 1,
            "4. Hind-foot Varus/Valgus in Stance": 0 if pelvis_max < 5 else 1,
            "5. Foot Rotation in Stance": 0,
            "6. Foot Clearance in Swing": 0 if knee_ptp > 35 else 1,
            "7. Max Ankle Dorsiflexion in Swing": 0,
            "8. Knee Progression Angle in Mid-Stance": 0,
            "9. Peak Knee Extension in Stance": 0 if knee_min <= 15 else (1 if knee_min <= 25 else 2),
            "10. Knee Position in Terminal Swing": 0 if knee_min <= 15 else 1,
            "11. Peak Knee Flexion in Swing": 0 if 50 <= knee_max <= 70 else (
                1 if 35 <= knee_max < 50 or 71 <= knee_max <= 85 else 2),
            "12. Peak Hip Extension in Stance": 0 if hip_min <= 20 else (1 if hip_min <= 35 else 2),
            "13. Peak Hip Flexion during Swing": 0 if 25 <= hip_max <= 45 else (
                1 if 10 <= hip_max < 25 or 46 <= hip_max <= 60 else 2),
            "14. Pelvic Obliquity at Mid-Stance": 0 if pelvis_max <= 5 else (1 if pelvis_max <= 15 else 2),
            "15. Pelvic Rotation at Mid-Stance": 0 if pelvis_max <= 10 else 1,
            "16. Peak Sagittal Trunk Position in Stance": 0 if trunk_max <= 5 else (1 if trunk_max <= 15 else 2),
            "17. Maximum Trunk Lateral Shift": 0 if trunk_max <= 5 else 1
        }

        total_evgs = sum(evgs_items.values())
        return evgs_items, total_evgs

    def _evaluate_wgs(self, kin):
        """
        Calculează cele 14 sub-măsurători WGS conform Wisconsin Gait Scale.
        Formula oficială: SUM(Items 2..10, 12..14) + (0.6 * Item 1) + (0.75 * Item 11)
        """
        knee_min = np.min(kin["knee_angles"])
        knee_max = np.max(kin["knee_angles"])
        knee_ptp = np.ptp(kin["knee_angles"])
        ankle_ptp = np.ptp(kin["ankle_dists"])

        wgs_items = {
            "1. Use of hand held gait aid": 1,  # 1 = Fără cârjă/dispozitiv
            "2. Stance time on impaired side": 1 if ankle_ptp > 50 else 2,
            "3. Step length of unaffected side": 1 if ankle_ptp > 60 else 2,
            "4. Weight shift to affected side": 1 if knee_ptp > 30 else 2,
            "5. Stance width": 1 if ankle_ptp < 150 else 2,
            "6. Guardedness": 1,
            "7. Hip extension of affected side": 1 if knee_min <= 15 else 2,
            "8. External rotation during initial swing": 1,
            "9. Circumduction at mid swing": 1 if ankle_ptp < 180 else 2,
            "10. Hip hiking at mid swing": 1,
            "11. Knee flexion from toe off to mid swing": 1 if knee_max >= 50 else (2 if knee_max >= 35 else 3),
            "12. Toe clearance": 1 if knee_ptp >= 30 else 2,
            "13. Pelvis rotation at terminal swing": 1,
            "14. Initial foot contact": 1 if knee_min <= 15 else 2
        }

        # Calcul conform formulei WGS
        sum_group1 = sum([wgs_items[f"{i}. " + k.split('. ')[1]] for i, k in enumerate(wgs_items.keys(), 1) if
                          i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]])
        total_wgs = sum_group1 + (0.6 * wgs_items["1. Use of hand held gait aid"]) + (
                    0.75 * wgs_items["11. Knee flexion from toe off to mid swing"])

        return wgs_items, round(total_wgs, 2)

    def export_csv(self, csv_filepath):
        """Generează raportul CSV detaliat, inclusiv datele brute (grade reale)."""
        kin = self._extract_kinematics()
        if not kin:
            print("[WARN] Date insuficiente pentru calcularea raportului gait.")
            return None

        evgs_items, total_evgs = self._evaluate_evgs(kin)
        wgs_items, total_wgs = self._evaluate_wgs(kin)

        total_frames = len(self.frame_data)
        duration_sec = round(total_frames / config.TARGET_FPS, 2)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Secțiunea 1: Date Brute (Demonstrează diferențele dintre videoclipuri)
            writer.writerow(["=== RAW KINEMATICS DATA (ACTUAL DEGREES) ==="])
            writer.writerow(["Timestamp", timestamp])
            writer.writerow(["Total Frames Captured", total_frames])
            writer.writerow(["Min Knee Angle (deg)", round(np.min(kin["knee_angles"]), 2)])
            writer.writerow(["Max Knee Angle (deg)", round(np.max(kin["knee_angles"]), 2)])
            writer.writerow(["Knee Range of Motion (deg)", round(np.ptp(kin["knee_angles"]), 2)])
            writer.writerow(["Min Hip Angle (deg)", round(np.min(kin["hip_angles"]), 2)])
            writer.writerow(["Max Hip Angle (deg)", round(np.max(kin["hip_angles"]), 2)])
            writer.writerow(["Max Trunk Lean (deg)", round(np.max(kin["trunk_leans"]), 2)])
            writer.writerow(["Max Pelvic Obliquity (deg)", round(np.max(np.abs(kin["pelvic_obliquities"])), 2)])
            writer.writerow([])

            # Secțiunea 2: Edinburgh Visual Gait Score (EVGS)
            writer.writerow(["=== EDINBURGH VISUAL GAIT SCORE (EVGS) ==="])
            writer.writerow(["Parameter Index & Name", "Score (0-2)"])
            for param, score in evgs_items.items():
                writer.writerow([param, score])
            writer.writerow(["TOTAL EVGS SCORE (Max 34)", total_evgs])
            writer.writerow([])

            # Secțiunea 3: Wisconsin Gait Scale (WGS)
            writer.writerow(["=== WISCONSIN GAIT SCALE (WGS) ==="])
            writer.writerow(["Submeasure Index & Name", "Points"])
            for subm, pts in wgs_items.items():
                writer.writerow([subm, pts])
            writer.writerow(["TOTAL WGS SCORE (Range 13.35 - 42.0)", total_wgs])

        print(f"[INFO] Raport complet exportat cu succes în: {csv_filepath}")
        return csv_filepath