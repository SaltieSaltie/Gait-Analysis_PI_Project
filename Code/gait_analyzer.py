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
        """Colecteaza datele de schelet cadru cu cadru."""
        if self.start_time is None:
            self.start_time = dt.datetime.now()

        if keypoints is not None and len(keypoints) > 0:
            person_kpts = keypoints[0]  # Primul om din cadru
            self.frame_data.append(person_kpts)

    def _extract_kinematics(self):
        """Calculeaza unghiuri si miscari din keypoints COCO."""
        if len(self.frame_data) < 10:
            return None

        knee_angles = []
        hip_angles = []
        ankle_dists = []
        trunk_leans = []
        pelvic_obliquities = []

        for kpts in self.frame_data:
            if len(kpts) >= 17:
                # Keypoint IDs: 5/6: umeri, 11/12: solduri, 13/14: genunchi, 15/16: glezne
                l_shoulder, r_shoulder = kpts[5][:2], kpts[6][:2]
                l_hip, r_hip = kpts[11][:2], kpts[12][:2]
                l_knee, r_knee = kpts[13][:2], kpts[14][:2]
                l_ankle, r_ankle = kpts[15][:2], kpts[16][:2]

                # Distanta glezne
                ankle_dists.append(np.linalg.norm(l_ankle - r_ankle))

                # Unghi genunchi stang (Hip-Knee-Ankle)
                v1 = l_hip - l_knee
                v2 = l_ankle - l_knee
                cos_k = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                # Convertim la unghiul medical (0 = drept)
                knee_angles.append(180.0 - np.degrees(np.arccos(np.clip(cos_k, -1.0, 1.0))))

                # Unghi sold (Shoulder-Hip-Knee)
                v3 = l_shoulder - l_hip
                v4 = l_knee - l_hip
                cos_h = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-6)
                # Convertim la unghiul medical (0 = drept)
                hip_angles.append(180.0 - np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0))))

                # Inclinare trunchi (Trunk Lean pe axa verticala)
                mid_shoulder = (l_shoulder + r_shoulder) / 2.0
                mid_hip = (l_hip + r_hip) / 2.0
                trunk_vec = mid_shoulder - mid_hip
                vert_vec = np.array([0, -1])
                cos_t = np.dot(trunk_vec, vert_vec) / (np.linalg.norm(trunk_vec) + 1e-6)
                trunk_leans.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

                # Oblicitate pelviana (Pelvic Obliquity)
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
        Calculeaza cele 17 criterii EVGS conform Edinburgh Visual Gait Score (0-2 puncte per parametru).
        Variabilele dinamice pot atinge acum toate treptele de severitate (0, 1, 2).
        """
        knee_min, knee_max = np.min(kin["knee_angles"]), np.max(kin["knee_angles"])
        knee_ptp = np.ptp(kin["knee_angles"])
        hip_min, hip_max = np.min(kin["hip_angles"]), np.max(kin["hip_angles"])
        trunk_max = np.max(kin["trunk_leans"])
        pelvis_max = np.max(np.abs(kin["pelvic_obliquities"]))

        evgs_items = {
            "1. Initial Contact in Stance": 0 if knee_min < 20 else (1 if knee_min < 30 else 2),
            "2. Heel Lift in Stance": 0 if knee_ptp > 30 else (1 if knee_ptp > 15 else 2),
            "3. Max Ankle Dorsiflexion in Stance": 0 if knee_ptp > 25 else (1 if knee_ptp > 10 else 2),
            "4. Hind-foot Varus/Valgus in Stance": 0 if pelvis_max <= 5 else (1 if pelvis_max <= 15 else 2),
            "5. Foot Rotation in Stance": 0,  # Hardcodat
            "6. Foot Clearance in Swing": 0 if knee_ptp >= 35 else (1 if knee_ptp >= 20 else 2),
            "7. Max Ankle Dorsiflexion in Swing": 0,  # Hardcodat
            "8. Knee Progression Angle in Mid-Stance": 0,  # Hardcodat
            "9. Peak Knee Extension in Stance": 0 if knee_min <= 15 else (1 if knee_min <= 25 else 2),
            "10. Knee Position in Terminal Swing": 0 if knee_min <= 15 else (1 if knee_min <= 25 else 2),
            "11. Peak Knee Flexion in Swing": 0 if 50 <= knee_max <= 70 else (
                1 if 35 <= knee_max < 50 or 71 <= knee_max <= 85 else 2),
            "12. Peak Hip Extension in Stance": 0 if hip_min <= 20 else (1 if hip_min <= 35 else 2),
            "13. Peak Hip Flexion during Swing": 0 if 25 <= hip_max <= 45 else (
                1 if 10 <= hip_max < 25 or 46 <= hip_max <= 60 else 2),
            "14. Pelvic Obliquity at Mid-Stance": 0 if pelvis_max <= 5 else (1 if pelvis_max <= 15 else 2),
            "15. Pelvic Rotation at Mid-Stance": 0 if pelvis_max <= 10 else (1 if pelvis_max <= 20 else 2),
            "16. Peak Sagittal Trunk Position in Stance": 0 if trunk_max <= 5 else (1 if trunk_max <= 15 else 2),
            "17. Maximum Trunk Lateral Shift": 0 if trunk_max <= 5 else (1 if trunk_max <= 15 else 2)
        }

        total_evgs = sum(evgs_items.values())
        return evgs_items, total_evgs

    def _evaluate_wgs(self, kin):
        """
        Calculeaza cele 14 sub-masuratori WGS conform Wisconsin Gait Scale.
        Formula oficiala: SUM(Items 2..10, 12..14) + (0.6 * Item 1) + (0.75 * Item 11)
        Extins pentru a acoperi toate treptele posibile (1, 2, 3 si 4 acolo unde e cazul).
        """
        knee_min = np.min(kin["knee_angles"])
        knee_max = np.max(kin["knee_angles"])
        knee_ptp = np.ptp(kin["knee_angles"])
        ankle_ptp = np.ptp(kin["ankle_dists"])

        wgs_items = {
            "1. Use of hand held gait aid": 1,  # Hardcodat: 1 = Fara carja/dispozitiv
            "2. Stance time on impaired side": 1 if ankle_ptp > 60 else (2 if ankle_ptp > 30 else 3),
            "3. Step length of unaffected side": 1 if ankle_ptp > 70 else (2 if ankle_ptp > 40 else 3),
            "4. Weight shift to affected side": 1 if knee_ptp > 30 else (2 if knee_ptp > 15 else 3),
            "5. Stance width": 1 if ankle_ptp < 150 else (2 if ankle_ptp < 200 else 3),
            "6. Guardedness": 1,  # Hardcodat
            "7. Hip extension of affected side": 1 if knee_min <= 15 else (2 if knee_min <= 30 else 3),
            "8. External rotation during initial swing": 1,  # Hardcodat
            "9. Circumduction at mid swing": 1 if ankle_ptp < 180 else (2 if ankle_ptp < 220 else 3),
            "10. Hip hiking at mid swing": 1,  # Hardcodat
            "11. Knee flexion from toe off to mid swing": 1 if knee_max >= 50 else (
                2 if knee_max >= 35 else (3 if knee_max >= 20 else 4)),
            "12. Toe clearance": 1 if knee_ptp >= 30 else (2 if knee_ptp >= 15 else 3),
            "13. Pelvis rotation at terminal swing": 1,  # Hardcodat
            "14. Initial foot contact": 1 if knee_min <= 15 else (2 if knee_min <= 30 else 3)
        }

        # Calcul conform formulei WGS
        sum_group1 = sum([wgs_items[f"{i}. " + k.split('. ')[1]] for i, k in enumerate(wgs_items.keys(), 1) if
                          i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]])
        total_wgs = sum_group1 + (0.6 * wgs_items["1. Use of hand held gait aid"]) + (
                0.75 * wgs_items["11. Knee flexion from toe off to mid swing"])

        return wgs_items, round(total_wgs, 2)
    def export_csv(self, csv_filepath):
        """Genereaza raportul CSV detaliat cu legenda aliniata perfect pe fiecare rand."""
        kin = self._extract_kinematics()
        if not kin:
            print("[WARN] Date insuficiente pentru calcularea raportului gait.")
            return None

        evgs_items, total_evgs = self._evaluate_evgs(kin)
        wgs_items, total_wgs = self._evaluate_wgs(kin)

        total_frames = len(self.frame_data)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Legendele pentru cele doua sectiuni
        evgs_legend = [
            "1. Initial Contact: 0 (Calcai), 1 (Talpa plata), 2 (Varf)",
            "2. Heel Lift in Stance: 0 (Normal), 1 (Devreme/Intarziat), 2 (Fara contact/Forefoot)",
            "3. Max Ankle Dorsiflexion (Stance): 0 (5-25 df), 1 (10pl-4df / 26-40df), 2 (>40df / >10pl)",
            "4. Hindfoot Varus/Valgus: 0 (0-5 valgus), 1 (6-15 valgus / 1-10 varus), 2 (>15 valgus / >10 varus)",
            "5. Foot Rotation in Stance: 0 (0-20 ext), 1 (21-40 ext / 1-25 int), 2 (>40 ext / >25 int)",
            "6. Foot Clearance in Swing: 0 (Complet), 1 (Redus/Pasi inalti), 2 (Zero garda la sol)",
            "7. Max Ankle Dorsiflexion (Swing): 0 (15df-5pl), 1 (16-30df / 6-20pl), 2 (>30df / >20pl)",
            "8. Knee Progression Angle: 0 (Neutru), 1 (Rotatie dar vizibila toata rotula), 2 (Rotula partial ascunsa)",
            "9. Peak Knee Extension (Stance): 0 (0-15 flex), 1 (16-30 flex / 1-10 ext), 2 (>30 flex / >10 ext)",
            "10. Knee Position (Terminal Swing): 0 (5-15 flex), 1 (16-25 flex / 4 flex-10 ext), 2 (>25 flex / >10 ext)",
            "11. Peak Knee Flexion in Swing: 0 (50-70 flex), 1 (35-49 / 71-85 flex), 2 (<35 / >85 flex)",
            "12. Peak Hip Extension (Stance): 0 (0-20 ext), 1 (1-15 flex / 21-35 ext), 2 (>15 flex / >35 ext)",
            "13. Peak Hip Flexion (Swing): 0 (25-45 flex), 1 (10-24 / 46-60 flex), 2 (<10 / >60 flex)",
            "14. Pelvic Obliquity: 0 (0-5 sus), 1 (6-15 sus / 1-10 jos), 2 (>15 sus / >10 jos)",
            "15. Pelvic Rotation: 0 (5 retr-10 pro), 1 (6-15 retr / 11-20 pro), 2 (>15 retr / >20 pro)",
            "16. Peak Sagittal Trunk: 0 (0-5 fwd/back), 1 (6-15 fwd / >5 back), 2 (>15 fwd)",
            "17. Max Trunk Lateral Shift: 0 (Normal), 1 (Moderat/Redus), 2 (Sever)"
        ]

        wgs_legend = [
            "1. Use of gait aid: 1 (Fara), 2 (Minim), 3 (Minim-baza larga), 4 (Marcat), 5 (Marcat-baza larga)",
            "2. Stance time impaired: 1 (Egal), 2 (Inegal), 3 (Foarte scurt)",
            "3. Step length unaffected: 1 (Depaseste), 2 (Nu depaseste/Nu elibereaza), 3 (Pana la/In spate)",
            "4. Weight shift: 1 (Complet), 2 (Scazut), 3 (Foarte limitat)",
            "5. Stance width: 1 (Normal<=1 pantof), 2 (Moderat<=2), 3 (Larg>2)",
            "6. Guardedness: 1 (Fara ezitare), 2 (Usoara), 3 (Ezitare marcata)",
            "7. Hip extension: 1 (Egala), 2 (Usoara flexie), 3 (Extensie marcata)",
            "8. External rotation initial swing: 1 (Egala), 2 (Crescuta), 3 (Marcata)",
            "9. Circumduction mid swing: 1 (Fara), 2 (Moderata), 3 (Marcata)",
            "10. Hip hiking mid swing: 1 (Fara), 2 (Elevare), 3 (Boltire)",
            "11. Knee flexion toe off to mid swing: 1 (Normal), 2 (Ceva flexie), 3 (Minima), 4 (Deloc)",
            "12. Toe clearance: 1 (Normal), 2 (Usoara tarare), 3 (Marcata)",
            "13. Pelvic rotation terminal swing: 1 (Inainte), 2 (Neutru), 3 (Retrasa)",
            "14. Initial foot contact: 1 (Calcai), 2 (Talpa plata), 3 (Fara contact calcai)"
        ]

        # Construim tabelul rand cu rand
        rows = []

        # Sectiunea 1: Raw Kinematics (partea stanga) + Titlul Legendei (partea dreapta, coloana G / index 6)
        rows.append(["=== RAW KINEMATICS DATA (ACTUAL DEGREES) ===", "", "", "", "", "",
                     "--- LEGENDA COMPLETA SCORURI CLINICE ---"])
        rows.append(["Timestamp", timestamp])
        rows.append(["Total Frames Captured", total_frames])
        rows.append(["Min Knee Angle (deg)", round(np.min(kin["knee_angles"]), 2)])
        rows.append(["Max Knee Angle (deg)", round(np.max(kin["knee_angles"]), 2)])
        rows.append(["Knee Range of Motion (deg)", round(np.ptp(kin["knee_angles"]), 2)])
        rows.append(["Min Hip Angle (deg)", round(np.min(kin["hip_angles"]), 2)])
        rows.append(["Max Hip Angle (deg)", round(np.max(kin["hip_angles"]), 2)])
        rows.append(["Max Trunk Lean (deg)", round(np.max(kin["trunk_leans"]), 2)])
        rows.append(["Max Pelvic Obliquity (deg)", round(np.max(np.abs(kin["pelvic_obliquities"])), 2)])

        rows.append([])  # Rand gol despartitor

        # Sectiunea 2: EVGS
        rows.append(["=== EDINBURGH VISUAL GAIT SCORE (EVGS) ===", "", "", "", "", "",
                     "=== EDINBURGH VISUAL GAIT SCORE (EVGS) ==="])
        rows.append(["Parameter Index & Name", "Score (0-2)"])

        # Aliniem fiecare parametru EVGS cu regula lui din legenda
        evgs_keys = list(evgs_items.keys())
        for i in range(len(evgs_keys)):
            param_nume = evgs_keys[i]
            param_scor = evgs_items[param_nume]
            regula_legenda = evgs_legend[i] if i < len(evgs_legend) else ""

            # Adaugam in fisier: [Nume, Scor, gol, gol, gol, gol, Regula]
            rows.append([param_nume, param_scor, "", "", "", "", regula_legenda])

        rows.append(["TOTAL EVGS SCORE (Max 34)", total_evgs])
        rows.append([])  # Rand gol despartitor

        # Sectiunea 3: WGS
        rows.append(["=== WISCONSIN GAIT SCALE (WGS) ===", "", "", "", "", "", "=== WISCONSIN GAIT SCALE (WGS) ==="])
        rows.append(["Submeasure Index & Name", "Points"])

        # Aliniem fiecare parametru WGS cu regula lui din legenda
        wgs_keys = list(wgs_items.keys())
        for i in range(len(wgs_keys)):
            param_nume = wgs_keys[i]
            param_scor = wgs_items[param_nume]
            regula_legenda = wgs_legend[i] if i < len(wgs_legend) else ""

            # Adaugam in fisier: [Nume, Scor, gol, gol, gol, gol, Regula]
            rows.append([param_nume, param_scor, "", "", "", "", regula_legenda])

        rows.append(["TOTAL WGS SCORE (Range 13.35 - 42.0)", total_wgs])

        # Scriem randurile gata prelucrate in fisierul CSV
        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"[INFO] Raport complet exportat cu succes in: {csv_filepath}")
        return csv_filepath