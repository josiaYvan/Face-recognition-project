import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
from tkinter import Tk, simpledialog, messagebox
from PIL import Image

# Initialisation de Tkinter (nécessaire pour utiliser les boîtes de dialogue)
root = Tk()
root.withdraw()  # Masquer la fenêtre principale Tkinter

# Charger les visages connus depuis le dossier images/
known_face_encodings = []
known_face_names = []

def load_known_faces(folder_path="images/"):
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                name = os.path.splitext(filename)[0]  # Utiliser le nom du fichier (sans extension) comme nom de la personne
                known_face_names.append(name)
                print(f"Visage chargé : {name}")
            else:
                print(f"Aucun visage trouvé dans {filename}")

# Charger toutes les images existantes dans le dossier "images/"
load_known_faces()

# Fonction pour enregistrer un nouveau visage
def register_new_face(face_encoding, frame):
    # Demander à l'utilisateur s'il souhaite ajouter ce visage
    if messagebox.askyesno("Nouvelle personne", "Visage inconnu détecté. Voulez-vous enregistrer cette personne ?"):
        # Demander le nom de la nouvelle personne via une boîte de dialogue
        name = simpledialog.askstring("Nom", "Veuillez entrer le nom de la personne :")
        if name:
            # Ajouter le visage à la liste des visages connus
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            
            # Sauvegarder l'image dans le dossier "images/" avec le nom donné
            image_path = f"images/{name}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Nouvelle personne enregistrée : {name}")
            return name
        else:
            messagebox.showerror("Erreur", "Nom invalide.")
    return None

# Initialisation de la webcam
video_capture = cv2.VideoCapture(0)

# Initialisation de certaines variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Dictionnaire pour éviter d'enregistrer plusieurs fois la même présence dans un court laps de temps
attendance_dict = {}

while True:
    # Capture d'une seule image de la webcam
    ret, frame = video_capture.read()

    # Redimensionner l'image pour un traitement plus rapide
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convertir l'image de BGR à RGB
    
    if process_this_frame:
        # Trouver les visages et encoder les visages
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Vérifier si le visage correspond à l'un des visages connus
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Utiliser la distance de visage pour trouver la meilleure correspondance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Si la personne n'est pas reconnue, on propose de l'ajouter
            if name == "Unknown":
                # Proposer d'ajouter le nouveau visage
                new_name = register_new_face(face_encoding, frame)
                if new_name:
                    name = new_name  # Utiliser le nom de la nouvelle personne enregistrée

            face_names.append(name)

            # Enregistrer la présence si elle n'a pas déjà été enregistrée dans un court laps de temps
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if name != "Unknown" and (name not in attendance_dict or (datetime.now() - attendance_dict[name]).seconds > 60):
                # Écrire dans le fichier CSV à chaque détection
                with open('attendance.csv', mode='a', newline='') as attendance_file:
                    fieldnames = ['Name', 'Time']
                    attendance_writer = csv.DictWriter(attendance_file, fieldnames=fieldnames)
                    attendance_writer.writerow({'Name': name, 'Time': current_time})
                    attendance_file.flush()  # Vider le tampon d'écriture immédiatement
                    print(f"{name} enregistré à {current_time}")
                attendance_dict[name] = datetime.now()

    process_this_frame = not process_this_frame

    # Afficher les résultats
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Afficher le nom sous le rectangle
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Afficher l'image résultante
    cv2.imshow('Face Attendance', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
video_capture.release()
cv2.destroyAllWindows()
