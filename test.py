import cv2  # Bibliothèque OpenCV

# Charger le modèle de détection de visage (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Ouvrir la caméra (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

while True:
    # Lire une image depuis la caméra
    ret, frame = cap.read()

    # Vérifier si la caméra fonctionne
    if not ret:
        print("Erreur de lecture caméra")
        break

    # Convertir l'image en gris (plus rapide pour détection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # précision
        minNeighbors=5    # moins de faux positifs
    )

    # Dessiner un rectangle autour de chaque visage
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # couleur (vert)
            2
        )

    # Afficher l'image
    cv2.imshow('Detection de visage', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra
cap.release()

# Fermer les fenêtres
cv2.destroyAllWindows()
