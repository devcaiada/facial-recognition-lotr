import face_recognition
import cv2
import numpy as np
from pathlib import Path

input_dir = Path('input')
output_dir = Path('output')
images_dir = Path('images')
output_dir.mkdir(parents=True, exist_ok=True)

known_face_encodings = []
known_face_names = []

character_images = {
    "Frodo": "frodo.jpg",
    "Sam": "sam.jpg",
    "Aragorn": "aragorn.jpg",
    "Legolas": "legolas.jpg",
    "Gandalf": "gandalf.jpg",
    "Merry": "merry.jpg"
}

for name, filename in character_images.items():
    image_path = images_dir / filename
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    else:
        print(f"Não foi possível detectar um rosto em {filename}")

input_image_path = input_dir / 'elenco.jpg'
image = cv2.imread(str(input_image_path))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

face_names = []

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Desconhecido"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

output_image_path = output_dir / 'elenco_annotated.jpg'
cv2.imwrite(str(output_image_path), image)

print(f'Imagem anotada salva em: {output_image_path}')