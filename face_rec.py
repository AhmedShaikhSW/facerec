## MacOS DStore error fix :
## find . -name "*.DS_Store" -type f -delete
## in terminal

import os
import face_recognition
import cv2

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"   # can also use hog, better for cpu-only

video = cv2.VideoCapture(0)

print("Loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
    # can also use os.join for better os independency
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoded_image = face_recognition.face_encodings(image)[0]
        known_faces.append(encoded_image)
        known_names.append(name)

print("Checking unknown faces...")

while True:
    ret, image = video.read()

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match! {match}")
            color = [0, 255, 0]

            top_left = (face_location[0], face_location[3])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(
                image, match, (face_location[3]+10, face_location[2]+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS
            )

    cv2.imshow(filename, image)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # break if user presses the q key
        break
