import cv2
import face_recognition
import numpy as np
import os
from google.colab import drive

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces" # Directory to store images of known individuals
TOLERANCE = 0.6               # How much distance between faces to consider it a match (lower is stricter)
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog"                 # 'hog' for CPU, 'cnn' for GPU (more accurate, slower without GPU)
OUTPUT_VIDEO_PATH = "output_video.mp4" # Path to save the output video

# --- 1. Create known_faces directory if it doesn't exist ---
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"Created directory: {KNOWN_FACES_DIR}. Please add your known faces images here.")
    # Exit or add a mechanism to wait for user to add images

# --- 2. Load Known Faces and Encode Them ---
print("Loading known faces...")

known_faces_encodings = []
known_faces_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    # Ensure it's a directory
    if not os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
        continue

    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
        # Check for common image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load image
        image_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        print(f"Loading image: {image_path}")
        image = face_recognition.load_image_file(image_path)

        # Get face encodings (assuming one face per image for known individuals)
        # Using model='cnn' can improve accuracy if you have a GPU
        encodings = face_recognition.face_encodings(image, num_jitters=1, model=MODEL)

        if encodings: # Ensure a face was actually found
            known_faces_encodings.append(encodings[0])
            known_faces_names.append(name)
        else:
            print(f"Warning: No face found in {filename} for {name}. Skipping.")

print(f"Loaded {len(known_faces_encodings)} known faces.")

if not known_faces_encodings:
    print("Error: No known faces loaded. Please add images to the known_faces directory.")
    # Exit or handle the case where no known faces are found

# --- 3. Initialize Video Capture ---
# For webcam: 0, 1, etc.
# For video file: "path/to/video.mp4"
video_path = "path/to/your/video.mp4" # Replace with your video file path
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video stream from {video_path}.")
    exit()

print("Processing video stream...")

# --- 4. Setup Video Writer ---
# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# 'mp4v' is a common codec, you might need to try others like 'XVID' or 'MJPG'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# --- 5. Process Video Frame by Frame ---
frame_count = 0
while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Finished processing video or could not read frame.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Convert the image from BGR color (OpenCV default) to RGB color (face_recognition needs)
    rgb_frame = frame[:, :, ::-1] # Efficient way to convert BGR to RGB

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the current frame
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Compare current face with known faces
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding, TOLERANCE)
        name = "Unknown"

        # Find the best match (smallest distance)
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS) # Green box

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), FONT_THICKNESS) # White text

    # Write the frame to the output video
    out.write(frame)

# Release handles
video_capture.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved as {OUTPUT_VIDEO_PATH}")
