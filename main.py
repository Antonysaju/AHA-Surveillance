import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Load Haar Cascade for face detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Load face images and encodings for face recognition
face1 = face_recognition.load_image_file("untitled.jpg")
face1_encoding = face_recognition.face_encodings(face1)[0]

face2 = face_recognition.load_image_file("untitled2.jpg")
face2_encoding = face_recognition.face_encodings(face2)[0]

face3 = face_recognition.load_image_file("untitled3.jpg")
face3_encoding = face_recognition.face_encodings(face3)[0]

known_face_encodings = [face1_encoding, face2_encoding, face3_encoding]
known_face_names = ["Antony Saju David", "Adithya A", "Hemachandru E"]

# Function to handle face recognition using the webcam
def open_camera():
    video_capture = cv2.VideoCapture(0)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    recognized_faces = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame for faster processing and convert to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    recognized_faces.append(name)

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Draw rectangles and labels on the detected faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Set color based on whether the face is recognized
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw the rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Prepare to handle text wrapping inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.75  # Reduced font size
            thickness = 1
            text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]

            # Wrap the text if it exceeds the width of the rectangle
            max_width = right - left
            if text_size[0] > max_width:
                # Split the text into two lines
                wrapped_name = name.split(' ')
                lines = []
                line = ""

                for word in wrapped_name:
                    # Add words to the current line until it exceeds the max width
                    test_line = line + word + " "
                    if cv2.getTextSize(test_line, font, font_scale, thickness)[0][0] <= max_width:
                        line = test_line
                    else:
                        lines.append(line.strip())
                        line = word + " "
                lines.append(line.strip())

                # Adjust the bottom rectangle to fit the lines of text
                line_height = text_size[1] + 8  # Padding reduced for neatness
                total_height = line_height * len(lines)
                cv2.rectangle(frame, (left, bottom), (right, bottom + total_height), color, cv2.FILLED)

                # Draw each line of text inside the rectangle, centered
                for i, line in enumerate(lines):
                    text_size_line = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    text_x = left + (max_width - text_size_line[0]) // 2  # Center the text
                    y = bottom + (i + 1) * line_height - 6
                    cv2.putText(frame, line, (text_x, y), font, font_scale, (255, 255, 255), thickness)

            else:
                # If no wrapping is needed, display the text normally and center it
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                text_x = left + (max_width - text_size[0]) // 2  # Center the text
                cv2.putText(frame, name, (text_x, bottom - 6), font, font_scale, (255, 255, 255), thickness)

        cv2.imshow('Video', frame)

        # Exit on pressing 'q' and print recognized face data
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show the final summary in the console
    recognized_faces_unique = list(set(recognized_faces))  # Unique faces
    print(f"Total recognized faces: {len(recognized_faces_unique)}")
    print("Recognized faces:", recognized_faces_unique)

    video_capture.release()
    cv2.destroyAllWindows()




# Function to handle face detection and recognition from a video file
def open_video():
    # Load the cascade file for face detection
    cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Use OpenCV's built-in path
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Check if the cascade has been loaded properly
    if faceCascade.empty():
        print("Error loading cascade file. Check the path!")
        return

    video = cv2.VideoCapture("video.mp4")  # Replace with your video file path

    recognized_faces = []

    # Set the desired output frame size, e.g., 640x360 (similar to camera window)
    output_width = 640
    output_height = 360

    frame_skip = 6

    frame_count = 0  # Track frame count for skipping
    while True:
        ret, frame = video.read()

        if not ret:
            print("End of video or video not loaded properly.")
            break

        frame_count += 1

        # Skip every second frame to speed up the processing
        if frame_count % frame_skip != 0:
            continue

        # Resize the frame to a smaller window (e.g., 640x360)
        frame = cv2.resize(frame, (output_width, output_height))

        gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale video frame
        faces = faceCascade.detectMultiScale(
            gray_video,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region for recognition
            face_image = frame[y:y + h, x:x + w]
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Compute the face encoding for the detected face
            face_encoding = face_recognition.face_encodings(rgb_face)

            if face_encoding:  # If face encoding is found
                face_encoding = face_encoding[0]
                
                # Compare the detected face encoding with known face encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                name = "Unknown"  # Default name if not recognized
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                # Check if there is a match
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                recognized_faces.append(name)  # Append recognized face name

        cv2.imshow("Video", frame)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord("q"):  # Adjust the delay to keep smoother playback with 1.5x speed
            break

    # Show the recognized faces
    recognized_faces_unique = list(set(recognized_faces))  # Unique faces
    print("Recognized faces:", recognized_faces_unique)
    print(f"Total recognized faces: {len(recognized_faces_unique)}")  # Count of recognized faces

    video.release()
    cv2.destroyAllWindows()




# Function to create a welcome window
def welcome_window():
    root = tk.Tk()
    root.title("AHA Surveillance")

    # Set window size and make it non-resizable
    root.geometry("800x600")
    root.resizable(False, False)

    # Add a title label
    title_label = tk.Label(root, text="Welcome to AHA Surveillance", font=("Helvetica", 24), pady=20)
    title_label.pack()

    # Add an image 
    img = Image.open("logo.png")  # Add your logo image here
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img)
    img_label.pack()

    # Create a frame for the buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=40)

    button_width = int(root.winfo_width() * 0.8 / 2)

    # Button for webcam face recognition
    webcam_button = tk.Button(button_frame, text="Open Camera", font=("Helvetica", 16), bg="#4CAF50", fg="white",
                              command=lambda: [root.destroy(), open_camera()], padx=20, pady=10)
    webcam_button.grid(row=0, column=0, padx=10)

    # Button for video face recognition
    video_button = tk.Button(button_frame, text="Open Video", font=("Helvetica", 16), bg="#2196F3", fg="white",
                             command=lambda: [root.destroy(), open_video()], padx=20, pady=10)
    video_button.grid(row=0, column=1, padx=10)

    # Make buttons occupy about 80% of the window width
    webcam_button.config(width=button_width)
    video_button.config(width=button_width)

    root.mainloop()

# Run the welcome window
welcome_window()
