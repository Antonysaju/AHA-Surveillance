# AHA-Surveillance
Recognizing faces from live-camera & video!

AHA Surveillance is a face detection and recognition system designed to process live camera feeds and video files in real-time. Using OpenCV and face_recognition libraries, the project detects and identifies known faces from an existing database and provides dynamic recognition in a user-friendly interface.

Features
Real-time Face Recognition: Detects and recognizes faces in real-time from a live camera feed.
Video File Processing: Supports face recognition on pre-recorded video files with adjustable frame rates for faster processing.
Dynamic Text Wrapping: Recognized face names are dynamically wrapped and centered within bounding rectangles, enhancing readability.
Efficient Recognition Logic: Reduces processing load by resizing frames and using a frame-skip mechanism to optimize performance.
User Interface (UI): Simple and intuitive UI built with Tkinter to start the camera or video file recognition.

Technologies Used
OpenCV: For video and image processing, Haar Cascade for face detection.
face_recognition: For accurate face encoding and matching against known faces.
Tkinter: Provides a graphical interface for easy access to functionality.
NumPy: Aids in face encoding distance calculation and data handling.
Project Structure
welcome_window: Initial GUI that allows users to select between live camera and video file recognition modes.
open_camera: Launches the camera feed and performs real-time face detection and recognition.
open_video: Processes a video file, detecting and recognizing faces in each frame at an accelerated speed.
Images Folder: Contains reference images of known faces for encoding.
Haar Cascade: Pre-trained model used for face detection.
How It Works
Welcome Screen: Opens a window where the user can select either live camera feed or video file recognition.
![Screenshot 2024-11-07 184156](https://github.com/user-attachments/assets/9c5bb842-eb74-4747-8388-3b39b0ae9e51)

Face Detection: Detects faces using Haar Cascade and marks them with rectangles.
Green Rectangle: Recognized face.
![Screenshot 2024-11-07 184301](https://github.com/user-attachments/assets/864a9b2b-7eec-4260-8d09-4126438b5823)
![Screenshot 2024-11-07 184247](https://github.com/user-attachments/assets/9d8319c9-68b3-4802-9f13-5056a419e456)

Red Rectangle: Unrecognized face.
Face Recognition: Compares detected faces to the reference images provided, assigning a name if recognized.
Output Summary: On exit, displays a summary of unique recognized faces.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/AHA-Surveillance.git
cd AHA-Surveillance
Install required libraries:

bash
Copy code
pip install opencv-python face_recognition numpy pillow
Place known face images in the project directory, ensuring each face has a unique encoding.

Ensure the haarcascade_frontalface_default.xml file is available in your OpenCV path.

Usage
Run the program:

bash
Copy code
python main.py
In the welcome window, select:

Open Camera: To begin real-time recognition via webcam.
Open Video: To process a video file for face recognition.
Exit the video window by pressing 'q', after which a summary of recognized faces will be displayed.

Result Evaluation
Metric	Description	Evaluation
Face Detection Accuracy	Accurate detection of faces in live/video feed.	High accuracy with minimal false negatives in good lighting conditions.
Recognition Accuracy	Recognizes known faces accurately from provided images.	~95% accuracy with clear frontal images; recognition decreases with occlusions or extreme angles.
Real-time Processing	Achieves real-time processing via frame resizing and skipping techniques.	15-20 FPS on average; smooth recognition optimized for standard webcam feeds.
Text Wrapping	Centers and wraps names dynamically within rectangles for improved readability.	Text wrapping is effective and prevents overlaps.
Error Handling	Handles common errors such as empty cascades and absent video feeds.	Provides robust error messages and smooth fail-safes for typical errors.
Output Display Quality	Clear rectangles around faces with color indicators and names, improving user experience.	Color-coded rectangles (green for recognized, red for unrecognized) enhance readability.
Future Improvements
Expand Database: Add more known faces to improve the robustness of the recognition system.
Enhanced Recognition Accuracy: Apply additional pre-processing for recognition in challenging lighting conditions.
Streamlined UI: Improve the graphical interface to allow easier addition and management of known face encodings.
