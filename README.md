A Python-based application that enables users to draw on a canvas using hand gestures detected via a webcam. The application utilizes Mediapipe for hand tracking, Kalman filters for smoothing movements, and Tkinter for the GUI.

Features
Real-time Hand Tracking: Utilizes Mediapipe to detect and track hand landmarks in real-time.
Drawing with Gestures: Draw on the canvas by bringing your index finger and thumb close together. The distance between the two fingers determines whether the system is in drawing or erasing mode.
Kalman Filter Smoothing: Applies a Kalman filter to the hand positions to enhance accuracy and stability.
Interpolation for Smooth Lines: Prevents breaks in lines due to frame drops or quick movements by interpolating between hand positions.
Erasing Mode: Erase parts of the drawing using a specific hand gesture (e.g., closed fist or wide separation between thumb and index finger).
UI Notifications: Provides real-time feedback when the hand is out of frame or when certain gestures are detected.
Installation
Prerequisites
Python 3.7 or higher
A webcam
Dependencies
Install the required Python libraries using pip:

bash
Copy code
pip install opencv-python mediapipe numpy Pillow pykalman
Running the Application
Clone the Repository

bash
Copy code
git clone https://github.com/ShashJan94/HandGestureDrawing.git
cd HandGestureDrawing
Run the Application

bash
Copy code
python main.py
This will launch the application, opening a window with two panels: one displaying the webcam feed and the other the drawing canvas.

Usage
Basic Controls
Drawing:

Position your hand so that the index finger and thumb are close together. The system will recognize this as the drawing gesture.
Move your hand to draw on the canvas.
To stop drawing, separate your fingers or move your hand out of the frame.
Erasing:

Form a closed fist or move your thumb and index finger far apart.
Move your hand over the area you want to erase.
Tips for Best Performance
Lighting: Ensure adequate lighting for the webcam to clearly capture your hand movements.
Positioning: Keep your hand within the webcam’s frame for consistent tracking. The system will attempt to interpolate if your hand briefly moves out of frame, but continuous detection is ideal.
Hand Gestures: The system primarily relies on the distance between the index finger and thumb. Practice the gestures to understand how the system responds.
Troubleshooting
Tracking Issues: Adjust lighting or the position of your hand relative to the webcam if tracking is unstable.
Lagging: If the application is lagging, consider reducing the resolution of the webcam feed or closing other applications to free up system resources.
Broken Lines: If lines are breaking during fast movements, ensure your webcam is not dropping frames and that your system can handle real-time video processing.
Future Improvements
Enhanced Gesture Recognition: Add more gestures for additional functionalities, such as changing colors or brush sizes.
Undo/Redo Functionality: Implement an undo/redo stack for better user control.
Save/Load Drawings: Allow users to save their drawings to a file or load a previous session.
Contributing
Contributions are welcome! If you'd like to contribute to this project:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
Please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Mediapipe: For providing the hand tracking model that powers this application.
OpenCV: For handling video capture and image processing.
Tkinter: For creating the user interface.
Kalman Filter: For smoothing the hand tracking data.
