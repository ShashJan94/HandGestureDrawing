Hand Gesture Drawing Application
This is a Python-based application that enables users to draw on a canvas using hand gestures detected via a webcam. The application utilizes Mediapipe for hand tracking, Kalman filters for smoothing movements, and Tkinter for the GUI.

Features
Real-time Hand Tracking: The application uses Mediapipe to detect and track hand landmarks in real-time.
Drawing with Gestures: Draw on the canvas by bringing your index finger and thumb close together. The distance between the two fingers determines whether the system is in drawing mode or erasing mode.
Kalman Filter Smoothing: To enhance the accuracy and stability of drawing, the application applies a Kalman filter to the hand positions.
Interpolation for Smooth Lines: The application interpolates between hand positions to prevent breaks in lines due to frame drops or quick movements.
Erasing: Erase parts of the drawing by using a specific hand gesture (usually a closed fist or a gesture where the index finger and thumb are far apart).
UI Notifications: The application provides real-time feedback when the hand is out of frame or when certain gestures are detected.
Installation
Prerequisites
Python 3.7 or higher
A webcam
Dependencies
The application requires several Python libraries, which can be installed using pip:

bash
Copy code
pip install opencv-python mediapipe numpy Pillow pykalman
Running the Application
Clone the Repository

bash
Copy code
git clone https://github.com/ShashJan94/HandGestureDrawing
cd hand-gesture-drawing
Run the Application

bash
Copy code
python main.py
This will launch the application, opening a window with two panels: one displaying the webcam feed and the other the drawing canvas.

Usage
Basic Controls
Drawing:

To start drawing, position your hand so that the index finger and thumb are close together. The system will recognize this as the drawing gesture.
Move your hand to draw on the canvas.
To stop drawing, separate your fingers or move your hand out of the frame.
Erasing:

To erase, form a closed fist or move your thumb and index finger far apart.
Move your hand over the area you want to erase.
Tips for Best Performance
Lighting: Ensure you have adequate lighting for the webcam to clearly capture your hand movements.
Positioning: Keep your hand within the webcam’s frame for consistent tracking. The system will attempt to interpolate if your hand briefly moves out of frame, but continuous detection is ideal.
Hand Gestures: The system primarily relies on the distance between the index finger and thumb. Practice the gestures to understand how the system responds.
Troubleshooting
Tracking Issues: If the hand tracking seems unstable, try adjusting the lighting or the position of your hand relative to the webcam.
Lagging: If the application is lagging, consider reducing the resolution of the webcam feed or closing other applications to free up system resources.
Broken Lines: If lines are breaking during fast movements, check that your webcam is not dropping frames and that your system can handle real-time video processing.
Future Improvements
Enhanced Gesture Recognition: Add more gestures for additional functionalities such as changing colors or brush sizes.
Undo/Redo Functionality: Implement an undo/redo stack for better user control.
Save/Load Drawings: Allow users to save their drawings to a file or load a previous session.
Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For any major changes, please open an issue to discuss what you would like to change.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Mediapipe: For providing the hand tracking model that powers this application.
OpenCV: For handling video capture and image processing.
Tkinter: For creating the user interface.
Kalman Filter: For smoothing the hand tracking data.
