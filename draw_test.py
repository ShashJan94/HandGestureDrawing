import threading
import tkinter as tk
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from pykalman import KalmanFilter


class HandGestureDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Drawing")

        # Create a frame to hold the canvases
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        # Create a canvas for the webcam feed
        self.video_canvas = tk.Canvas(self.frame, width=640, height=480, bg="white")
        self.video_canvas.grid(row=0, column=0)

        # Create a canvas for drawing
        self.draw_canvas = tk.Canvas(self.frame, width=640, height=480, bg="black")
        self.draw_canvas.grid(row=0, column=1)

        self.cap = self.try_open_camera()
        if self.cap is None:
            print("Failed to open webcam")
            return

        self.drawing = False
        self.erasing = False
        self.prev_x, self.prev_y = None, None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.update_interval = 1  # Reduced interval for higher frame capture rate

        self.drawing_buffer = deque(maxlen=5)
        self.frame_rgb = None

        # Kalman filter setup with more aggressive smoothing
        self.kalman_filter = KalmanFilter(initial_state_mean=np.zeros(4), n_dim_obs=2)
        self.state_mean = np.zeros(4)
        self.state_covariance = np.eye(4)

        self.smooth_factor = 0.7  # Further increased smoothing
        self.base_dist_threshold = 25  # Base distance threshold for drawing state

        # Buffer to store the last few points for multi-frame averaging
        self.position_buffer = deque(maxlen=7)
        self.hand_in_frame = True

        self.drawing_active_time = 10  # Time to allow drawing after the fingers are separated

        self.thread = threading.Thread(target=self.update_canvas)
        self.thread.daemon = True
        self.thread.start()

        self.update_ui()

    def try_open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam is in use by another application. Waiting for it to be available...")
            time.sleep(1)
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        print("Webcam is now available.")
        return cap

    def update_canvas(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            frame = cv2.flip(frame, 1)
            self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(self.frame_rgb)

            if results.multi_hand_landmarks:
                self.hand_in_frame = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(self.frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                    # Get the coordinates
                    ix, iy = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                    tx, ty = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

                    # Kalman filter prediction and update
                    self.state_mean, self.state_covariance = self.kalman_filter.filter_update(
                        self.state_mean,
                        self.state_covariance,
                        np.array([ix, iy])
                    )

                    pred_x, pred_y = self.state_mean[0], self.state_mean[1]

                    # Apply temporal smoothing with multi-frame averaging
                    self.position_buffer.append((pred_x, pred_y))
                    avg_x = int(np.mean([p[0] for p in self.position_buffer]))
                    avg_y = int(np.mean([p[1] for p in self.position_buffer]))

                    # Determine drawing or erasing state
                    distance = np.linalg.norm(np.array([ix, iy]) - np.array([tx, ty]))

                    if distance < 10:  # Small distance indicates drawing gesture
                        self.drawing = True
                        self.erasing = False
                        self.drawing_active_time = 10  # Reset active drawing time
                    else:  # Thumb alone controls the erasing
                        self.drawing = False
                        self.erasing = True

                    # Clear old marker
                    self.draw_canvas.delete("marker")

                    if self.drawing:
                        self.draw(avg_x, avg_y)
                        # Moving marker during drawing
                        self.draw_canvas.create_oval(avg_x - 5, avg_y - 5, avg_x + 5, avg_y + 5, outline="green", width=2, tags="marker")
                    elif self.erasing:
                        self.erase(tx, ty)
                        # Moving marker during erasing
                        self.draw_canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, outline="green", width=2, tags="marker")
                    else:
                        self.prev_x, self.prev_y = None, None

            else:
                self.hand_in_frame = False
                # Clear all markers when the hand is out of frame
                self.draw_canvas.delete("marker")

    def update_ui(self):
        if self.frame_rgb is not None:
            img = Image.fromarray(self.frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.imgtk = imgtk

        # Display message if hand goes out of frame
        if not self.hand_in_frame:
            self.draw_canvas.create_text(320, 240, text="Hand out of frame", fill="red", font=('Helvetica', 24), tags="warning")
        else:
            self.draw_canvas.delete("warning")  # Clear the message when the hand is back in frame

        self.root.after(self.update_interval, self.update_ui)

    def draw(self, x, y):
        if self.prev_x is not None and self.prev_y is not None:
            if abs(x - self.prev_x) > 50 or abs(y - self.prev_y) > 50:  # Check for large jumps
                print("Skipping large jump")
                self.prev_x, self.prev_y = x, y
                return

            self.draw_canvas.create_line(self.prev_x, self.prev_y, x, y, fill="white", width=3)  # Drawing in white
        self.prev_x, self.prev_y = x, y

    def erase(self, x, y):
        self.draw_canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="black", outline="red", tags="marker")  # Red eraser

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = HandGestureDrawingApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except ValueError as e:
        print(e)
