import threading
import tkinter as tk
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageTk
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

        # Initialize the drawing image
        self.drawing_image = Image.new("RGB", (640, 480), "black")
        self.draw_tool = ImageDraw.Draw(self.drawing_image)

        self.drawing = False
        self.erasing = False
        self.prev_x, self.prev_y = None, None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_drawing = mp.solutions.drawing_utils

        self.update_interval = 1  # Reduced interval for higher frame capture rate

        self.drawing_buffer = deque(maxlen=15)  # Increased buffer for better smoothing
        self.frame_rgb = None

        # Kalman filter setup with more aggressive smoothing
        self.kalman_filter = KalmanFilter(initial_state_mean=np.zeros(4), n_dim_obs=2)
        self.state_mean = np.zeros(4)
        self.state_covariance = np.eye(4)

        self.smooth_factor = 0.7  # Further increased smoothing
        self.base_dist_threshold = 30  # Base distance threshold for drawing state
        self.min_dist_threshold = 20  # Minimum threshold to adapt to faster movements

        self.hand_in_frame = True

        self.drawing_active_time = 10  # Time to allow drawing after the fingers are separated
        self.drawing_counter = 0  # Counter to ensure a stable drawing gesture before drawing
        self.erasing_counter = 0  # Counter to ensure a stable erasing gesture before erasing

        self.draw_stable_counter = 0  # Counter to allow drawing only if gesture is stable for a few frames
        self.draw_stable_threshold = 5  # Number of frames that drawing must be stable

        self.break_drawing = False  # Flag to break drawing continuity

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
                self.draw_canvas.delete("marker")  # Clear previous marker

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
                    self.drawing_buffer.append((pred_x, pred_y))
                    avg_x = int(np.mean([p[0] for p in self.drawing_buffer]))
                    avg_y = int(np.mean([p[1] for p in self.drawing_buffer]))

                    # Determine drawing state based on distance between thumb and index finger
                    distance = np.linalg.norm(np.array([ix, iy]) - np.array([tx, ty]))

                    if distance < self.base_dist_threshold:
                        self.drawing_counter += 1
                        self.erasing_counter = 0
                    else:
                        self.erasing_counter += 1
                        self.drawing_counter = 0
                        self.break_drawing = True  # Break drawing when distance increases

                    # Ensure stable gesture recognition
                    if self.drawing_counter > 3:
                        self.draw_stable_counter += 1
                        if self.draw_stable_counter >= self.draw_stable_threshold:
                            self.drawing = True
                            self.erasing = False
                            self.drawing_active_time = 10  # Reset active drawing time
                    elif self.erasing_counter > 3:
                        self.drawing = False
                        self.erasing = True
                        self.draw_stable_counter = 0
                    else:
                        self.drawing = False
                        self.erasing = False
                        self.draw_stable_counter = 0

                    if self.drawing:
                        self.draw(avg_x, avg_y)
                    elif self.erasing:
                        self.erase(tx, ty)
                    else:
                        self.prev_x, self.prev_y = None, None
            else:
                # If hands are not detected, use the last known position to try and interpolate
                if self.prev_x is not None and self.prev_y is not None:
                    print("Hand out of frame, using previous position for interpolation.")
                    self.draw(self.prev_x, self.prev_y)

                self.hand_in_frame = False
                self.draw_canvas.delete("marker")  # Clear marker when hand is out of frame

    def update_ui(self):
        if self.frame_rgb is not None:
            img = Image.fromarray(self.frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.imgtk = imgtk

        # Update the drawing canvas
        self.update_canvas_image()

        # Display message if hand goes out of frame
        if not self.hand_in_frame:
            self.draw_canvas.create_text(320, 240, text="Hand out of frame", fill="red", font=('Helvetica', 24),
                                         tags="warning")
        else:
            self.draw_canvas.delete("warning")  # Clear the message when the hand is back in frame

        self.root.after(self.update_interval, self.update_ui)

    def draw(self, x, y):
        if self.prev_x is not None and self.prev_y is not None:
            distance = np.linalg.norm(np.array([x, y]) - np.array([self.prev_x, self.prev_y]))
            speed_threshold = 10  # Slightly lower threshold to capture faster movements more accurately

            # Interpolate if the movement is too fast or if there's a significant gap
            if distance > speed_threshold:
                num_interpolations = int(distance / 5)  # Adjust interpolation granularity
                for i in range(1, num_interpolations + 1):
                    interp_x = self.prev_x + (x - self.prev_x) * (i / num_interpolations)
                    interp_y = self.prev_y + (y - self.prev_y) * (i / num_interpolations)
                    self.draw_tool.line([self.prev_x, self.prev_y, interp_x, interp_y], fill="white", width=3)
                    self.prev_x, self.prev_y = interp_x, interp_y
            else:
                # If no significant distance, just draw the line segment
                self.draw_tool.line([self.prev_x, self.prev_y, x, y], fill="white", width=3)

            self.break_drawing = False  # Ensure break drawing is reset after drawing

        # Update the previous point to the current one
        self.prev_x, self.prev_y = x, y

    def erase(self, x, y):
        # Erase a circle around the thumb tip location
        radius = 10
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius

        # Erase on the image
        self.draw_tool.ellipse([x1, y1, x2, y2], fill="black", outline="black")

        # Draw a red marker for the thumb location
        self.draw_canvas.delete("marker")  # Clear previous marker
        self.draw_canvas.create_oval(x1, y1, x2, y2, outline="red", width=2, tags="marker")

        self.update_canvas_image()

    def update_canvas_image(self):
        # Update the Tkinter canvas with the PIL image
        self.imgtk = ImageTk.PhotoImage(image=self.drawing_image)
        self.draw_canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        self.draw_canvas.tag_raise("marker")  # Ensure the marker is on top

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
