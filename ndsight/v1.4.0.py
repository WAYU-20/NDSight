# Added screen capture and save file in the same folrder as the script
# Added black and white in the filter toolbox

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Load the full body Haar cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# --- Tkinter GUI integration ---
class App:
    def __init__(self, window, window_title):
        print("App __init__ started")
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            print("ERROR: Could not open video source.")
        self.prev_x = None
        self.direction = ""
        self.error_threshold = 10

        self.brightness_slider = tk.Scale(
            window,
            from_=-100,
            to=100,
            orient=tk.HORIZONTAL,
            label="Brightness",
            length=300
        )
        self.brightness_slider.set(0)  # Default brightness
        self.brightness_slider.pack()

        # Add menu bar for filters
        self.menu_bar = tk.Menu(self.window)
        self.filter_menu = tk.Menu(self.menu_bar, tearoff=0)

        self.filter_menu.add_command(label="No Filter", command=self.set_no_filter)
        self.filter_menu.add_separator()
        self.filter_menu.add_command(label="Grayscale", command=self.set_grayscale)
        self.filter_menu.add_command(label="Black & White", command=self.set_black_white)
        self.menu_bar.add_cascade(label="Filters", menu=self.filter_menu)
        self.window.config(menu=self.menu_bar)

        self.current_filter = "none"

        self.canvas = tk.Label(self.window)
        self.canvas.pack()

        # Capture button
        self.capture_button = tk.Button(self.window, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=5)

        self.captured_frame = None  # Store last frame for capture

        print("App __init__ finished GUI setup")
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_no_filter(self):
        self.current_filter = "none"

    def set_grayscale(self):
        self.current_filter = "grayscale"

    def set_black_white(self):
        self.current_filter = "blackwhite"

    def capture_image(self):
        if self.captured_frame is not None:
            filename = f"capture_{int(cv2.getTickCount())}.png"
            cv2.imwrite(filename, cv2.cvtColor(self.captured_frame, cv2.COLOR_RGB2BGR))
            print(f"Image saved as {filename}")

    def update(self):
        ret, frame = self.vid.read()
        print(f"update called, ret={ret}")
        brightness_value = self.brightness_slider.get()
        if ret:
            # Apply brightness
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_value)
            # Apply filter
            if self.current_filter == "grayscale":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self.current_filter == "blackwhite":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                frame = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            self.captured_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()
            h, w = frame.shape[:2]
            # Orange warning triangle
            warning_tip = (w // 2, 0)
            warning_left = (0, h)
            warning_right = (w, h)
            warning_points = np.array([warning_tip, warning_left, warning_right], dtype=np.int32)
            cv2.fillPoly(overlay, [warning_points], (0, 165, 255))
            # Red danger zone
            danger_top_y = h // 2
            x1, y1 = w // 2, 0
            x2, y2 = 0, h
            if y2 != y1:
                left_x = int(x1 + (danger_top_y - y1) * (x2 - x1) / (y2 - y1))
            else:
                left_x = x1
            x3, y3 = w, h
            if y3 != y1:
                right_x = int(x1 + (danger_top_y - y1) * (x3 - x1) / (y3 - y1))
            else:
                right_x = x1
            danger_top_left = (left_x, danger_top_y)
            danger_top_right = (right_x, danger_top_y)
            danger_bottom_left = (0, h)
            danger_bottom_right = (w, h)
            danger_points = np.array([
                danger_top_left,
                danger_top_right,
                danger_bottom_right,
                danger_bottom_left
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [danger_points], (0, 0, 255))
            alpha = 0.1
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            in_green_zone = False
            in_warning_zone = False
            in_danger_zone = False
            if len(bodies) > 0:
                (x, y, bw, bh) = bodies[0]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                center_x = x + bw // 2
                center_y = y + bh // 2
                pt = (float(center_x), float(center_y))
                if cv2.pointPolygonTest(danger_points.astype(np.float32), pt, False) >= 0:
                    in_danger_zone = True
                elif cv2.pointPolygonTest(warning_points.astype(np.float32), pt, False) >= 0:
                    in_warning_zone = True
                else:
                    in_green_zone = True
                if self.prev_x is not None:
                    diff = center_x - self.prev_x
                    if abs(diff) > self.error_threshold:
                        if diff > 0:
                            self.direction = "Moving Right"
                        else:
                            self.direction = "Moving Left"
                    else:
                        self.direction = "Stationary"
                self.prev_x = center_x
            else:
                self.direction = "No Detection"
                self.prev_x = None
            if self.direction == "Moving Right":
                text = "Right"
            elif self.direction == "Moving Left":
                text = "Left"
            elif self.direction == "Stationary":
                text = "No Motion"
            else:
                text = "No Detection"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_x = 10
            text_y = 30
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            zone_text = None
            zone_color = None
            if in_danger_zone:
                zone_text = "Danger"
                zone_color = (0, 0, 255)
            elif in_warning_zone:
                zone_text = "Warning"
                zone_color = (0, 165, 255)
            elif in_green_zone:
                zone_text = "Safe"
                zone_color = (0, 255, 0)
            if zone_text:
                warn_size, _ = cv2.getTextSize(zone_text, font, 1.2, 3)
                warn_x = (frame.shape[1] - warn_size[0]) // 2
                warn_y = frame.shape[0] // 2
                cv2.putText(frame, zone_text, (warn_x, warn_y), font, 1.2, zone_color, 3, cv2.LINE_AA)
            # Convert BGR to RGB for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)
        self.window.after(10, self.update)

    def on_closing(self):
        self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    print("Starting main...")
    root = tk.Tk()
    App(root, "NDsight - Full Body Detection GUI")
    print("Main finished")
    root.mainloop()
