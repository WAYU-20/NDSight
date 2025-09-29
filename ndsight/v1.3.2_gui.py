
## [v1.3.2] - 2025-09-27
### Move the video on to a Tkinter GUI window
# Added a dropdown menu for future filters (currently disabled)

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import tkinter as tk
from PIL import Image, ImageTk

# Load the full body Haar cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


# --- Tkinter GUI integration ---
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.prev_x = None
        self.direction = ""
        self.error_threshold = 10

        # Add menu bar for future filters
        self.menu_bar = tk.Menu(self.window)
        self.filter_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.filter_menu.add_command(label="No filters yet", state="disabled")
        self.menu_bar.add_cascade(label="Filters", menu=self.filter_menu)
        self.window.config(menu=self.menu_bar)
        self.canvas = tk.Label(window)
        self.canvas.pack()
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
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
            alpha = 0.2
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
    root = tk.Tk()
    App(root, "NDsight - Full Body Detection GUI")
