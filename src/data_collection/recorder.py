import time
import os
import csv
import mss
import cv2
import numpy as np
from pynput import mouse, keyboard
from datetime import datetime
import threading
import sys

# Adjust sys.path to include src if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.window_manager import WindowManager

# Configuration
OUTPUT_DIR = "images"
LOG_FILE = "log.csv"
FPS = 5  # Capture rate (FPS)

class DataRecorder:
    def __init__(self):
        self.wm = WindowManager()
        self.running = True
        self.mouse_events = []
        self.start_time = None
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
    def transform_click_coordinates(self, x, y, window_rect):
        """
        Transforms global screen coordinates to usage-relative coordinates (0.0-1.0)
        or pixels relative to the game window.
        Returns: (rel_x_pixel, rel_y_pixel)
        """
        top = window_rect['top']
        left = window_rect['left']
        # Coordinates relative to the window
        rel_x = x - left
        rel_y = y - top
        return rel_x, rel_y

    def on_click(self, x, y, button, pressed):
        if not self.running:
            return False # Stop listener
        
        if self.wm.window:
            rect = self.wm.get_window_rect()
            # simple check if click is inside window
            if (rect['left'] <= x <= rect['left'] + rect['width'] and 
                rect['top'] <= y <= rect['top'] + rect['height']):
                
                event_type = "press" if pressed else "release"
                rel_x, rel_y = self.transform_click_coordinates(x, y, rect)
                
                timestamp = time.time()
                self.mouse_events.append({
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "global_x": x,
                    "global_y": y,
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                    "button": str(button)
                })
                # print(f"Click {event_type} at ({rel_x}, {rel_y})")

    def on_press(self, key):
        if key == keyboard.Key.esc:
            print("Stopping recording...")
            self.running = False
            return False

    def capture_loop(self):
        if not self.wm.window:
            print("Window not explicitly selected. Trying auto-detection or selection...")
            # Try auto-detect first if configured, else force selection
            # For now, let's just force selection if not already set.
            if not self.wm.find_window():
                 if not self.wm.select_window():
                     print("No window selected. Exiting capture loop.")
                     self.running = False
                     return

        with mss.mss() as sct:
            while self.running:
                # Window might be closed or lost, check validity if possible?
                # accessing self.wm.window properties usually works if window is open
                try:
                    rect = self.wm.get_window_rect()
                except Exception as e:
                    print(f"Window lost: {e}")
                    self.running = False
                    break

                rect = self.wm.get_window_rect()
                # mss requires dict: top, left, width, height
                monitor = {
                    "top": rect["top"],
                    "left": rect["left"],
                    "width": rect["width"],
                    "height": rect["height"]
                }
                
                # Capture
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Convert for storage
                
                # Timestamp filename
                timestamp = time.time()
                filename = f"{timestamp:.3f}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Save
                cv2.imwrite(filepath, frame)
                
                # Sleep to match FPS
                time.sleep(1.0 / FPS)

    def run(self):
        print("Starting recorder.")
        
        # Select window first
        if not self.wm.select_window():
            print("Window selection failed. Aborting.")
            return

        print("Press ESC to stop.")
        self.start_time = time.time()
        
        # Start hooks
        mouse_listener = mouse.Listener(on_click=self.on_click)
        key_listener = keyboard.Listener(on_press=self.on_press)
        
        mouse_listener.start()
        key_listener.start()
        
        self.capture_loop()
        
        mouse_listener.stop()
        key_listener.join()
        
        self.save_logs()

    def save_logs(self):
        print(f"Saving {len(self.mouse_events)} events to {LOG_FILE}...")
        file_exists = os.path.isfile(LOG_FILE)
        
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "event_type", "global_x", "global_y", "rel_x", "rel_y", "button"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.mouse_events)
        print("Done.")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()
