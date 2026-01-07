import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import cv2

# Config
DATASET_FILE = "dataset/labeled_actions.csv"
RAW_FILE = "dataset/raw_actions.csv"
CARD_DIR = "dataset/card_images"
LABELS = {
    0: "Skeleton",
    1: "Evo Skeleton",
    2: "Ice Spirit",
    3: "Ice Golem",
    4: "Hog Rider",
    5: "Musketeer",
    6: "Fireball",
    7: "Cannon",
    8: "Evo Cannon",
    9: "The Log",
    -1: "Unknown / Noise"
}
SHORTCUTS = {
    '0': 0, '1': 1, '2': 2, '3': 3,
    '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9
}

class ManualLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clash Royale Data Labeler")
        self.geometry("800x600")
        
        self.current_index = 0
        self.df = self.load_data()
        self.image_cache = {}
        
        # UI
        self.setup_ui()
        
        if not self.df.empty:
            # Jump to first unlabeled
            unlabeled = self.df.index[self.df['unit_id'] == -1].tolist()
            if unlabeled:
                self.current_index = unlabeled[0]
            else:
                self.current_index = 0
            self.show_sample(self.current_index)
            
    def load_data(self):
        df_labeled = pd.DataFrame()
        if os.path.exists(DATASET_FILE):
             print(f"Loading {DATASET_FILE}")
             df_labeled = pd.read_csv(DATASET_FILE)
        
        df_raw = pd.DataFrame()
        if os.path.exists(RAW_FILE):
             print(f"Loading {RAW_FILE}")
             df_raw = pd.read_csv(RAW_FILE)
             
        if df_labeled.empty and df_raw.empty:
            messagebox.showerror("Error", "No dataset found!")
            return pd.DataFrame()
            
        if df_labeled.empty:
            df = df_raw
            df['unit_id'] = -1
        elif df_raw.empty:
            df = df_labeled
        else:
            # Merge
            # Find timestamps in raw that are not in labeled
            existing_timestamps = set(df_labeled['timestamp'].values)
            new_rows = df_raw[~df_raw['timestamp'].isin(existing_timestamps)].copy()
            
            if not new_rows.empty:
                print(f"Found {len(new_rows)} new actions in raw file.")
                new_rows['unit_id'] = -1
                df = pd.concat([df_labeled, new_rows], ignore_index=True)
            else:
                df = df_labeled
                
        # Ensure unit_id column exists
        if 'unit_id' not in df.columns:
            df['unit_id'] = -1
            
        # Prioritize showing -1 (unlabeled) first?
        # Sort so -1 comes first, but keep timestamp order for context?
        # Maybe just jump to first -1.
        return df

    def setup_ui(self):
        # Left Panel: Image
        self.img_panel = tk.Frame(self, bg='gray')
        self.img_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.img_label = tk.Label(self.img_panel, text="No Image", bg='gray')
        self.img_label.pack(expand=True)
        
        self.info_label = tk.Label(self.img_panel, text="", font=("Arial", 12))
        self.info_label.pack(side=tk.BOTTOM, pady=20)

        # Right Panel: Controls
        self.ctrl_panel = tk.Frame(self, width=300)
        self.ctrl_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Navigation
        nav_frame = tk.Frame(self.ctrl_panel)
        nav_frame.pack(pady=10)
        tk.Button(nav_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=5)
        
        # Label Buttons
        lbl_frame = tk.Frame(self.ctrl_panel)
        lbl_frame.pack(pady=20, fill=tk.X)
        
        tk.Label(lbl_frame, text="Assign Label (Key 0-7):").pack()
        
        for uid, name in LABELS.items():
            if uid == -1: continue
            btn = tk.Button(lbl_frame, text=f"[{uid}] {name}", command=lambda u=uid: self.set_label(u), height=2)
            btn.pack(fill=tk.X, padx=10, pady=2)
            
        # Save
        tk.Button(self.ctrl_panel, text="SAVE", command=self.save_data, bg="green", fg="white", height=3).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)
        
        # Key bindings
        self.bind("<Key>", self.handle_keypress)
        self.bind("<Left>", lambda e: self.prev_sample())
        self.bind("<Right>", lambda e: self.next_sample())

    def handle_keypress(self, event):
        if event.char in SHORTCUTS:
            self.set_label(SHORTCUTS[event.char])
            
    def set_label(self, uid):
        if self.df.empty: return
        self.df.at[self.current_index, 'unit_id'] = uid
        print(f"Set index {self.current_index} to {uid} ({LABELS[uid]})")
        self.next_sample()
        
    def next_sample(self):
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.show_sample(self.current_index)
        else:
            messagebox.showinfo("End", "End of dataset reached.")

    def prev_sample(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_sample(self.current_index)

    def show_sample(self, idx):
        row = self.df.iloc[idx]
        img_name = row['card_image']
        current_uid = int(row['unit_id']) if not pd.isna(row['unit_id']) else -1
        
        # Load Image
        path = os.path.join(CARD_DIR, img_name)
        if os.path.exists(path):
            # Scale up for visibility
            cv_img = cv2.imread(path)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img, (200, 250), interpolation=cv2.INTER_NEAREST)
            pil_img = Image.fromarray(cv_img)
            self.photo = ImageTk.PhotoImage(pil_img)
            self.img_label.config(image=self.photo, text="")
        else:
            self.img_label.config(image='', text="Image Not Found")
            
        # Info
        status = f"Index: {idx+1}/{len(self.df)}\n"
        status += f"File: {img_name}\n"
        status += f"Current Label: [{current_uid}] {LABELS.get(current_uid, 'Unknown')}"
        self.info_label.config(text=status)
        
    def save_data(self):
        self.df.to_csv(DATASET_FILE, index=False)
        messagebox.showinfo("Saved", f"Saved to {DATASET_FILE}")

if __name__ == "__main__":
    app = ManualLabeler()
    app.mainloop()
