import pandas as pd
import cv2
import os
import glob
import numpy as np

# Configuration
LOG_FILE = "log.csv"
IMAGE_DIR = "images"
OUTPUT_DIR = "dataset/card_images"
DATASET_FILE = "dataset/raw_actions.csv"

# Thresholds
DECK_Y_THRESHOLD = 760  # Y coordinate splitting Arena and Deck (approx 80% of 975)
MAX_DELAY_SEC = 3.0     # Max time between selecting card and placing it
CARD_CROP_SIZE = (80, 100) # (Width, Height) to crop around click

def load_data():
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return None
    df = pd.read_csv(LOG_FILE)
    return df

def find_closest_image(timestamp, image_files):
    # image_files: list of (timestamp, filepath)
    # Find the image with timestamp closest to event timestamp (and preferably before)
    # Since we capture at 5 FPS, there should be one close.
    
    # Simple search (can be optimized)
    best_file = None
    min_diff = 1.0 # Max 1 sec diff
    
    for img_ts, filepath in image_files:
        diff = abs(img_ts - timestamp)
        if diff < min_diff:
            min_diff = diff
            best_file = filepath
            
    return best_file

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_data()
    if df is None:
        return

    # Load all image timestamps
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    image_files = []
    for p in image_paths:
        try:
            ts = float(os.path.basename(p).replace(".jpg", ""))
            image_files.append((ts, p))
        except ValueError:
            pass
    image_files.sort(key=lambda x: x[0])
    
    # Sort by timestamp just in case
    df = df.sort_values('timestamp')
    events = df.to_dict('records')
    
    actions = []
    
    # State tracking
    # We want to catch:
    # 1. Click Deck (Press) -> ... -> Click Arena (Press) [Tap Selection]
    # 2. Click Deck (Press) -> ... -> Release Arena [Drag Selection]
    
    active_deck_press = None # Stores the press event started in deck
    
    for ev in events:
        if ev['rel_y'] > DECK_Y_THRESHOLD:
            # Event in Deck Area
            if ev['event_type'] == 'press':
                # Start of potential action (Tap select or Drag start)
                active_deck_press = ev
            else:
                # Release in deck? meaning just selected a card (Tap select case completed phase 1)
                # We interpret "Active Deck Press" as the selection event.
                # If we release in deck, we still hold the "selected card" state in game until we tap arena?
                # Actually, in CR, if you tap card, it selects. If you drag and drop back in deck, it cancels.
                # Let's assume 'active_deck_press' stays valid until a timeout or a new press.
                pass
        else:
            # Event in Arena Area
            if not active_deck_press:
                continue

            # Check timeout
            if (ev['timestamp'] - active_deck_press['timestamp']) > MAX_DELAY_SEC:
                active_deck_press = None
                continue

            # Case A: Drag & Drop (Press Deck -> Release Arena)
            if ev['event_type'] == 'release':
                # Valid placement!
                action_type = "drag_drop"
                process_action(actions, active_deck_press, ev, image_files, action_type)
                active_deck_press = None # Consumed

            # Case B: Tap Select -> Tap Place (Press Deck -> Release Deck -> Press Arena)
            elif ev['event_type'] == 'press':
                # Valid placement!
                action_type = "tap_place"
                process_action(actions, active_deck_press, ev, image_files, action_type)
                active_deck_press = None # Consumed

    # Save provisional dataset
    # Save provisional dataset
    if actions:
        new_df = pd.DataFrame(actions)
        
        if os.path.exists(DATASET_FILE):
            existing_df = pd.read_csv(DATASET_FILE)
            # Filter out duplicates based on timestamp (arena click time should be unique enough)
            # Floating point comparison hazard? Use epsilon or just string match?
            # Let's simple check:
            existing_timestamps = set(existing_df['timestamp'].values)
            
            # Filter
            new_df = new_df[~new_df['timestamp'].isin(existing_timestamps)]
            
            if not new_df.empty:
                full_df = pd.concat([existing_df, new_df], ignore_index=True)
                full_df.to_csv(DATASET_FILE, index=False)
                print(f"Appended {len(new_df)} new actions to {DATASET_FILE} (Total: {len(full_df)})")
            else:
                print("No new unique actions found (duplicates skipped).")
        else:
            new_df.to_csv(DATASET_FILE, index=False)
            print(f"Saved {len(new_df)} actions to {DATASET_FILE}")
    else:
        print("No paired actions found. Check thresholds?")

def process_action(actions, deck_ev, arena_ev, image_files, action_type):
    # 1. Get image for the Deck Click
    img_path = find_closest_image(deck_ev['timestamp'], image_files)
    
    if img_path:
        # 2. Crop the card
        img = cv2.imread(img_path)
        if img is None: return
        h, w, _ = img.shape
        
        cx, cy = int(deck_ev['rel_x']), int(deck_ev['rel_y'])
        
        # Clamp crop
        cw, ch = CARD_CROP_SIZE
        x1 = max(0, cx - cw//2)
        y1 = max(0, cy - ch//2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        
        # Adjust if out of bounds (bottom right)
        if x2 - x1 < cw: x1 = max(0, x2 - cw)
        if y2 - y1 < ch: y1 = max(0, y2 - ch)

        crop = img[y1:y2, x1:x2]
        
        # Save crop
        crop_filename = f"card_{int(deck_ev['timestamp']*1000)}.jpg"
        crop_path = os.path.join(OUTPUT_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        actions.append({
            "timestamp": arena_ev['timestamp'],
            "card_image": crop_filename,
            "deck_x": deck_ev['rel_x'],
            "deck_y": deck_ev['rel_y'],
            "arena_x": arena_ev['rel_x'],
            "arena_y": arena_ev['rel_y'],
            "action_type": action_type
        })
        print(f"Action found ({action_type}): Select ({cx},{cy}) -> Place ({int(arena_ev['rel_x'])},{int(arena_ev['rel_y'])})")


if __name__ == "__main__":
    main()
