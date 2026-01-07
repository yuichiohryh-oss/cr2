import os
import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import json

CARD_DIR = "dataset/card_images"
ACTION_FILE = "dataset/raw_actions.csv"
OUTPUT_FILE = "dataset/labeled_actions.csv"
CLUSTER_VIS_FILE = "clusters.png"

def main():
    if not os.path.exists(ACTION_FILE):
        print("No actions found.")
        return

    df = pd.read_csv(ACTION_FILE)
    
    # Load images
    images = []
    
    # We only process images that are in the CSV
    valid_files = sorted(list(set(df['card_image'].values))) # SORTED for determinism
    filenames = []
    
    print(f"Loading {len(valid_files)} images...")
    
    for f in valid_files:
        path = os.path.join(CARD_DIR, f)
        if os.path.exists(path):
            img = cv2.imread(path)
            # Use HSV Color Histogram for shift invariance
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate Histogram: H and S are most important for cards. V changes with selection highlight?
            # Selection highlight (white border / glow) might change V. 
            # Use H, S.
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            images.append(hist.flatten())
            filenames.append(f)
            
    X = np.array(images)
    # X is already normalized 0-1 per image.
    
    print("Clustering...")
    # Histogram distance? 
    # DBSCAN uses Euclidean by default. Euclidean on    print("Clustering...")
    # DBSCAN was merging too aggressively.
    # Since we know a deck has 8 cards, let's force 8 clusters with KMeans.
    # We use n_init=50 to find a good stable centroid set.
    # User reported missing classes with K=8. Maybe visual variants? Increase to 12 to over-segment.
    clustering = KMeans(n_clusters=12, n_init=50, random_state=42).fit(X)
    labels = clustering.labels_
        
    print(f"Found {len(set(labels))} clusters: {set(labels)}")
    
    # 1. Create initial file->cluster map
    file_to_cluster = dict(zip(filenames, labels))
    
    # Map back to dataframe
    # file_to_label = dict(zip(filenames, labels))
    # We want to propagate overrides to the CLUSTER, not just the file.
    
    # 3. Apply User-Defined Mapping
    # User provided mapping for K=12 clusters:
    # 0,5,6 -> Skeleton
    # 1,8 -> Musketeer
    # 2,9 -> Ice Spirit
    # 3 -> Evo Skeleton
    # 4 -> Hog Rider
    # 7 -> Ice Golem
    # 10 -> Cannon
    # 11 -> Fireball
    
    # We map them to standard IDs 0-7
    # 0: Skeleton
    # 1: Evo Skeleton
    # 2: Ice Spirit
    # 3: Ice Golem
    # 4: Hog Rider
    # 5: Musketeer
    # 6: Fireball
    # 7: Cannon

    CLUSTER_TO_UNIT_ID = {
        0: 0, 5: 0, 6: 0, # Skeleton
        3: 1,             # Evo Skeleton
        2: 2, 9: 2,       # Ice Spirit
        7: 3,             # Ice Golem
        4: 4,             # Hog Rider
        1: 5, 8: 5,       # Musketeer
        11: 6,            # Fireball
        10: 7             # Cannon
    }
    
    UNIT_NAMES = {
        0: "Skeleton",
        1: "Evo Skeleton",
        2: "Ice Spirit",
        3: "Ice Golem",
        4: "Hog Rider",
        5: "Musketeer",
        6: "Fireball",
        7: "Cannon"
    }
    
    # Create file -> final_unit_id map
    final_file_map = {}
    for f, cluster_id in file_to_cluster.items():
        if cluster_id in CLUSTER_TO_UNIT_ID:
            final_file_map[f] = CLUSTER_TO_UNIT_ID[cluster_id]
        else:
            print(f"Warning: Cluster {cluster_id} for file {f} is not mapped! Mapping to -1.")
            final_file_map[f] = -1
            
    df['unit_id'] = df['card_image'].map(final_file_map)
    df['unit_name'] = df['unit_id'].map(UNIT_NAMES)
    
    # Handle missing labels
    df['unit_id'] = df['unit_id'].fillna(-1).astype(int)
    
    # Save final
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved labeled actions to {OUTPUT_FILE}")
    
    # Visualize Final Groups
    unique_ids = sorted([uid for uid in CLUSTER_TO_UNIT_ID.values()])
    n_clusters = len(unique_ids)

    cols = 4
    rows = (n_clusters + cols - 1) // cols
    
    plt.figure(figsize=(10, 3 * rows))
    
    for uid in unique_ids:
        # Find representative
        sample = df[df['unit_id'] == uid]
        if sample.empty: continue
        
        rep_file = sample.iloc[0]['card_image']
        rep_img_path = os.path.join(CARD_DIR, rep_file)
            
        if os.path.exists(rep_img_path):
            img = cv2.imread(rep_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(rows, cols, uid + 1)
            plt.imshow(img)
            name = UNIT_NAMES.get(uid, "Unknown")
            plt.title(f"ID {uid}: {name}\n(cnt={len(sample)})")
            plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(CLUSTER_VIS_FILE)
    print(f"Saved cluster visualization to {CLUSTER_VIS_FILE}")

if __name__ == "__main__":
    main()
