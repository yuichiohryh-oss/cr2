# AGENTS.md - Context for AI Assistants

This file provides context for AI agents working on this project.

## Current Status (2026-01-07)
- **Phase**: Transitioning from Phase 2 (Model 1 Complete) to Phase 3 (Model 2 Data Collection).
- **Model 1**: Successfully trained `ActionRecognitionModel` (ResNet18 backbone) to 99% accuracy on a small dataset (116 samples).
- **Environment**: CUDA 12.8 (Nightly) on RTX 5060 Ti set up and verified.

## Key Files & Logic

### Data Collection
- `src/data_collection/recorder.py`:
  - Uses `mss` for screenshots and `pynput` for global hooks.
  - Syncs mouse clicks with frame timestamps.
  - **Note**: The window selection logic relies on `pygetwindow` and user input.

### Processing
- `src/processing/labeler.py`:
  - **Logic**: Matches "Deck Area Click (Press)" -> "Arena Area Click (Press/Release)" to identify card plays.
  - Crops card images 80x100px.
  - Appends new actions to `dataset/raw_actions.csv` (does NOT overwrite).
  - Handles timestamp matching between Click Log (log.csv) and Images (filenames).

- `src/processing/manual_labeler.py`:
  - Tkinter GUI for efficient labeling.
  - Merges `raw_actions.csv` (new data) into `labeled_actions.csv` (master dataset).
  - Auto-skips finalized labels.

### Training
- `src/training/train_model1.py`:
  - Pytorch based training script.
  - **Heads**: Multi-task learning.
    1. Classification (Unit ID 0-9)
    2. Regression (Coordinates X,Y normalized 0-1)
  - **Important**: Requires Nightly PyTorch for CUDA 12.8 support on RTX 50 series.

## Next Steps (Phase 3)
1. **Video Analysis Pipeline**:
   - Need to download high-level gameplay videos (YouTube).
   - Use `Model 1` (`model1.pth`) to infer actions from video frames.
   - Challenge: `Model 1` is trained on Scrcpy screenshots (clean). YouTube videos may have overlays, compression artifacts, or different aspect ratios.
   - **Task**: Create a script to process video files, detect card plays using Model 1, and build the "Strategic Dataset" for Model 2.

2. **Model 2 Development**:
   - Input: Game State (Image/History).
   - Output: Suggested Action (Card + Position).
   - Architecture: Needs to handle temporal information (RNN/Transformer?).

## Known Issues / Notes
- The clustering script (`cluster_cards.py`) was used for initial bootstrapping but superseded by `manual_labeler.py`. Can be deprecated or kept for analytics.
- `requirements.txt` contains `torch` logic but user environment is specific (Nightly). Be careful when reinstalling.
