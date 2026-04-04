# GestureVoice — ISL Recognition System

## Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Make sure these files exist:
   model/isl_model.pkl
   model/hand_landmarker.task

3. Run the app:
   python run.py

## Controls
- Start  → begin recognition
- Stop   → pause recognition
- SPACE  → finish current word, add to sentence
- Clear  → reset everything

## Requirements
- Python 3.9+
- Webcam
- Windows 10/11