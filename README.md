
# PulseNet-HR

PulseNet-HR is a real-time heart rate and stress monitoring system that uses computer vision and signal processing techniques to estimate physiological parameters from a webcam feed.

The system supports two modes of operation:

* **Face Mode** – Extracts pulse-related information from facial regions using remote photoplethysmography (rPPG).
* **Finger Mode** – Uses finger-based photoplethysmography (PPG) as a fallback when facial signal quality is poor.

The extracted signals are processed using normalization, Butterworth filtering, and Fast Fourier Transform (FFT) to estimate heart rate in beats per minute (BPM). The system also performs Heart Rate Variability (HRV) analysis, signal quality assessment, motion detection, and basic stress classification.

## Features

* Real-time heart rate estimation
* Face-based rPPG detection
* Automatic fallback to finger-based PPG
* FFT-based BPM calculation
* Heart Rate Variability (HRV) analysis
* Signal quality monitoring
* Motion detection
* Stress level indication
* Live BPM graph visualization
* CSV data logging

## Tech Stack

* Python
* OpenCV
* NumPy
* SciPy
* Matplotlib

## Project Structure

```text
multimodel2.py
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
heart_data.csv
```

## Installation

```bash
pip install opencv-python numpy scipy matplotlib
```

## Run

```bash
python multimodel2.py
```

Press `Q` to exit the application.

## Output

The application displays:

* Heart Rate (BPM)
* Signal Quality
* Heart Rate Variability (HRV)
* Stress Status
* Motion Alerts
* Live BPM vs Time Graph

A CSV file is generated to store collected readings for further analysis.

## Results

### Finger Mode Detection
<img width="506" height="410" alt="image" src="https://github.com/user-attachments/assets/3c77f97b-53c9-4387-a23e-ebd6771dc957" />


### Face-Based Detection
<img width="471" height="378" alt="image" src="https://github.com/user-attachments/assets/781fa896-f66b-489a-b8bc-9943a7be3138" />


### BPM vs Time Graph
<img width="506" height="385" alt="image" src="https://github.com/user-attachments/assets/f8dcc091-9e2b-46dc-8110-08f62cf2f9ca" />


