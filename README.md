
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

