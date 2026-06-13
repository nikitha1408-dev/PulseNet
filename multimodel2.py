import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from collections import deque
import time
import csv
import os

# -------- LOAD MODEL --------
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

cap = cv2.VideoCapture(0)

# -------- BIG WINDOW WITH ❌ --------
cv2.namedWindow("Heart Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Heart Monitor", 1280, 720)

# -------- DATA --------
heart_rate_values = deque(maxlen=100)
bpm_values = deque(maxlen=20)

start_time = time.time()
prev_x, prev_y = 0, 0
motion_flag = False

# -------- GRAPH --------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("BPM vs Time")
time_data, bpm_data = [], []

# -------- CSV --------
if not os.path.exists("heart_data.csv"):
    with open("heart_data.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time","BPM","HRV","Mode","Quality"])

# -------- FILTER --------
def butter_lowpass_filter(data, cutoff, fs):
    b, a = butter(3, cutoff/(0.5*fs))
    return filtfilt(b, a, data)

# -------- FFT --------
def calculate_bpm_fft(signal, fs):
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = freqs[:len(freqs)//2]
    fft_vals = fft_vals[:len(fft_vals)//2]
    mask = (freqs >= 0.8) & (freqs <= 3)
    if np.any(mask):
        return freqs[mask][np.argmax(fft_vals[mask])] * 60
    return 0

# -------- SIGNAL QUALITY --------
def signal_quality(signal):
    if len(signal) < 20:
        return 0
    return np.var(signal)

# -------- FEATURES --------
def detect_stress(bpm):
    if bpm == 0: return "Detecting"
    elif bpm < 75: return "Normal"
    elif bpm < 90: return "Mild Stress"
    else: return "High Stress"

def calculate_hrv(bpm_values):
    return np.std(bpm_values) if len(bpm_values) > 5 else 0

def heart_alert(bpm):
    if bpm == 0: return "Detecting"
    elif bpm < 50: return "Low HR"
    elif bpm <= 100: return "Normal"
    else: return "High HR"

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed = time.time() - start_time
    fs = cap.get(cv2.CAP_PROP_FPS) or 30

    roi = None
    mode = "FACE"
    quality_flag = "Poor"

    # -------- FINGER DETECTION --------
    red = frame[:,:,2]
    green = frame[:,:,1]

    red_mean = np.mean(red)
    green_mean = np.mean(green)
    red_std = np.std(red)

    if red_mean > 130 and red_std < 40 and red_mean > green_mean:
        mode = "FINGER"
        roi = frame

    # -------- FACE MODE --------
    else:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,117,123))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]])
                x,y,x2,y2 = box.astype(int)
                w,h = x2-x, y2-y

                # -------- STRICT DISTANCE --------
                face_area = w*h
                if face_area < 20000:
                    h_frame, w_frame, _ = frame.shape
                    cv2.putText(frame, "MOVE CLOSER!",
                                (int(w_frame/6), int(h_frame/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0,0,255), 4)

                    cv2.imshow("Heart Monitor", frame)
                    continue

                roi = frame[y:y+int(h*0.3), x:x+w]

                if prev_x != 0:
                    motion_flag = abs(prev_x-x)+abs(prev_y-y) > 20
                prev_x, prev_y = x, y

                cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
                break

    # -------- SIGNAL --------
    if roi is not None and roi.size > 0:
        green_signal = roi[:,:,1]
        avg = np.mean(green_signal)
        heart_rate_values.append(avg)

    # -------- BPM --------
    if len(heart_rate_values) > 25:
        signal = np.array(heart_rate_values)
        signal = (signal-np.mean(signal))/np.std(signal+1e-7)
        signal = butter_lowpass_filter(signal,2.0,fs)

        quality = signal_quality(signal)

        if quality > 0.5:
            quality_flag = "Good"
            bpm = calculate_bpm_fft(signal,fs)

            if bpm > 0:
                bpm_values.append(bpm)

            avg_bpm = np.mean(bpm_values) if bpm_values else 0
        else:
            avg_bpm = 0
    else:
        avg_bpm = 0

    # -------- FEATURES --------
    status = detect_stress(avg_bpm)
    hrv = calculate_hrv(bpm_values)
    alert = heart_alert(avg_bpm)

    # -------- CSV --------
    with open("heart_data.csv","a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round(elapsed,2), round(avg_bpm,2), round(hrv,2), mode, quality_flag])

    # -------- GRAPH --------
    time_data.append(elapsed)
    bpm_data.append(avg_bpm)

    line.set_xdata(time_data)
    line.set_ydata(bpm_data)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    # -------- DISPLAY --------
    cv2.putText(frame,f"Mode: {mode}",(10,20),0,0.6,(255,255,255),2)
    cv2.putText(frame,f"BPM: {avg_bpm:.1f}",(10,50),0,1,(0,0,255),2)
    cv2.putText(frame,f"Quality: {quality_flag}",(10,80),0,0.7,(0,255,0),2)
    cv2.putText(frame,f"Status: {status}",(10,110),0,0.8,(255,255,0),2)
    cv2.putText(frame,f"HRV: {hrv:.2f}",(10,150),0,0.8,(0,255,255),2)
    cv2.putText(frame,f"Alert: {alert}",(10,190),0,0.8,(0,255,0),2)

    if motion_flag and mode=="FACE":
        cv2.putText(frame,"Too Much Motion",(10,230),0,0.8,(0,0,255),2)

    if avg_bpm == 0:
        cv2.putText(frame,"INVALID SIGNAL",(10,270),0,1,(0,0,255),2)

    cv2.imshow("Heart Monitor",frame)

    # -------- EXIT --------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    try:
        if cv2.getWindowProperty("Heart Monitor", cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

# -------- CLEANUP --------
cap.release()
cv2.destroyAllWindows()
plt.close('all')