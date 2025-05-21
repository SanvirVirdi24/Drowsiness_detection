# Drowsiness_detection
I made a drowsiness detection system in python learning using Machine learning logic
# Drowsiness Detection System

A real-time drowsiness detection system using computer vision and facial landmark detection. This project uses a webcam feed to analyze eye aspect ratios and detect signs of drowsiness, triggering a voice alert when the user's eyes are closed for too long.

---

## 🛠️ Features

- Real-time video stream analysis
- Eye Aspect Ratio (EAR) based drowsiness detection
- Visual feedback with facial landmarks
- Text-to-speech alert using `pyttsx3`

---

## ⚙️ Installation

### Step 1: Install Required Libraries

Install the dependencies using `pip`:

```bash
pip install scipy
pip install opencv-python
pip install numpy
pip install pyglet
pip install pyttsx3

```
## Step 2: Install dlib
🐧 Linux / 🍎 macOS
Pre-requisites:

Python 3.x

CMake

Boost libraries

Installation:

```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python3 setup.py install --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA
```
🪟 Windows
Follow the official tutorial to install dlib on Windows:
🔗 Install Dlib on Windows – LearnOpenCV

---

## 📦 Additional Files
Step 3: Download Pre-trained Shape Predictor
Download the pre-trained dlib facial landmark predictor file:

📥 Download shape_predictor_68_face_landmarks.dat

Place the file in the same directory as your script and make sure to update the file path accordingly in your code:
```
model_path = r"C:/Users/yourusername/Downloads/shape_predictor_68_face_landmarks.dat"
```
⚠️ Without this .dat file, the script will not function.

## 🚀 Usage
Step 4: Run the Code
Make sure your webcam is connected and simply run:

```
python drowsiness_detection.py
```
🔁 Press ESC to stop the webcam and exit the program.

📄 How It Works
Captures video from your webcam.

Detects facial landmarks using dlib.

Calculates the Eye Aspect Ratio (EAR).

If the eyes remain closed (EAR < 0.25) for 20+ consecutive frames, it triggers a text-to-speech alarm to alert the user.

## 👁️ Eye Aspect Ratio (EAR) Formula


![image](https://github.com/user-attachments/assets/f86baf54-35cb-4b34-ad7d-a3bcf237a0c0)

​
 
Where p1 to p6 are the coordinates of specific eye landmarks.

## 📌 Notes
Make sure your environment supports camera access.

Tweak EAR_THRESHOLD or CONSEC_FRAMES values for different sensitivity levels.

Works best in well-lit environments with clear frontal face visibility.

## 🧠 Acknowledgements
dlib

OpenCV

pyttsx3

scipy


## 📝 License
This project is intended for academic and personal use. Please give credit if you use or modify it in your own projects.


---

Let me know if you want this as a downloadable `.md` file or need help uploading it directly to your GitHub repo
