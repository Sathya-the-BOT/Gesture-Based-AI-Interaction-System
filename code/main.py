from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

# Add instructions section
st.title("Gesture-Based AI Interaction System")
st.markdown("""
### Instructions:
- ✌️ **Two Fingers (Index & Middle)**: Draw on the canvas.
- ☝️ **Index Finger Only**: Erase the specific area where the index finger moves.
- 🖐️ **Five Fingers**: Clear the entire canvas.
- 🤙 **Pinky Gesture**: Submit the drawing for AI analysis (Ensure the canvas is not empty).
            
            -- Designed By Team EyeQ.
""")

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyA3U-cN9rTTKKy-aWXizgz0NFgrWNqaMpE")
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def gethandinfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]

        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas, eraser_mode=False):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 1, 0, 0]:  # Draw mode (Index and middle finger up)
        current_pos = tuple(map(int, lmList[8][0:2]))
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)

    elif eraser_mode:  # Erase mode (Only index finger up)
        current_pos = tuple(map(int, lmList[8][0:2]))
        cv2.circle(canvas, current_pos, 20, (0, 0, 0), -1)  # Erase a circular area

    elif fingers == [1, 1, 1, 1, 1]:  # Clear canvas
        canvas = np.zeros_like(canvas)

    return current_pos, canvas

def sendtoai(model, canvas, fingers):
    # Check for the pinky gesture and ensure the canvas is not empty
    if fingers == [0, 0, 0, 0, 1] and np.any(canvas != 0):  # Pinky up and canvas not empty
        pil_image = Image.fromarray(canvas)
        response = model.generate_content([
            "Analyze the text in the image provided and give the corresponding output. "
            "If it is a problem, solve it, or identify the language.", pil_image
        ])
        return response.text
    elif fingers == [0, 0, 0, 0, 1]:
        return "Canvas is empty. Please draw something before submitting."
    return None

prev_pos = None
canvas = None
image_combined = None
output_text = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = gethandinfo(img)
    if info:
        fingers, lmList = info
        eraser_mode = fingers == [0, 1, 0, 0, 0]  # Activate eraser mode when only the index finger is up
        prev_pos, canvas = draw(info, prev_pos, canvas, eraser_mode)
        output_text = sendtoai(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
