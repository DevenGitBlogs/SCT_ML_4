import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class GestureRecognizer:
    def __init__(self):
        # Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Face
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.7)

        # Gesture detection states
        self.prev_wrist_x = None
        self.direction = 0
        self.wave_count = 0
        self.last_wave_time = 0
        self.show_hello_until = 0

        self.ok_stable_frames = 0
        self.OK_FRAMES_REQUIRED = 5
        self.show_ok_until = 0

        self.finger_history = deque(maxlen=3)

    # Count fingers
    def fingers_up(self, landmarks, handedness):
        tips = [4, 8, 12, 16, 20]
        pip = [2, 6, 10, 14, 18]
        count = 0
        if handedness == 'Right':
            if landmarks.landmark[tips[0]].x < landmarks.landmark[pip[0]].x:
                count += 1
        else:
            if landmarks.landmark[tips[0]].x > landmarks.landmark[pip[0]].x:
                count += 1
        for tip_idx, pip_idx in zip(tips[1:], pip[1:]):
            if landmarks.landmark[tip_idx].y < landmarks.landmark[pip_idx].y:
                count += 1
        return count

    # HELLO detection
    def detect_hello(self, landmarks, fingers):
        current_time = time.time()
        if fingers < 5:
            self.prev_wrist_x = None
            self.wave_count = 0
            self.direction = 0
            return current_time < self.show_hello_until

        wrist_x = landmarks.landmark[0].x
        if self.prev_wrist_x is not None:
            dx = wrist_x - self.prev_wrist_x
            current_dir = 1 if dx > 0.02 else -1 if dx < -0.02 else 0
            if current_dir != 0 and current_dir != self.direction:
                self.wave_count += 1
                self.direction = current_dir
                self.last_wave_time = current_time
            if current_time - self.last_wave_time > 1.5:
                self.wave_count = 0
                self.direction = 0
            if self.wave_count >= 3:
                self.show_hello_until = current_time + 3
                self.wave_count = 0
                return True
        self.prev_wrist_x = wrist_x
        return current_time < self.show_hello_until

    # OK detection
    def detect_ok(self, landmarks, fingers):
        current_time = time.time()
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        wrist = landmarks.landmark[0]
        index_mcp = landmarks.landmark[5]

        dist_thumb_index = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
        )
        hand_size = np.linalg.norm(
            np.array([wrist.x, wrist.y]) - np.array([index_mcp.x, index_mcp.y])
        )
        normalized_dist = dist_thumb_index / hand_size

        other_fingers_folded = (
            middle_tip.y > landmarks.landmark[10].y and
            ring_tip.y > landmarks.landmark[14].y and
            pinky_tip.y > landmarks.landmark[18].y
        )

        if normalized_dist < 0.25 and fingers >= 3 and other_fingers_folded:
            self.ok_stable_frames += 1
        else:
            self.ok_stable_frames = 0

        if self.ok_stable_frames >= self.OK_FRAMES_REQUIRED:
            self.show_ok_until = current_time + 3
            self.ok_stable_frames = 0
            return True

        return current_time < self.show_ok_until

    def get_gesture_text(self, landmarks, handedness):
        fingers = self.fingers_up(landmarks, handedness)
        self.finger_history.append(fingers)
        smoothed_fingers = round(sum(self.finger_history) / len(self.finger_history))

        if self.detect_ok(landmarks, smoothed_fingers):
            return 'Deven says OK :)', (0, 255, 255)
        elif self.detect_hello(landmarks, smoothed_fingers):
            return 'Deven saying "HELLO" :)', (0, 200, 255)
        else:
            if smoothed_fingers == 0:
                return "Closed", (0, 0, 255)
            elif smoothed_fingers == 5:
                return "Open", (0, 255, 0)
            else:
                return str(smoothed_fingers), (255, 0, 0)

def main():
    cap = cv2.VideoCapture(0)
    recognizer = GestureRecognizer()

    # Try sunglasses overlay
    sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
    use_sunglasses = sunglasses is not None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = recognizer.hands.process(rgb)
        result_face = recognizer.face_detection.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        gesture_text = "Deven, show your hands"
        text_color = (0, 255, 255)
        rect_color = (30, 30, 30)

        # --- Hands ---
        if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
            landmarks = result_hands.multi_hand_landmarks[0]
            handedness = result_hands.multi_handedness[0].classification[0].label
            recognizer.mp_draw.draw_landmarks(image, landmarks, recognizer.mp_hands.HAND_CONNECTIONS)
            gesture_text, text_color = recognizer.get_gesture_text(landmarks, handedness)

        # --- Face ---
        if result_face.detections:
            for detection in result_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Clip bbox inside frame
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)

                face_roi = image[y:y+h, x:x+w]
                if face_roi.size > 0:
                    smooth = cv2.bilateralFilter(face_roi, 5, 50, 50)
                    image[y:y+h, x:x+w] = smooth

                if use_sunglasses:
                    sg_h, sg_w = int(h * 0.4), w
                    sg_y = y + int(h * 0.25)
                    sg_x = x
                    if sg_y+sg_h < ih and sg_x+sg_w < iw:
                        sg_resized = cv2.resize(sunglasses, (sg_w, sg_h))
                        alpha_s = sg_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(3):
                            image[sg_y:sg_y+sg_h, sg_x:sg_x+sg_w, c] = (
                                alpha_s * sg_resized[:, :, c] +
                                alpha_l * image[sg_y:sg_y+sg_h, sg_x:sg_x+sg_w, c]
                            )

        # --- UI Text ---
        (w, h), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        overlay = image.copy()
        cv2.rectangle(overlay, (40, 30), (40 + w + 20, 30 + h + 20), rect_color, -1)
        cv2.addWeighted(overlay, 0.9, image, 0.1, 0, image)

        cv2.putText(image, gesture_text, (50, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 6)
        cv2.putText(image, gesture_text, (50, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        cv2.imshow("Hand + Face Filter", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    