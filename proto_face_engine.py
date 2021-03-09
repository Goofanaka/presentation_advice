import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# 초기값 설정
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
EYE_INDICES_TO_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
model = load_model('models/eye_model.h5')
IMG_SIZE = (34, 26)
cnt = 0
n = 0
li_cnt = []

# 웹 캠 경우 초기값 설정
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 캠 로드
cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FRAME_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(frame_count)


# 눈 자르기
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


# 눈 깜빡임 모델 적용
def eye_pre(eye_img_l, eye_img_r):
    # 이미지 변경
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    # eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # 전처리
    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    # model 적용 -> 예측
    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    # 시각화, 0.05보다 높으면(눈 떴을때) 0, 감으면 1 표시
    state_l = 'O' if pred_l > 0.05 else '1'
    state_r = 'O' if pred_r > 0.05 else '1'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    return state_l, state_r


# 눈 좌표값 ndarray 변환
def to_ndarray(dict):
    eye_np = [x for i, x in dict.items() if i in EYE_INDICES_TO_LANDMARKS]
    eye_np = np.array(eye_np)  # np.array 변환
    return eye_np


# 눈 좌표값 추출 및 마킹
def eye_drawing(landmark_dict):
    for i in EYE_INDICES_TO_LANDMARKS:  # 눈 좌표값만 그리기
        # print(landmark_dict.get(i))
        cv2.circle(eye_image, landmark_dict.get(i), 3, (0, 0, 255), -1)


# 얼굴 랜드마크 dict 저장
def landmark_dict(results):
    face_landmark = {}

    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
            if landmark_px:
                face_landmark[idx] = landmark_px
    return face_landmark


# 캠 실행
while cap.isOpened():
    ret, image = cap.read()

    if not ret:
      print("Ignoring empty camera frame.")
      break

    # 얼굴 좌표값 받기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # 얼굴 좌표 그리기 준비
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 눈 인식 표시만 할 이미지 복사
    eye_image = image.copy()

    if results.multi_face_landmarks :
        # 얼굴 랜드마크 dict 저장
        idx_to_coordinates = landmark_dict(results)
        # 눈 부분 마킹
        eye_drawing(idx_to_coordinates)
        # 눈 좌표 np.array 변환
        eye_np = to_ndarray(idx_to_coordinates)
        # 눈 부분 crop
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=eye_np[0:16])  # 왼쪽 눈
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=eye_np[16:32])  # 오른쪽 눈

        # 눈 깜빡임 예측 값 반환
        state_l, state_r = eye_pre(eye_img_l, eye_img_r)

        # 눈 깜빡임 카운트


        # 눈 표시
        cv2.rectangle(eye_image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(eye_image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        # 텍스트 넣기
        cv2.putText(eye_image, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(eye_image, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(eye_image, "BLINK: {}".format(cnt), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 확인
        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)
    cv2.imshow('MediaPipe EyeMesh', eye_image)  # 눈 인식 표시

    if cv2.waitKey(1) == ord('q'):
        break

face_mesh.close()
cap.release()