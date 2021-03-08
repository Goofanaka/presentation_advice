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

# 웹 캠 경우 초기값 설정
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 캠 로드
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

# 캠 실행
while cap.isOpened():
    success, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # 얼굴 좌표값 받기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    # print(face_mesh)
    # print(results.multi_face_landmarks)
    # print(type(results.multi_face_landmarks))


    # 얼굴 좌표 그리기 준비
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    eye_image = image.copy()  # 눈 인식 표시만 할 이미지 복사
    idx_to_coordinates = {}  # 얼굴 랜드마크 좌표 dict 저장할 변수
    # 얼굴 랜드마크 dict 저장
    if results.multi_face_landmarks != None :
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and
                    landmark.visibility < VISIBILITY_THRESHOLD) or
                      (landmark.HasField('presence') and
                       landmark.presence < PRESENCE_THRESHOLD)):
                    continue
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px

    # image : 얼굴 인식 그리기 ( mediapipe draw_landmarks 메소드 사용 )
    # if results.multi_face_landmarks:
    #   for face_landmarks in results.multi_face_landmarks:
    #     mp_drawing.draw_landmarks(
    #                                image=image,
    #                                landmark_list=face_landmarks,
    #                                connections=mp_face_mesh.FACE_CONNECTIONS,
    #                                landmark_drawing_spec=drawing_spec,
    #                                connection_drawing_spec=drawing_spec)
        # print(face_landmarks)
    # eye_image : 눈 인식 그리기
    m = list(idx_to_coordinates.values())
    if results.multi_face_landmarks != None:
        for i in EYE_INDICES_TO_LANDMARKS:  # 눈 좌표값만 그리기
            # print(idx_to_coordinates.get(i))
            cv2.circle(eye_image, m[i], 3, (0, 0, 255), -1)

    # 랜드마크 dict에서 eye 랜드마크만 리스트 저장
    ex = [ x for i, x in idx_to_coordinates.items() if i in EYE_INDICES_TO_LANDMARKS ]
    ex = np.array(ex)  # np.array 변환
    # print('ex:{}'.format(ex))


    for face in results.multi_face_landmarks:

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=ex[0:16])  # 왼쪽 눈
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=ex[16:32])  # 오른쪽 눈

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        
        # 확인
        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        # model predict
        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        # 시각화, 0.1보다 높으면(눈 떴을때) 0, 감으면 1 표시
        state_l = 'O' if pred_l > 0.1 else '1'
        state_r = 'O' if pred_r > 0.1 else '1'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        # 눈 표시.
        cv2.rectangle(eye_image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(eye_image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)
        # 텍스트 넣기
        cv2.putText(eye_image, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(eye_image, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(eye_image, "BLINK: {}".format(cnt), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 확인
    # cv2.imshow('MediaPipe FaceMesh', image)  # 얼굴 인식 표시
    cv2.imshow('MediaPipe EyeMesh', eye_image)  # 눈 인식 표시

    if cv2.waitKey(1) == ord('q'):
        break

face_mesh.close()
cap.release()