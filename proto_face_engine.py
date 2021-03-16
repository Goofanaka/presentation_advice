import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import plotly.graph_objects as go
import os
from scipy.spatial import distance

class Eye_check:
    def __init__(self):
        self.VISIBILITY_THRESHOLD = 0.5
        self.PRESENCE_THRESHOLD = 0.5
        self.Landmark_eye = [
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 246, 161, 160, 159, 158, 157, 173,
            263, 249, 390, 373, 374, 380, 381, 382,
            362, 466, 388, 387, 386, 385, 384, 398
        ]
        self.model = load_model('models/eye_model.h5')
        self.IMG_SIZE = (34, 26)
        self.mp_drawing = mp.solutions.drawing_utils
        self.Landmark_mouth = [61, 37, 0, 267, 291, 405, 17, 181]
        self.m_dict = {}
        self.e_dict = {}

        for i in range(len(self.Landmark_mouth)):
            self.m_dict[i] = []

        for i in range(len(self.Landmark_eye)):
            self.e_dict[i] = []

    # 눈 자르기
    def crop_eye(self, img, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * self.IMG_SIZE[1] / self.IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect

    # 눈 좌표값 np.array 변환
    def to_ndarray(self, dict):
        return np.array([x for i, x in dict.items() if i in self.Landmark_eye])

    # 눈 깜박임 모델 적용, 예측
    def eye_pre(self, eye_img_l, eye_img_r):
        # 이미지 변경
        eye_img_l = cv2.resize(eye_img_l, dsize=self.IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=self.IMG_SIZE)
        # eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        # 전처리
        eye_input_l = eye_img_l.reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.

        # model 적용 -> 예측
        pred_l = self.model.predict(eye_input_l)
        pred_r = self.model.predict(eye_input_r)

        # 시각화, 0.02보다 높으면(눈 떴을때) 0, 감으면 1 표시
        state_l = '0' if pred_l > 0.02 else '1'
        state_r = '0' if pred_r > 0.02 else '1'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        return int(state_l), int(state_r)

    # 눈 좌표값 추출 및 마킹
    def eye_drawing(self, landmark_dict):
        for i in self.Landmark_eye:  # 눈 좌표값만 그리기
            cv2.circle(eye_image, landmark_dict[i], 1, (0, 0, 255), -1)

    # 얼굴 랜드마크 dict 저장
    def landmark_dict(self, results):
        face_landmark = {}
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and
                     landmark.visibility < self.VISIBILITY_THRESHOLD) or
                        (landmark.HasField('presence') and
                         landmark.presence < self.PRESENCE_THRESHOLD)):
                    continue
                landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                if landmark_px:
                    face_landmark[idx] = landmark_px
        return face_landmark

    #눈 깜박임 횟수 시각화
    def eye_blink_Visualization(self, eye_list):
        # 시각화 파일 저장 폴더 설정,  import os
        if not os.path.exists("images"):
            os.mkdir("images")

        eye_len = list(range(len(eye_list) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eye_len[1:],
            y=eye_list,
            name="분당 깜박임 횟수 데이터",
            line = dict(color='Black', width=2)
        )
        )

        fig.add_hrect(y0=10, y1=20, line_width=0, fillcolor="red", opacity=0.1)

        fig.add_annotation(
            x=max(range(len(eye_list)), key=eye_list.__getitem__) + 1,
            y=max(eye_list),
            xref="x",
            yref="y",
            text="가장 눈을 많이 감은 횟수",
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#000000"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bordercolor="#FFFFFF",
            borderwidth=2,
            borderpad=4,
            bgcolor="#FFFFFF",
            opacity=0.8
        )

        fig.update_layout(template="plotly_white",
                          showlegend=True,
                          title="눈깜박임 횟수를 바탕으로한 긴장도 측정",
                          xaxis_title="시간(분)",
                          yaxis_title="깜박임 횟수",
                          legend_title="눈 깜박임 횟수",
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="Black"
                          )
                          )

        # PNG, JPEG, and WebP 가능
        fig.write_image("images/fig1.png")
        #fig.show()

    # 얼굴 좌표 변화 거리 계산
    def face_euclidean(self, dict, part=1):
        # 1이면 입
        if part == 1 :
            for i, k in zip(self.Landmark_mouth, self.m_dict.keys()) :
                res = int(distance.euclidean(dict[5], dict[i]))
                self.m_dict[k].append(res)
        # 2면 눈
        elif part == 2 :
            for i, k in zip(self.Landmark_eye, self.e_dict.keys()) :
                res = int(distance.euclidean(dict[5], dict[i]))
                self.e_dict[k].append(res)

        return self.m_dict if part == 1 else self.e_dict

if __name__ == '__main__':
    eye_cnt = 0
    frame = 0
    eye = 0
    eye_list = []

    # cap = cv2.VideoCapture('video2.mp4')
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m_fps = cap.get(cv2.CAP_PROP_FPS) * 60  # 분당 프레임
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    check = Eye_check()

    # 캠 실행
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            print("End frame")
            eye_list.append(int(eye_cnt / 2))
            break

        frame += 1
        if frame % m_fps == 0 :
            eye_list.append(int(eye_cnt / 2))
            eye_cnt = 0
            frame = 0

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
            idx_to_coordinates = check.landmark_dict(results)

            # 눈 부분 마킹
            # check.eye_drawing(idx_to_coordinates)

            # 눈 좌표 np.array 변환
            eye_np = check.to_ndarray(idx_to_coordinates)

            # 눈 부분 crop, 주의! 오른쪽부터 나올 경우 안됨, 입장시 왼쪽으로 입장해주세요
            eye_img_l, eye_rect_l = check.crop_eye(gray, eye_points=eye_np[0:16])  # 왼쪽 눈
            eye_img_r, eye_rect_r = check.crop_eye(gray, eye_points=eye_np[16:32])  # 오른쪽 눈

            # 눈 깜빡임 예측 값 반환
            state_l, state_r = check.eye_pre(eye_img_l, eye_img_r)
            state_l, state_r = check.eye_pre(eye_img_l, eye_img_r)
            state = 1 if state_l == 1 or state_r == 1 > 0.05 else 0

            # 눈 깜빡임 카운트
            if eye != state:
                eye = state
                eye_cnt += 1
            else :
                eye = state

            # 입 좌표 변화율
            m_dict = check.face_euclidean(idx_to_coordinates, part=1)
            print(m_dict)
            # 눈 좌표 변화율
            e_dict = check.face_euclidean(idx_to_coordinates, part=2)
            print(e_dict)

            # 눈 표시
            cv2.rectangle(eye_image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
            cv2.rectangle(eye_image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

            # 텍스트 넣기
            cv2.putText(eye_image, str(state_l), tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(eye_image, str(state_r), tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(eye_image, "BLINK: {}".format(int(eye_cnt/2)), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # writer.write(image)
        cv2.imshow('MediaPipe EyeMesh', eye_image)  # 눈 인식 표시

        if cv2.waitKey(1) == ord('q') :
            eye_list.append(int(eye_cnt / 2))
            break

    check.eye_blink_Visualization(eye_list)
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
