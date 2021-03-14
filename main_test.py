import pose_check
import proto_face_engine
import cv2
import mediapipe as mp
import threading
import Queue
# from pathos.multiprocessing import ProcessingPool as Pool


if __name__ == '__main__':

    #포즈 객체선언
    pose = pose_check.Pose_Check(
        'E:\\project_doc\\2nd_project\\project_code\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_deploy_linevec.prototxt',
        'E:\\project_doc\\2nd_project\\project_code\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_iter_440000.caffemodel'
    )
    #아이 객체선언
    check = proto_face_engine.eye_check()


    #pose를 체크하기 위한 변수 코드
    n = 0
    idx = 0
    video = cv2.VideoCapture('E:\\project_doc\\2nd_project\\project_code\\Handsome - 36702.mp4')
    name = 'frame'
    frame_path = 'E:\\project_doc\\2nd_project\\project_code\\Frames'
    converted_path = 'E:\\project_doc\\2nd_project\\project_code\\new_img'
    FPS = int(video.get(cv2.CAP_PROP_FPS))

    #eye를 체크하기 위한 코드
    eye_cnt = 0
    eye = 0
    eye_list = []

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m_fps = int(video.get(cv2.CAP_PROP_FPS)) * 60  # 분당 프레임
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)  # 총 프레임
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)



    pose.set_dir(frame_path)

    while (video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            print("Ignoring empty camera frame.")
            eye_list.append(int(eye_cnt / 2))
            break


        n += 1
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        # 얼굴 좌표 그리기 준비
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        eye_image = image.copy()

        if results.multi_face_landmarks:
            # 얼굴 랜드마크 dict 저장
            idx_to_coordinates = check.landmark_dict(results, width, height)
            # print(idx_to_coordinates)

            # 눈 부분 마킹
            # check.eye_drawing(idx_to_coordinates)



            # 눈 좌표 np.array 변환
            eye_np = check.to_ndarray(idx_to_coordinates)
            # 눈 부분 crop, 주의! 오른쪽부터 나올 경우 안됨, 입장시 왼쪽으로 입장해주세요

            # eye_img_l, eye_rect_l, eye_img_r, eye_rect_r = p.map(check.crop_eye, [eye_np[0:16], eye_np[16:32]])
            t1 = threading.Thread(target=check.crop_eye, args=eye_np[0:16])

            # eye_img_l, eye_rect_l = check.crop_eye(gray, eye_points=eye_np[0:16])  # 왼쪽 눈
            # eye_img_r, eye_rect_r = check.crop_eye(gray, eye_points=eye_np[16:32])  # 오른쪽 눈

            # 눈 깜빡임 예측 값 반환

            # state_l, state_r = p.map(check.eye_pre, [eye_img_l, eye_img_r])
            # p.close()

            # state_l, state_r = check.eye_pre(eye_img_l, eye_img_r)

            # state = 1 if state_l == 1 or state_r == 1 > 0.05 else 0
            #
            # # 눈 깜빡임 카운트
            # if eye != state:
            #     eye = state
            #     eye_cnt += 1
            # else :
            #     eye = state
            #
            # # 눈 표시
            # cv2.rectangle(eye_image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
            # cv2.rectangle(eye_image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

            # 텍스트 넣기
            # cv2.putText(eye_image, str(state_l), tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(eye_image, str(state_r), tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(eye_image, "BLINK: {}".format(int(eye_cnt/2)), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # 확인
            # cv2.imshow('l', eye_img_l)
            # cv2.imshow('r', eye_img_r)

        cv2.imshow('MediaPipe EyeMesh', eye_image)  # 눈 인식 표시


        #1초의 마지막 프레임에서 프레임 저장
        if n == FPS:
            cv2.imwrite(frame_path + '/{}{}.jpg'.format(name, idx), frame)
            # eye_list.append(int(eye_cnt/2))

            #판별 코드 초기화
            n = 0
            #감은 눈 세는 코드 초기화
            eye_cnt = 0
        if cv2.waitKey(1) & 0xff == 27:
            # eye_list.append(int(eye_cnt / 2))
            break

    video.release()
    cv2.destroyAllWindows()

    pose.set_dir(converted_path)

    frame_list = pose.get_frame_list(frame_path, name)

    sh_count = []
    eye_count = []
    pel_count = []
    n = 0

    print(frame_list)
    for i in frame_list:
        points, frame = pose.estimation(i)
        cv2.imwrite(i.replace(frame_path, converted_path),frame)

        shoulder = pose.isHorizontal(points[2], points[5])
        eye = pose.isHorizontal(points[14], points[15])
        sh_count.append(shoulder)
        eye_count.append(eye)
        if pose.isVertical((points[8], points[9]), (points[11], points[12])) == 1 or pose.isHorizontal(points[8], points[11]) == 1:
            pelvis = 1
            pel_count.append(pelvis)
        elif pose.isVertical((points[8], points[9]), (points[11], points[12])) == None or pose.isHorizontal(points[8], points[11]) == None:
            pelvis = None
            pel_count.append(pelvis)
        else:
            pelvis = 0
            pel_count.append(pelvis)

        print(i, eye)




