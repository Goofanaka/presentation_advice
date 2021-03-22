# 1. Presentation advice Project
* 프로젝트 명 : 발표의 참견 
* 개요 : 인공지능을 활용한 사용자 프레젠테이션의 역량 강화 시스템, 사용자의 발표영상 중 비언어적 표현인 자세분석모델, 눈 깜박임 분석모델, 표정변화율 분석모델을 개발
* 팀명 : Goofanaka 팀원 : 손기훈 김동건 유주아 김은찬
* 개발 기간 : 2021년 03월 02일 ~ 2021년 3월 16일
___

# 2. Technologies
- 개발언어 : Python
- 모델 구현 : OpenPose, OpenCV, Mediapipe, Tensorflow2.0
- 분석 및 시각화 : Numpy, Pandas, matplotlib, Plotpy
- Git, Github

___

# 3. Features
- pose_check.py : 자세분석 엔진 및 시각화 
- proto_face_engine.py : 눈 깜박임 및 표정변화율 분석엔진 및 시각화
- models(eye_model.h5) : 눈 깜박임 감지 모델

___

# 4. Execution Environment
- OpenCV GPU 환경 설정
- pre-trained model
   https://github.com/CMU-Perceptual-Computing-Lab/openpose openpose-master/models/models 파일에 있는 get models.bat 설치
- Mediapipe 설치

___

# 5. WorkFlow
![workflow](https://user-images.githubusercontent.com/71329051/111962202-5d504000-8b35-11eb-9c8d-48cce40997af.png)


