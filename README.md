# presentation_advices
ssac 인공지능 과정 2차 프로젝트 
## 필요 설치 파일
1. 라이브러리 : numpy, opencv-python
2. pre-trained model
https://github.com/CMU-Perceptual-Computing-Lab/openpose에 들어가서 zip 파일을 다운 받은 후, openpose-master/models/models 파일에 있는 get models.bat 파일을 관리자권한으로 실행시켜줘야한다.(sudo로 cmd로 실행하는 것을 추천한다.)

## 필요 환경 설정
1. OpenCV GPU 환경 설정 <br>
OpenCV는 기본적으로 CPU를 사용하는데 연산처리속도의 문제가 있기 때문에 GPU를 사용해 연산처리속도를 향상 시켜야 한다. <br>
**설정 시 필요한 파일**
    - Visual Studio
    - OpenCV
    - OpenCV-Contrib
    - CMake
    - nvidia CUDA
    - nvidia cuDNN
