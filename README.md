# 스마트 안경 시스템: YOLO 기반 객체 인식 및 Firebase 연동  

이 프로젝트는 **Raspberry Pi**와 **YOLOv5**를 활용하여 카메라로 실시간 객체 인식을 수행하고, 인식된 객체 이미지를 **Firebase Cloud Storage**로 업로드하여 사용자의 휴대폰에 알림을 전송하는 **스마트 안경 시스템**입니다.  

---

## SmartGlassesFinder 개발팀 소개
| 강기영 |
|:---:|
| <img src="https://github.com/kang0048.png" width="120" height="120"/> | 
| [@youngK](https://github.com/Kang0048)|
| 한성대학교 컴퓨터공학과 4학년 |

## 기술 스택

## Environment & Platform

<p align="left">
  <img src="https://img.shields.io/badge/Android-3DDC84?style=flat&logo=android&logoColor=white"/>
  <img src="https://img.shields.io/badge/Raspberry_Pi-A22846?style=flat&logo=raspberrypi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Node.js-339933?style=flat&logo=node.js&logoColor=white"/>
</p>

## Languages & Frameworks

<p align="left">
  <img src="https://img.shields.io/badge/Kotlin-7F52FF?style=flat&logo=kotlin&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black"/>
</p>

## AI Models & Processing

<p align="left">
  <img src="https://img.shields.io/badge/YOLO-FF6F00?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVQYV2NkwA7+//8/AwMDAwACxQG/X2vH2wAAAABJRU5ErkJggg=="/>
  <img src="https://img.shields.io/badge/ResNet18-FF6F00?style=flat"/>
</p>


##  주요 기능  

- **실시간 객체 인식**  
  YOLOv5 모델로 카메라에 포착된 객체를 실시간으로 식별 및 분류  

- **이미지 캡처 및 전송**  
  특정 조건(예: 객체 인식 시)에 마지막 인식 이미지를 자동 캡처  

- **Firebase 연동**  
  캡처된 이미지를 Firebase Cloud Storage에 업로드 및 관리  

- **모바일 알림**  
  업로드된 이미지를 Firebase 기반으로 사용자 휴대폰에 실시간 알림 전송  

---

## 기술 스택  

| 분류         | 기술                          | 설명                                       |
|------------- |-------------------------------|------------------------------------------- |
| **하드웨어** | Raspberry Pi                  | 메인 컨트롤러 역할                           |
|              | Logitech C270                 | 실시간 영상 스트리밍 및 이미지 캡처           |
|              | MIC Module                    | 실시간 음성 인식                            |
| **소프트웨어** | Python                         | 프로젝트 주요 프로그래밍 언어              |
|              | YOLOv5                        | 객체 인식 모델                             |
|              | OpenCV                        | 영상/이미지 처리                           |
|              | Firebase Admin SDK            | Firebase 연동 (Cloud Storage, 알림)        |

---

## 🚀 설치 및 실행 방법  

### 1. 필수 라이브러리 설치  

```bash
pip install firebase-admin
pip install opencv-python
pip install torch torchvision numpy
```
### 2. YOLOv5 모델 다운로드
1. YOLOv5 공식 GitHub에서 사전 학습 모델(yolov5s.pt 등)을 다운로드합니다.
2. 다운로드한 모델 파일을 프로젝트 디렉토리에 저장합니다.

### 3. Firebase 설정
1. Firebase 콘솔에서 새 프로젝트 생성
2. Cloud Storage 활성화
3. 서비스 계정 키 생성 → your-service-account.json 파일 다운로드
4. 해당 JSON 파일을 프로젝트 폴더에 추가

### 4. 실행
```bash
python main.py
```




### 관련 저장소
1. Raspberry Pi (YOLO + Firebase) → [(현재 저장소)](https://github.com/Kang0048/SmartGlassesFinderRaspberryPi)
2. Node.js 서버 → [링크](https://github.com/Kang0048/SmartGlassesFinderNode.js)
3. Android 앱 → [링크](https://github.com/Kang0048/SmartGlassesFinder)

