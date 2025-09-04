# Smart Glasses Project 👓  

YOLO 기반 객체 인식, 음성 명령, Firebase 연동을 활용한 **스마트 안경 프로젝트**입니다.  
라즈베리파이에서 카메라와 마이크를 통해 물건을 인식하고, Node.js 서버와 Firebase를 거쳐 안드로이드 앱에서 결과를 확인할 수 있습니다.  

---

## 📌 아키텍처  

```mermaid
flowchart LR
    A[Raspberry Pi] --> B[Node.js Server]
    B --> C[Firebase]
    C --> D[Android App]

