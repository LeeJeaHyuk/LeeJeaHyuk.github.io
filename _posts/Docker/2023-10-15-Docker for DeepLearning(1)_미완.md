---
layout: single
title: "WSL과 Docker를 사용한 DeepLearning 훈련 환경 구축"
categories: [Docker]
tags:
  - Docker
  - WSL
  - DeepLearning
toc : true
---

## 학습목표

WSL과 Docker를 사용하여 리눅스 환경에서 gpu를 사용할 수 있는 딥러닝 훈련 환경 만들기

선행
- Ubuntu 배포판 설치
- WSL 설치
- Docker Desktop설치
- Hyper-v 사용 가능한 환경(Windows pro, edu)



## vscode의 WSL 플러그인 설치

![image-20231015114331173](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015114331173.png)
## Connect to WSL

![image-20231015114517469](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015114517469.png)좌측 하단의 >< 버튼을 누르면 connet to WSL 이 검색창에 나오고 클릭해준다.

그러면 Windows 환경에서 VS Code를 사용하면서 WSL의 리눅스 환경에서 코드를 실행하고 디버깅할 수 있게 된다.

오류가 발생하면 wsl -l 을 확인하여 기본값이 Ubuntu로 되어있는지 확인하고 다시 시도한다.

![image-20231015114906895](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015114906895.png)


## 기반 이미지 pull 해오기

먼저 dockerfile에 기반이 되는 이미지를 docker hub에서 pull 하려고 한다. 나는 jupyter를 사용할 것이기 때문에 
먼저 기본 python을 pull 해왔다.

```
docker pull python
```

그러면 Docker Hub에서 Python 공식 이미지를 다운로드하게 된다.

그리고 폴더를 하나 만들고 dockerfile을 만들어주었다.
```bash
mkdir project
code Dockerfile
```


### dockerfile
만들어진 dockerfile을 채워준다.

```dockerfile
From python:latest

WORKDIR /deepl

RUN pip install jupyter

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root" ]
```


From python:latest를 사용하여 방금 불러온 이미지를 기반으로 작성한다.

이후 jupyter에서 저장될 디렉토리를 `WORKDIR`에 지정해준다. 위에서는 deepl이라는 폴더를 생성하여 하위에 파일들을 저장하게 된다.


### requirement 파일
`requirement.txt` 파일을 기반으로 모듈 등을 설치한다.

```txt
matplotlib
numpy
pandas
scikit-learn
```
`requirement.txt` 파일은 테스트를 위해 간단히 위와 같이 설정해주었다.
버전을 지정하지 않으면 최신 버전으로 설치하게 된다.

`ENTRYPOINT`
- `jupyter`: Jupyter Notebook 및 Jupyter Lab과 같은 Jupyter 프로젝트의 명령을 실행하는 프로그램입니다.
- `lab`: Jupyter Lab을 실행하는 서브 명령입니다. Jupyter Lab은 Jupyter Notebook과 비슷하지만 더 강력한 웹 기반 대화형 개발 환경을 제공합니다.
- `--ip=0.0.0.0`: Jupyter Lab을 0.0.0.0 IP 주소에서 수신하도록 구성합니다. 이렇게 하면 컨테이너 내부 및 외부에서 Jupyter Lab에 액세스할 수 있게 됩니다.
- `--no-browser`: Jupyter Lab이 기본 웹 브라우저를 자동으로 열지 않도록 설정합니다. 이 옵션을 사용하면 컨테이너에서 Jupyter Lab을 실행할 때 브라우저를 열지 않습니다.
- `--allow-root`: Jupyter Lab을 루트 사용자 권한으로 실행하도록 허용합니다. 이것은 컨테이너 내에서 Jupyter Lab을 실행할 때 루트 권한을 사용하도록 설정합니다.

### docker-compose 파일
```docker-compose.yml
version: '1.0'
services:
  jupyter-lab:
    build: .
    ports:
      - 8888:8888
    volumes:
      - ./deepl:/deepl
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

- `verson` : 버전을 의미
- `services  jupyter-lab`  : 이름을 "jupyter-lab"으로 설정, 이미지의 이름이 jupyter lab으로 만들어짐
- `build .` : 현재 디렉터리 (`.`)에서 Docker 이미지를 빌드하도록 지정
- `port` : 호스트 시스템의 포트 8888을 컨테이너의 포트 8888에 매핑하여 Jupyter Lab을 호스트 시스템에서 접근 가능하게 한다
- `volumes: - ./deepl:/deepl` :  현재 디렉터리의 "deepl" 디렉터리를 컨테이너 내부의 "/deepl" 디렉터리에 마운트한다. 이를 통해 컨테이너와 호스트 간에 파일을 공유할 수 있다.
- - `driver: nvidia`: 이 설정은 NVIDIA GPU 드라이버를 사용하는 디바이스를 예약하도록 지정합니다. 즉, 이 서비스가 GPU를 사용할 수 있도록 합니다.
- `count: 1`: GPU 디바이스를 1개만 예약하도록 지정합니다. 따라서 이 서비스는 GPU 리소스를 1개만 사용할 수 있습니다.
- `capabilities: [gpu]`: 이 설정은 예약된 GPU 디바이스가 GPU 기능을 가지고 있음을 나타냅니다. 이를 통해 서비스가 해당 GPU를 사용할 수 있도록 합니다.

## 실행

```bash
docker-compose build
docker-compose up
```

![image-20231015120843167](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015120843167.png)

jupyter를 사용할 수 있게 된다.

![image-20231015120957193](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015120957193.png)

테스트 파일을 하나 생성하고 다시 vscode를 확인해보면

![image-20231015121027064](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015121027064.png)

동일한 파일이 vscode의 지정한 폴더에 들어있는 것을 확인할 수 있다.


![image-20231015121103746](../../images/2023-10-15-Docker for DeepLearning(1)_미완/image-20231015121103746.png)
