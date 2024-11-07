---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

# 👋🏻 About Me

안녕하세요. 3년차 머신러닝 엔지니어 이영석입니다.  
의료 및 제조 도메인의 영상 데이터를 이용한 ML/DL 모델 연구 및 개발 경험이 있으며, 사용자 경험에 기반한 실질적인 문제 해결을 위해 고민하고 있습니다.
현재는 LLM 및 RAG 기술을 이용한 AI Assistant 솔루션을 개발하고 있으며, 모델 배포 및 동작 파이프라인 개선에 관심이 많아 백엔드 개발쪽으로 직무 영역을 확장하고 있습니다.

# 💼 Experience

- *2023.09. ~ 현재*, **ML Engineer, at <a href="https://laonpeople.com/main/main.php">LAON PEOPLE</a>**  
  - **AI Assistant 플랫폼 개발:**  
    - LLM 및 RAG 기술을 이용한 AI Assistant 플랫폼 개발
    - 사용자 대화 내용 정리 및 API 연동을 위한 OpenAI Structured Output 적용 및 JSON schema 생성 기능 개발
    - 사용자 문서 기반 semantic search 기능 개발을 위한 document parsing 및 vector database 구축 파이프라인 개발
    - Embedding model 및 LLM을 이용한 기본 RAG 구현 및 성능 개선을 위한 reranking, query transformation 기능 개발
    - LLM 답변 출처 표기 및 사용자 신뢰도 향상을 위한 self-citation 기능 개발
  
  - **머신비전 솔루션 개발 및 성능 개선:**  
    *(Active Learning, Out-of-Distribution Detection)*
    - Pseudo Labeling 및 Active Learning을 이용한 모델 재학습 파이프라인 성능 개선 연구
    - Out-of-distribution detection 방법 적용 및 성능 개선 실험 수행
    - Pytorch(Python) 실험 코드를 Libtorch(C++)로 변환 및 성능 재현 검증, 솔루션 탑재

- *2022.02. ~ 2023.03.*, **ML Researcher, at <a href="https://www.ingradient.ai/en/">INGRADIENT</a>**  
  - **반도체 불량판정 딥러닝 모델 연구 개발:**  
    *(Self-supervised Learning, Active Learning, Image Classification)*
    - 반도체 recipe 변경 시마다 새로운 이미지 및 불량 판정 기준에 빠르게 적응가능한 모델 재학습 파이프라인 연구 및 개발
    - Self-supervised Learning 방법 적용 및 모델 기본 성능 및 일반화 성능 개선
    - Entropy 기반 Uncertainty Estimation 및 Active Learning 파이프라인 적용, 신규 recipe 전환 시, 모델 재학습 시간 단축
  - **의료영상 반자동 라벨링 솔루션 성능 개선 연구:**  
    *(Medical Image Segmentation, Interactive Segmentation)*
    - 3D volume 단위의 의료영상 반자동 라벨링 기능 연구 및 개발
    - 3D Interactive Segmentation을 이용해 사용자 마우스 클릭 지점에 대한 3D 라벨링 기능 개발
    - Geodesic distance map을 이용한 interaction 지점의 context augmentation 및 Segmentation 성능 개선
    - PyQT를 이용한 데모 어플리케이션 개발 및 동작 파이프라인 성능 검증

- *2020.12. ~ 2021.12.*, **Learning Facilitator at <a href="https://modulabs.co.kr">MODULABS</a>**
  - **습성황반변성 환자의 치료 반응 예측 연구:**  
    *(GAN, Image-to-Image Translation)*
    - 김안과병원과 협업하여 습성 황반변성 환자의 치료 반응 예측을 위한 연구 수행
    - GAN 기반 Image-to-Image Translation 방법을 적용하여 치료 전 이미지로부터 치료 후 이미지 생성 실험 수행
    - Attention mechanism을 적용하여 주요 병변부위의 생성 quality 개선, 전문의 평가 및 성능 검증
    - 논문 작성 및 SCI 저널 공동 1저자 게재 (Scientific Reports, 2023)
  - **AI 교육 프로그램 강사 활동:**  
    *(Learning Facilitation, Education)*
    - AI 교육 프로그램 커리큘럼 설계 및 운영
    - DL/ML 기초 및 Python 프로그래밍 교육 진행
    - 컴퓨터 비전 분야 기본 논문 구현 강의 진행

# 📄 Publications

- **Prediction of Anti-Vascular Endothelial Growth Factor Agent-specific Treatment Outcomes in Neovascular Age-Related Macular Degeneration Using a Generative Adversarial Network** \[[paper](https://www.nature.com/articles/s41598-023-32398-7)\]  
  - **Scientific Reports** 2023  
  - Sehwan Moon\*, **Youngsuk Lee\***, Jeongyoung Hwang, Chul Gu Kim, Jong Woo Kim, Won Tae Yoon, and Jae Hui Kim (\*:equal contribution)

# 🎓 Education

- *2024.09. ~ 현재*, **고려대학교 의료정보학과 석사과정 재학 중**
- *2012.03. ~ 2020.08.*, **한국해양대학교 냉동공조에너지시스템공학 전공 학사 졸업**

# 📖 Other Activities

- *2021.12. ~ 2022.03.*, **ML Researcher, at <a href="https://modulabs.co.kr/product/hit-lab">HITLAB</a>**
  - **Human-in-the-Loop System 연구 스터디**  
    *(Interactive Segmentation, Human-in-the-loop)*
    - PyTorch를 이용한 interactive segmentation 논문 구현 <a href="https://ieeexplore.ieee.org/abstract/document/8370732">DeepIGeoS</a>[![](https://img.shields.io/github/stars/HITLAB-DeepIGeoS/DeepIGeoS?style=social&label=Stars)](https://github.com/HITLAB-DeepIGeoS/DeepIGeoS)
