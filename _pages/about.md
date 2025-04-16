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

3년차 머신러닝 엔지니어로서, 제조 및 의료 분야에서 Computer Vision과 LLM 기반의 RAG/Agent 시스템을 개발하고 성능을 고도화한 경험이 있습니다.

고객 및 사용자 피드백을 기반으로 문제를 명확히 정의하고, 빠르게 baseline 성능을 확보한 뒤 최신 기술을 접목해 점진적으로 성능을 고도화하며 실질적인 성과를 만들어왔습니다.

비즈니스 가치를 실현하는 엔지니어로서의 역량뿐만 아니라, 문제를 깊이 있게 고민하고 해결하는 연구자로서의 전문성 또한 함께 추구하고 있습니다.

# 💼 Work Experience

## Machine Learning Engineer, <a href="https://laonpeople.com/main/main.php">LAON PEOPLE</a>
  ( *2023.09. ~ 현재* )

  > **LLM Agent 도입을 통한 AI Assistant 플랫폼 성능 개선 (2025년 1월 ~ 현재)**  
    *Agent 기반 Multi-step Reasoning을 통해, 기존 RAG 대비 정확도 약 25% 향상*
- 단일 query 기반 기존 RAG의 한계점 개선을 위해, LLM Agent 기반 방법론 도입 및 성능 고도화 업무 수행
- LangGraph를 이용한 ReAct Agent 구현 및 reasoning 기반 multi-step 검색 기능 개발
- 약 2700 페이지 분량의 고객사 문서를 이용한 실험 결과, 기존 RAG 대비 정확도 약 25% 향상
- 현재 Slack, Email 등 플랫폼 확장을 위한 MCP Tool 연동 및 Multi-agent 관련 연구 진행 중
- 사용 기술: Python, LangGraph, MCP(Model Context Protocol)

  > **RAG 기반 AI Assistant 플랫폼 개발 (2024년 3월 ~ 2024년 12월, 총 10개월)**  
    *고객사 피드백 기반 RAG 성능 개선을 통해, 고객사 계약 체결*
- 팀 초기 멤버로 합류하여, 전체 RAG 파이프라인 구축 및 LLM, RAG 관련 최신 기술 연구 및 개발 업무 수행
- RAG 성능 정량화를 위해 사용자 채팅 기록 모니터링 환경 구축 및 LLM-as-a-judge 기반 evaluation 파이프라인 개발
- 고객사 미팅을 주도하여 도메인 지식에 기반한 동의어 사전을 구축하고, Query Expansion, Hybrid Retrieval, Reranking 기법을 적용하여 retrieval 성능을 개선
- 기존 retrieval 문서 출처 표기 방식을 prompting 및 structured output을 활용해 LLM generation pipeline에 통합, hallucination 감소 및 사용자 답변 신뢰도 개선
- 사용 기술: Python, LangChain, Milvus, MLFlow

  > **AI 비전 검사 솔루션 성능 고도화 (2023년 9월 ~ 2023년 12월, 총 4개월)**  
    *기존 Active Learning 파이프라인의 한계점을 개선하여, Accuracy 및 F1-score 기준 약 5~10% 성능 향상*
- 비전 검사 솔루션에 탑재된 Active Learning 알고리즘의 성능 개선 업무 수행
- 기존 파이프라인의 성능 검증 및 OOD(Out-of-Distribution) 샘플에 대한 한계점 파악
- OOD Detection 알고리즘 구현 및 실험 설계, 솔루션 탑재를 위한 수치연산 최적화 및 성능 검증
- 고객사 데이터 실험 결과, 기존 대비 Accuracy 및 F1-score 약 5~10% 향상된 성능을 확인
- 사용 기술: Python, PyTorch, C++, LibTorch

## **Machine Learning Researcher, <a href="https://www.ingradient.ai/">INGRADIENT</a>**
  ( *2022.02. ~ 2023.03., 1년 2개월* )

  > **반도체 불량 검사 AI 모델 학습 파이프라인 개발 (2022년 9월 ~ 2023년 2월, 총 6개월)**  
    *Self-supervised Learning 및 Active Learning을 적용하여, 빠른 재학습이 가능한 모델 학습 파이프라인을 구축*
- 반도체 recipe 변경 시, 새로운 데이터에 빠르게 적응할 수 있는 모델 재학습 파이프라인 연구 및 개발 업무 수행
- Self-supervised Learning 기반 pre-training 적용을 통해, 모델 일반화 성능 약 20% 향상  
- Active Learning을 이용한 모델 재학습 파이프라인을 구축하여, 모델 학습 시간 단축 및 효율 개선
- 사용 기술: Python, PyTorch

  > **의료영상 반자동 라벨링 솔루션 성능 개선 (2022년 2월 ~ 2022년 9월, 총 8개월)**  
    *2D 단위의 라벨링 작업을 3D volume 단위로 개선하여, 라벨링 효율 및 작업 시간 단축*
- 기존 솔루션의 2D 기반 라벨링 성능 개선을 위해, 3D 단위 반자동 라벨링 기능 연구 개발 업무 수행
- 3D Interactive Segmentation 및 Geodesic Distance Map을 적용하여, 사용자 마우스 입력 기반 3D volume 라벨링 기능 구현
- PyQT 기반 데모 애플리케이션 구현 및 동작 테스트 결과, 3D MR 및 CT 영상의 라벨링 작업 시간 감소
- 사용 기술: Python, PyTorch, PyQT

## **Learning Facilitator, <a href="https://modulabs.co.kr/">모두의연구소</a>**
  ( *2020.12. ~ 2021.12., 1년 1개월* )
  > **AI 교육 프로그램 커리큘럼 설계 및 강사 활동 (2020년 12월 ~ 2021년 12월, 총 1년 1개월)**  
    *Flipped Learning 방식을 활용한 토론 중심 수업을 통해, 학습자 참여도 및 만족도 향상에 기여*
- AI 기초 이론 및 프로그래밍 수업 진행
- 논문 리뷰 및 구현 실습, 과제 코드 리뷰 및 프로젝트 멘토링 업무 수행

# 📖 Other Activities

> **안과질환 연구 프로젝트 참여 (2021년 8월 ~ 2022년 12월, 총 1년 5개월)**  
  *GAN을 이용한 치료 예후 예측 모델 연구를 수행하여, SCI 저널 공동 1저자 논문 게재*
- 김안과병원과 협업하여, 습성 황반변성 환자의 치료 반응 예측을 위한 연구 수행
- GAN 기반 Image-to-Image Translation 방법을 적용하여 치료 전 이미지로부터 치료 후 이미지 생성 모델 학습
- Attention mechanism을 적용하여 주요 병변부위의 생성 quality 개선, 안과 전문의의 정성적 평가를 통해 치료 예후 예측 모델 성능 개선
- 논문 작성 및 SCI 저널 공동 1저자 게재 (Scientific Reports, 2023)

# 📄 Publications

- **Prediction of Anti-Vascular Endothelial Growth Factor Agent-specific Treatment Outcomes in Neovascular Age-Related Macular Degeneration Using a Generative Adversarial Network** [[paper](https://www.nature.com/articles/s41598-023-32398-7)]  
  - **Scientific Reports** 2023  
  - Sehwan Moon\*, **Youngsuk Lee\***, Jeongyoung Hwang, Chul Gu Kim, Jong Woo Kim, Won Tae Yoon, and Jae Hui Kim (\*:equal contribution)

# 🎓 Education

- *2024.09. ~ 현재*, 고려대학교 의료정보학과 석사과정 재학 중
- *2012.03. ~ 2020.08.*, 한국해양대학교 냉동공조에너지시스템공학 전공 학사 졸업
