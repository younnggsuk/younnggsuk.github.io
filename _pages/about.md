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

I’m Youngsuk Lee, an ML Engineer passionate about data-driven solutions in the various domain. My current focus is on deploying and optimizing ML/DL models for real-world applications to enhance user experience and solve practical challenges.

In addition to my work, I’m pursuing a Master’s degree at Korea University College of Medicine, where my research centers on text-guided vision models aimed at overcoming the limitations of current vision models.

# 💼 Experience

- *2023.09. ~ Present*, **ML Engineer, at <a href="https://laonpeople.com/main/main.php">LAON PEOPLE</a>**  
  - **Developing a question-answering AI Assistant Solution:**  
    *(Large Language Models (LLM), Retrieval-Augmented Generation (RAG))*
    - Developing an AI Assistant Solution that can answer user questions based on various documents.
    - Utilizing Retrieval-Augmented Generation (RAG) and document parsing methods to enhance the model’s ability to generate responses based on document content.
    - Design and implement a chat / rag pipeline to handle various situations, such as multi-turn conversations and citation-based responses.
  
  - **Developed machine vision solutions for manufacturing industry images:**  
    *(Active Learning, Out-of-Distribution Detection)*
    - Developed a machine vision solution for the manufacturing industry to detect defects in various products.
    - Conducting research on active learning and apply out-of-distribution detection methods to improve the model's performance with limited data.

- *2022.02. ~ 2023.03.*, **ML Researcher, at <a href="https://www.ingradient.ai/en/">INGRADIENT</a>**  
  - **Developed visual inspection system for semiconductor images:**  
    *(Self-supervised Learning, Active Learning, Image Classification)*
    - Developed an automatic visual inspection system for defect discrimination in a scenario with continually changing data and defect criteria.
    - Applied self-supervised learning to enhance generalizability and active learning for rapid adaptation to new data.
  - **Developed semi-automatic labeling system for medical images:**  
    *(Medical Image Segmentation, Interactive Segmentation)*
    - Applied interactive segmentation algorithms to label 3D lesions with minimal user interactions.
    - Utilized geodesic distance transform to effectively convey user interactions to the model, improving performance.

- *2020.12. ~ 2021.12.*, **Learning Facilitator at <a href="https://modulabs.co.kr">MODULABS</a>**
  - **Researched on predicting treatment outcomes in neovascular Age-related Macular Degeneration:**  
    *(GAN, Image-to-Image Translation)*
    - Conducted research projects aimed at predicting patient responses to treatment for neovascular age-related macular degeneration (AMD) in collaboration with Kim's Eye Hospital.
    - Applied GAN-based Image-to-Image Translation methods with an attention mechanism to generate post-treatment images focusing on lesion areas.
  - **Performed educational facilitation in AI education programs:**  
    *(Learning Facilitation, Education)*
    - Acted as a facilitator to help students understand AI, DL/ML basics and Python programming.
    - Conducted a lecture on implementing computer vision research papers to enhance students' proficiency indeep learning frameworks.

# 📄 Publications

- **Prediction of Anti-Vascular Endothelial Growth Factor Agent-specific Treatment Outcomes in Neovascular Age-Related Macular Degeneration Using a Generative Adversarial Network** \[[paper](https://www.nature.com/articles/s41598-023-32398-7)\]  
  - **Scientific Reports** 2023  
  - Sehwan Moon\*, **Youngsuk Lee\***, Jeongyoung Hwang, Chul Gu Kim, Jong Woo Kim, Won Tae Yoon, and Jae Hui Kim  
  (\*:equal contribution)

# 🎓 Education

- *2024.09. ~ Present*, **Korea University College of Medicine**
  - Master’s Student in Biomedical Informatics
- *2012.03. ~ 2020.08.*, **Korea Maritime and Ocean University**
  - Bachelor’s Degree in Refrigeration and Air Conditioning Engineering

# 📖 Other Activities

- *2021.12. ~ 2022.03.*, **ML Researcher, at <a href="https://modulabs.co.kr/product/hit-lab">HITLAB</a>**
  - **Study on human-in-the-loop systems with a focus on interactive segmentation.**  
    *(Interactive Segmentation, Human-in-the-loop)*
    - Implemented the interactive segmentation paper <a href="https://ieeexplore.ieee.org/abstract/document/8370732">DeepIGeoS</a> using PyTorch. [![](https://img.shields.io/github/stars/HITLAB-DeepIGeoS/DeepIGeoS?style=social&label=Stars)](https://github.com/HITLAB-DeepIGeoS/DeepIGeoS)
