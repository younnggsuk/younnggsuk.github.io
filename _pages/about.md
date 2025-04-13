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

I'm a Machine Learning Engineer, specializing in developing AI models applicable to real-world services in manufacturing and healthcare domains.  
I have hands-on experience applying various AI technologies, including Computer Vision, LLM-based RAG, and Agent systems.  
Currently, I'm pursuing a Master's degree at Korea University, integrating practical problem-solving skills with academic depth to achieve continuous growth in both industrial and research fields.

# 💼 Work Experience

## Machine Learning Engineer, [LAON PEOPLE](https://laonpeople.com/main/main.php)
(*Sep 2023 ~ Present*)

> **Improvement of AI Assistant Platform Performance using LLM Agents (Jan 2025 ~ Present)**  
  *Achieved approximately a 25% accuracy improvement over the existing RAG by applying agent-based multi-step reasoning.*
- Enhanced the existing RAG-based platform by adopting an LLM agent-based methodology.
- Developed a LangGraph-based ReAct Agent and implemented reasoning-based multi-step search capabilities.
- Conducted a PoC for classifying product code from manual of about 2,700 pages, achieving approximately 25% higher accuracy compared to the existing RAG approach.
- Currently integrating MCP tools for expanding platforms such as Slack and Email, and conducting research related to multi-agent systems.

> **Development of RAG-based AI Assistant Platform (Mar 2024 ~ Dec 2024, 10 months)**  
  *Improved RAG performance based on client feedback, resulting in client contracts and platform launch.*
- Joined as an initial team member to build a RAG pipeline and conducted research and development on the latest LLM and RAG technologies.
- Developed an evaluation pipeline based on an LLM-as-a-judge method by monitoring user chat logs to quantify RAG performance.
- Led client meetings to build a domain-specific synonym dictionary, applied Query Expansion, Hybrid Retrieval, and Reranking techniques to enhance search performance.
- Integrated document-source citations into the LLM generation pipeline using prompting and structured output, reducing hallucinations and improving response reliability.

> **Active Learning Performance Improvement for Vision Inspection Solution (Sep 2023 ~ Dec 2023, 4 months)**  
  *Analyzed and overcame limitations of the existing active learning pipeline, achieving a 5~10% improvement in Accuracy and F1-score.*
- Enhanced the Active Learning algorithm performance embedded in vision inspection solutions.
- Verified the performance of the existing pipeline and identified limitations regarding Out-of-Distribution (OOD) samples.
- Implemented OOD detection algorithms, designed experiments, integrated improvements into the existing pipeline, and verified performance.
- Customer data experiments demonstrated approximately a 5~10% improvement in Accuracy and F1-score.

## Machine Learning Researcher, [INGRADIENT](https://www.ingradient.ai/)
(*Feb 2022 ~ Mar 2023, 1 year 2 months*)

> **Development of AI Model Training Pipeline for Semiconductor Defect Inspection (Sep 2022 ~ Feb 2023, 6 months)**  
  *Constructed a retraining pipeline capable of rapid model adaptation using Self-supervised Learning and Active Learning.*
- Developed a retraining pipeline to quickly adapt to new semiconductor data when recipe changes occurred.
- Applied Self-supervised Learning-based pre-training, improving generalization performance by approximately 20%.
- Built an Active Learning-based retraining pipeline, reducing model training time and improving efficiency.

> **Performance Improvement of Medical Image Semi-Automatic Labeling Solution (Feb 2022 ~ Sep 2022, 8 months)**  
  *Improved labeling efficiency and reduced working time by upgrading labeling tasks from 2D to 3D volumes.*
- Developed 3D semi-automatic labeling functionality to enhance existing 2D labeling solution performance.
- Implemented a 3D volume labeling function using 3D Interactive Segmentation and Geodesic Distance Map based on user mouse input.
- Developed a PyQT-based demo application and conducted operational tests, reducing labeling times for 3D MR and CT images.

# 📖 Other Activities

> **Research Project Participation in Ophthalmic Diseases (Aug 2021 ~ Dec 2022, 1 year 5 months)**  
  *Conducted research on treatment outcome prediction models using GAN, resulting in a co-first author publication in an SCI journal.*
- Collaborated with Kim's Eye Hospital to research treatment response predictions in patients with wet Age-related Macular Degeneration.
- Trained an Image-to-Image translation GAN model to generate post-treatment images from pre-treatment images.
- Improved generation quality of major lesion regions by applying an Attention mechanism, resulting in better qualitative evaluations by ophthalmologists.
- Authored a paper published as co-first author in an SCI journal (Scientific Reports, 2023).

# 📄 Publications

- **Prediction of Anti-Vascular Endothelial Growth Factor Agent-specific Treatment Outcomes in Neovascular Age-Related Macular Degeneration Using a Generative Adversarial Network** [[paper](https://www.nature.com/articles/s41598-023-32398-7)]  
  - **Scientific Reports** 2023  
  - Sehwan Moon*, **Youngsuk Lee***, Jeongyoung Hwang, Chul Gu Kim, Jong Woo Kim, Won Tae Yoon, and Jae Hui Kim (*equal contribution)

# 🎓 Education

- *Sep 2024 ~ Present*, **Master’s Degree in Biomedical Informatics, Korea University**
- *Mar 2012 ~ Aug 2020*, **Bachelor’s Degree in Refrigeration and Air Conditioning Engineering, Korea Maritime and Ocean University**