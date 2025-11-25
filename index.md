---
layout: project
title: "Automatically Detecting Online Deceptive Patterns"
description: "AutoBot accurately identifies and localizes deceptive patterns from website screenshots without relying on HTML code, achieving an F1-score of 0.93. In this paper we introduce a two-stage framework combining vision models and LLMs for detecting dark patterns on websites."
keywords: "deceptive patterns, dark patterns, AutoBot, machine learning, web security, UI detection, privacy, computer vision, NLP, ACM CCS 2025, browser extension, web measurement, YOLO, knowledge distillation, automated detection, user manipulation, cookie banners, online privacy, University of Wisconsin Madison"
og_image: /assets/images/system_overview.jpg
date: 2025-10-14
last_modified_at: 2025-10-22
authors:
  - name: '<a href="https://asmitnayak.com" target="_blank">Asmit Nayak</a>'
    email: "anayak6@wisc.edu"
  - name: '<a href="https://wiscprivacy.com/member/member_yash/" target="_blank">Yash Wani</a>*'
    email: "ywani@wisc.edu"
  - name: '<a href="https://www.annienobear.com/" target="_blank">Shirley Zhang</a>*'
    email: "hzhang664@wisc.edu"
  - name: 'Rishabh Khandelwal'
    email: "rkhandelwal3@wisc.edu"
  - name: '<a href="https://kassemfawaz.com" target="_blank">Kassem Fawaz</a>'
    email: "kfawaz@wisc.edu"
affiliation: "University of Wisconsin - Madison"
venue: "ACM CCS 2025"
award: "🏆 Distinguished Paper Award"
equal_contribution: "*Indicates Equal Contribution"
paper_url: "https://arxiv.org/pdf/2411.07441"
code_url: ""
#arxiv_url: "https://arxiv.org/abs/2411.07441"
dataset_urls:
  - name: "D3 Dataset"
    url: "https://huggingface.co/datasets/WIPI/deceptive_patterns_synthetic"
  - name: "Dataset 2"
    url: ""
demo_url: "https://huggingface.co/spaces/WIPI/DeceptivePatternDetector"
slides_url: "/assets/slides/ccs_2025.pptx"
permalink: /
bibtex: |
  @inproceedings{10.1145/3719027.3765191,
  author = {Nayak, Asmit and Wani, Yash and Zhang, Shirley and Khandelwal, Rishabh and Fawaz, Kassem},
  title = {Automatically Detecting Online Deceptive Patterns},
  year = {2025},
  isbn = {9798400715259},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3719027.3765191},
  doi = {10.1145/3719027.3765191},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  pages = {96–110},
  numpages = {15},
  keywords = {automated detection of deceptive patterns, computer vision, deceptive patterns, knowledge distillation, multi-modal large language models, synthetic data generation, ui element detection},
  location = {Taipei, Taiwan},
  series = {CCS '25}
}
---

## Abstract

<div class="abstract">
<p>Deceptive patterns in digital interfaces manipulate users into making unintended decisions, exploiting cognitive biases and psychological vulnerabilities. These patterns have become ubiquitous on various digital platforms. While efforts to mitigate deceptive patterns have emerged from legal and technical perspectives, a significant gap remains in creating usable and scalable solutions. We introduce our <strong>AutoBot framework</strong> to address this gap and help web stakeholders navigate and mitigate online deceptive patterns. <em><strong>AutoBot</strong></em> accurately identifies and localizes deceptive patterns from a screenshot of a website without relying on the underlying HTML code. <em><strong>AutoBot</strong></em> employs a two-stage pipeline that leverages the capabilities of specialized vision models to analyze website screenshots, identify interactive elements, and extract textual features. Next, using a large language model, <em><strong>AutoBot</strong></em> understands the context surrounding these elements to determine the presence of deceptive patterns. We also use <em><strong>AutoBot</strong></em>, to create a synthetic dataset to distill knowledge from ‘teacher’ LLMs to smaller language models. Through extensive evaluation, we demonstrate <em><strong>AutoBot</strong></em>’s effectiveness in detecting deceptive patterns on the web, achieving an F1-score of 0.93 in this task, underscoring its potential as an essential tool for mitigating online deceptive patterns.</p>
<p>We implement <em><strong>AutoBot</strong></em>, across three downstream applications targeting different web stakeholders: (1) a local browser extension providing users with real-time feedback, (2) a Lighthouse audit to inform developers of potential deceptive patterns on their sites, and (3) as a measurement tool for researchers and regulators.</p>
</div>


## How Off-the-Shelf LLMs Fall Short

While state-of-the-art vision-language models show promise in understanding visual content, directly applying them to detect deceptive patterns reveals significant limitations. These models often struggle with hallucination and lack of localization needed to identify deceptive patterns in real-world web interfaces.

<div class="project-figure">
  <div style="display: flex; gap: 20px; justify-content: space-evenly; align-items: flex-start; flex-wrap: wrap;">
    <div style="width: 35%; min-width: 300px; display: flex; flex-direction: column; align-items: center;">
      <img src="{{ '/assets/images/gemini-failure.jpg' | relative_url }}" alt="Example of Gemini AI incorrectly detecting deceptive patterns in cookie consent banner, showing hallucination issues in large language models" style="width: 100%;">
      <figcaption style="margin-top: 10px; font-size: 0.9em; text-align: center;">Gemini hallucinates that the "Accept all cookies" button being more visually prominent than the "Necessary cookies only" one.</figcaption>
    </div>
    <div style="width: 35%; min-width: 300px; display: flex; flex-direction: column; align-items: center;">
      <img src="{{ '/assets/images/gpt-failure.jpg' | relative_url }}" alt="Example of GPT-4V incorrectly analyzing visual differences in cookie banner buttons, demonstrating limitations of vision-language models" style="width: 100%;">
      <figcaption style="margin-top: 10px; font-size: 0.9em; text-align: center;">GPT-4.5 hallucinates that the "Accept all cookies" and "Reject all" button are visually different.</figcaption>
    </div>
  </div>
  <figcaption>Out-of-the-box LLMs demonstrate high false positive rates and poor localization accuracy when detecting deceptive patterns.</figcaption>
</div>

These limitations motivated us to develop a specialized framework that combines vision and language models in a structured framework, rather than relying on end-to-end models that lack the precision required for this task.


## The AutoBot Framework

<div class="project-figure">
  <img src="{{ '/assets/images/system_overview.jpg' | relative_url }}" alt="AutoBot framework architecture showing two-stage pipeline: Vision Module for UI element localization and Language Module for deceptive pattern detection" style="width: 100%; height: auto;">
  <figcaption>AutoBot's two-stage pipeline: (1) Vision Module to localize UI elements and extract features, (2) Language Module to detect deceptive patterns.</figcaption>
</div>

AutoBot adopts a modular design, breaking down the task into two distinct modules: a Vision Module for element localization and feature extraction, and a Language Module for deceptive pattern detection. This approach allows AutoBot to work with screenshots alone, without requiring access to the underlying HTML code, which tends to be less stable across different webpage implementations.

### Vision Module

To address high false positive rates and localization issues, the Vision Module parses a webpage screenshot and maps it to a tabular representation we call *ElementMap*. As illustrated in the figure above, the *ElementMap* contains the text associated with each UI element, along with its features: element type, bounding box coordinates, font size, background color, and font color. For UI element detection, we train a YOLOv10 model on a synthetically generated dataset. The evaluation of our model is presented below.
<div class="project-figure" style="margin-top: -2rem; margin-bottom: 2rem;">
  <div class="vision-module-flex" style="display: flex; gap: 20px; justify-content: center; align-items: flex-end;">
    <div class="vision-module-item" style="width: 48%; display: flex; flex-direction: column; align-items: center;">
      <img src="{{ '/assets/images/ui-detector.png' | relative_url }}" alt="Synthetic dataset generation pipeline for training the Web-UI Detector using YOLO model" style="width: 100%; height: auto;">
      <figcaption style="margin-top: 10px; font-size: 0.9em; text-align: center;">(a) Synthetic dataset generation pipeline for training the Web-UI Detector</figcaption>
    </div>
    <div class="vision-module-item" style="width: 48%; display: flex; flex-direction: column; align-items: center;">
      <div style="display: flex; justify-content: center; align-items: flex-end; height: 100%;">
        <div style="width: 1008px; height: 294px; overflow: hidden; display: flex; justify-content: center; align-items: flex-end;">
          <iframe src="{{ '/assets/plotly/ccs-2025-vision-eval.html' | relative_url }}"
                  style="width: 1008px; height: 530px; border: none; transform: scale(0.55); transform-origin: center bottom;margin-bottom: -30px"
                  frameborder="0">
          </iframe>
        </div>
      </div>
      <figcaption style="margin-top: 10px; font-size: 0.9em; text-align: center;">(b) Evaluation of YOLO (Ours) vs Molmo across different UI element types</figcaption>
    </div>
  </div>
  <figcaption>The Vision Module processes screenshots to extract UI elements and their features into an <em>ElementMap</em> representation.</figcaption>
</div>

### Language Module

The Language Module takes the *ElementMap* as input and maps each element to a deceptive pattern from our taxonomy. This module reasons about each element considering its spatial context and visual features. We explore different instantiations of this module—such as distilling smaller models like Qwen and T5 from a larger teacher model like Gemini—to achieve various trade-offs in terms of cost, need for training, and accuracy.

<div class="project-figure">
  <img src="{{ '/assets/images/language-module.png' | relative_url }}" alt="Language Module process showing ElementMap analysis for identifying and classifying deceptive patterns using large language models" style="width: 100%; height: auto;">
  <figcaption>The Language Module analyzes <em>ElementMap data</em> to identify and classify deceptive patterns in context.</figcaption>
</div>

## Knowledge Distillation

<div class="project-figure">
  <div style="display: flex; gap: 30px; align-items: center; flex-wrap: wrap; margin-bottom: 30px;">
    <div style="flex: 1; min-width: 300px; text-align: left;">
      <p style="margin-top: 0;">While large language models like Gemini achieve high accuracy in detecting deceptive patterns, using such closed-source models presents its own challenges such as high usage cost, considerable latency and potential data-privacy concerns. To address this, we employ knowledge distillation to create smaller, more efficient models that maintain strong detection performance while being faster and more cost-effective.</p>

      <p>We leverage <em>AutoBot</em> with Gemini as the underlying language model to label deceptive patterns, we create a large-scale dataset of labeled examples, <code><em>D<sub>distill</sub></em></code>. This synthetic dataset captures the teacher model's, i.e. Gemini's, classification along with it's reasoning for those classifications.</p>
      
      <p style="margin-bottom: 0;">Using this dataset, <code><em>D<sub>distill</sub></em></code>, we distill knowledge from the Gemini teacher model into two smaller student models: Qwen-2.5-1.5B and T5-base. The distillation process trains these models to replicate Gemini's pattern detection capabilities by learning from its predictions. This approach enables us to achieve different trade-offs across various metrics such as performance, data privacy, and latency.</p>
    </div>
    
    <div style="flex: 1; min-width: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <img src="{{ '/assets/images/distillation.png' | relative_url }}" alt="Knowledge distillation pipeline showing how AutoBot generates synthetic dataset to train smaller models like Qwen and T5 from Gemini teacher model" style="width: 100%; max-width: 500px; height: auto;">
      <figcaption style="margin-top: 10px; font-size: 0.9em; text-align: center;">Knowledge distillation pipeline: <em>AutoBot</em> generates a synthetic dataset which is used to distill smaller models.</figcaption>
    </div>
  </div>

  <div style="text-align: center; margin: 30px 0;">
    <img src="{{ '/assets/images/model-tradeoffs.png' | relative_url }}" alt="Comparison chart showing performance trade-offs between Gemini, Qwen-2.5-1.5B, and T5-base models in terms of accuracy, latency, cost, and privacy" style="width: 100%; max-width: 800px; height: auto;">
  </div>
</div>

## E2E Evaluation

To quantify the trade-offs between these three model instantiations, we evaluate *AutoBot*'s end-to-end performance on deceptive pattern detection. The interactive visualizations below compare Gemini, distilled Qwen-2.5-1.5B, and distilled T5-base across our deceptive pattern taxonomy, demonstrating how each model choice affects detection accuracy, precision, and recall at both the category and subtype levels.

<div class="project-figure" style="margin: 30px auto !important; text-align: center; display: flex; flex-direction: column; align-items: center;">
  <iframe src="{{ '/assets/plotly/ccs-2025-llm-eval-category.html' | relative_url }}" 
          style="width: 100%; max-width:90vw; height: 531px; border: none;" 
          frameborder="0">
  </iframe>
  <figcaption style="max-width:90vw;">Interactive comparison of performance of <em>AutoBot</em>(with three underlying language models: Gemini, Qwen, and T5) at the Category Level.</figcaption>
</div>

<div class="project-figure" style="margin: 30px auto !important; text-align: center; display: flex; flex-direction: column; align-items: center;">
  <iframe src="{{ '/assets/plotly/ccs-2025-llm-eval-subtype.html' | relative_url }}" 
          style="width: 1510px; max-width:90vw; height: 531px; border: none;" 
          frameborder="0">
  </iframe>
  <figcaption style="">Interactive comparison of performance of <em>AutoBot</em>(with three underlying language models: Gemini, Qwen, and T5) and <em><a href="https://dl.acm.org/doi/10.1145/3696410.3714593" target="_blank">DPGuard</a></em> at the Subtype Level</figcaption>
</div>


## Demo Video

<div class="video-section">
  <video controls muted loop playsinline preload="auto" autoplay>
    <source src="{{ '/assets/video/demo_detailed.webm' | relative_url }}" type="video/webm">
    Your browser does not support the video tag.
  </video>
  <figcaption>AutoBot in action: Detection of deceptive patterns on live websites. <a href="{{ page.demo_url }}" target="_blank">Try the interactive demo →</a></figcaption>
</div>