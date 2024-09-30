# Face Guard

[Link to Webapp](https://face-guard-louis-jz.streamlit.app/)
[Link to Paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1809.00888)

## Background

Deepfake detection uses machine learning to spot manipulated videos or images, where faces or voices are altered to create false content. 
It helps combat digital deception by identifying fake media, preventing misinformation, fraud, and reputational harm, 
thus protecting the credibility of online information.

## Technical Overview

The neural network model was trained based on the MesoNet architecture on the deep fake dataset provided by the author.
please find the code at `training_mesonet.ipynb`. This app is deployed on streamlit, with the video inference pipeline hosted on Google VertexAI servers. 
