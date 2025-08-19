# Text Summarizer App

A web application that summarizes long texts using state-of-the-art transformer models.

## 🚀 Live Demo

Try the live app here: [Live Demo](https://huggingface.co/spaces/toqeeryasir/Summarizer-For-News-and-Scientific-Articles)

## 📋 Features

- Summarizes long texts using Pegasus transformer models
- Handles text chunking for documents longer than model context
- Web interface built with Gradio
- Deployed on Hugging Face Spaces
- GPU-accelerated inference

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Hugging Face Transformers** - Model inference
- **PyTorch** - Deep learning framework
- **Gradio** - Web interface
- **Hugging Face Spaces** - Deployment platform

## 🏗️ Architecture

text-summarizer-app/
│
├── app.py                 # Main application code
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore file
├── assets/              # Screenshots, demo images
│   └── app-screenshot.png
└── examples/            # Sample texts for testing
    └── sample_texts.txt
