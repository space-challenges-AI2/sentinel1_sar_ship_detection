# WaveTrack.AI - Ship Detection Backend

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the source code for the backend service that powers **WaveTrack.AI**, a demonstration platform for real-time ship detection from SAR (Synthetic Aperture Radar) imagery. The goal of this project is to showcase a custom-trained YOLOv5 model served via a fast, scalable, and efficient web API.

**[Try the live web application here](https://your-app-url.com)** 👈

---

## Overview

Detecting vessels in SAR imagery presents unique challenges due to noise, varying resolutions, and complex coastal environments. This backend provides a robust solution by leveraging a state-of-the-art deep learning model, making advanced geospatial analysis accessible through a simple web interface.

### Key Features

-   🧠 **Accurate AI Detection:** Utilizes a custom-trained YOLOv5 model optimized to identify ships in complex SAR scenes with high precision and recall.
-   🚀 **High-Performance API:** Built with FastAPI, the API is fully asynchronous and delivers low-latency responses, making it ideal for interactive applications.
-   🖼️ **Flexible Image Processing:** The API processes user-uploaded images and returns structured data, including vessel counts, confidence scores, and annotated images.
-   ☁️ **Cloud-Ready:** Designed to be lightweight and stateless for easy deployment on modern cloud hosting platforms.

---

## Technology Stack

The backend is built with a curated selection of modern, high-performance technologies to ensure reliability and speed.

| Component              | Technology                               | Purpose                                          |
| ---------------------- | ---------------------------------------- | ------------------------------------------------ |
| **Language** | Python 3.9+                              | The core language for the application.           |
| **Web Framework** | FastAPI                                  | For building a high-performance, asynchronous API. |
| **ASGI Server** | Uvicorn                                  | A lightning-fast server to run the application.  |
| **Deep Learning** | PyTorch & YOLOv5                         | For running inference with our trained model.    |
| **Image Processing** | OpenCV & Pillow                          | For handling and manipulating image data.        |

---

## System Architecture

The API acts as the central engine of the WaveTrack.AI platform. It receives an image from the frontend client, performs inference using the YOLOv5 model, processes the results, and returns a structured JSON response. This separation of concerns allows the frontend to remain lightweight and focused solely on user experience.



### API Functionality

The backend exposes a simple yet powerful set of endpoints:

-   `POST /predict/`: The core endpoint. It accepts an image file and returns a JSON object containing the number of detected vessels, their confidence scores, and a base64-encoded image with detection boxes drawn on it.
-   `GET /sample-images`: Provides a list of pre-loaded SAR image filenames that the frontend can use for demonstration purposes.

This architecture ensures that all heavy computational work is handled efficiently on the server, providing a smooth experience for the end-user.