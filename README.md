---
title: TrackMyHealth
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: mit
tags:
  - health
  - llm
  - nemotron
  - modal
  - mongodb
  - symptom-tracker
  - gradio-hackathon
---

# TrackMyHealth — Daily Symptom Tracker

A private daily health journal powered by Nemotron Nano 4B running on Modal.
Log symptoms throughout the day, generate clinical PDF reports with timelines, and chat with your own health history.

## Tech Stack

| Layer | Technology |
| --- | --- |
| UI | Gradio 4 (custom UI) |
| GPU Runtime | Modal (serverless T4 GPU) |
| LLM | nvidia/Nemotron-Nano-4B-Instruct via llama.cpp |
| Database | MongoDB Atlas |
| Reports | ReportLab (PDF) |

## Hackathon Awards Targeted

- 🐜 Tiny Titan (4B model)
- 🟩 NVIDIA Nemotron Quest
- 🟢 Modal Awards
- 🤖 Best Agent
- 🎨 Off-Brand custom UI
- 🏡 Backyard AI track

## Setup

1. Clone repo
2. Copy .env.example to .env and fill credentials
3. Run: modal run modal_app.py::download_model
4. Run: python app.py

## Modal + Hugging Face Space Guide

Yes, the Hugging Face Space should run the Gradio frontend, while Modal hosts the GPU LLM functions. The Space does not need to keep a raw HTTP endpoint for the model in this codebase. Instead, `app.py` imports `modal_app.py` and calls Modal functions with `.remote(...)`, such as `modal_app.structure_entries.remote(...)` and `modal_app.chat_with_history.remote(...)`.

To connect Modal:

1. Create a Modal account and install the Modal CLI locally.
2. Run `modal token new` and complete the browser login.
3. Download the model into the Modal volume with `modal run modal_app.py::download_model`.
4. Deploy the Modal app with `modal deploy modal_app.py`.
5. In Hugging Face Space secrets, add the Modal credentials from your local Modal token:
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`
6. Also add your database secret:
   - `MONGO_URI`
7. Deploy the Space. When the Gradio app needs the LLM, it calls Modal remotely and Modal spins up the T4 GPU container.

The model file lives in the Modal volume named `model-cache` at `/models/nemotron-nano-4b.gguf`. The Space only stores UI code and credentials; the GPU work and model loading happen inside Modal.

## How it works

- Write path: raw entries saved instantly to MongoDB, no LLM on every input
- Report path: Modal spins up Nemotron Nano, batch processes entries, generates PDF
- Chat path: LLM retrieves user's MongoDB entries and answers grounded questions
