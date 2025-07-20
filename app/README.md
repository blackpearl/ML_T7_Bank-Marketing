# Bank Marketing App - Prototype Version 1.0

This is the first version of our Bank Marketing Prediction application, developed following design thinking methodology principles. The current implementation focuses on rapid prototyping to gather stakeholder feedback rather than model performance.

## About This Directory

This README is specific to the `app` directory and will be consolidated with the main project README in future iterations. The current version serves as a quick-start guide for running the prototype.

## Quick Prototype Philosophy

Following design thinking methodology:
- We prioritized quick prototyping over model performance
- Created a basic model to generate necessary pickle files
- Built a functional UI for stakeholder feedback
- Focused on user interaction and feature presentation

## Running the Application

### 1. Activate Conda Environment
```bash
conda activate bank_marketing
```

### 2. Install Requirements
```bash
pip install -r app/requirements.txt
```

### 3. Launch the Application
From the project root directory:
```bash
streamlit run app/streamlit_app.py
```

### 4. Access the Application
Open your web browser and navigate to:
```
http://localhost:8501

or 

http://localhost:8502

Check the terminal for any messages.
```

## Note
This is a prototype version. The model's predictions should not be used for actual decision-making as it's intended for UI/UX testing and feedback gathering only.
