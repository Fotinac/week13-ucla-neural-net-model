
# UCLA Admission Neural Network Classifier

This project is part of **CST2216: Individual Term Project** at Algonquin College (Week 13).  
It builds and deploys a modular neural network pipeline to predict graduate admission chances at UCLA based on test scores, GPA, and recommendation strength.

---

## Live App

[Streamlit app link coming after deployment]

---

## Project Features

- Modularized code using custom Python scripts
- Data loading and preprocessing (binary classification setup)
- 2-layer neural network built with TensorFlow/Keras
- Model evaluation with accuracy, classification report, confusion matrix
- Visualization of training accuracy and loss
- Deployed as a Streamlit app (user can upload data or use default)

---

## Folder Structure

```
week13_ucla_neural_net_model/
├── app.py                  ← Streamlit app interface
├── main.py                 ← Script to run everything locally
├── requirements.txt        ← Dependencies for deployment
├── README.md               ← This file
├── data/                   ← Input CSV file(s)
├── logs/                   ← Logs (e.g., app.log)
├── models/                 ← Neural network architecture + training
├── utils/                  ← Data loader, preprocessing, evaluation, visuals
```

---

## Run Locally

1. Clone this repo:
```bash
git clone https://github.com/Fotinac/week13-ucla-neural-net-model.git
cd week13-ucla-neural-net-model
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run app.py
```

---

## Dependencies

- `streamlit`
- `tensorflow`
- `pandas`
- `scikit-learn`
- `matplotlib`

---

## Author

- **Name**: Fotinacao
- **Course**: CST2216 — Business Intelligence System Infrastructure
- **Institution**: Algonquin College
- **Instructor**: Swapnil Kangralkar

---

## License

This project is for educational purposes only.
