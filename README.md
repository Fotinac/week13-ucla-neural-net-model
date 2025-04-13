
## Live App  
[Click to open the app](https://week13-ucla-neural-net-model-elfrs4ibngggsr9hwdt23f.streamlit.app)

---

# UCLA Admission Neural Network Classifier

This project was developed as part of **CST2216: Individual Term Project** at Algonquin College (Week 13).  
It builds and deploys a modular neural network pipeline that predicts graduate admission chances at UCLA based on standardized test scores, GPA, and recommendation strength.

---

## Project Features

- Modularized codebase using custom Python scripts
- Data loading and preprocessing for binary classification
- Two-layer neural network built with TensorFlow/Keras
- Model evaluation via accuracy, classification report, and confusion matrix
- Training accuracy and loss visualization
- Deployed as an interactive Streamlit web app

---

## 📁 Folder Structure

```
week13_ucla_neural_net_model/
├── app.py                  ← Streamlit app interface
├── main.py                 ← Script to run the full pipeline
├── requirements.txt        ← Python dependencies
├── README.md               ← Project documentation
├── data/                   ← Input dataset (Admission.csv)
├── logs/                   ← Log files (e.g., app.log)
├── models/                 ← Trained model (admission_model.h5)
├── utils/                  ← Data loaders, preprocessing, evaluation, plotting
```

---

## 💻 Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/Fotinac/week13-ucla-neural-net-model.git
cd week13-ucla-neural-net-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch the Streamlit app:**
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

This project is developed for educational purposes only.
