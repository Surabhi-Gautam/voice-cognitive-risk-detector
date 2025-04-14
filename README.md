
# 🧠 Voice-Based Cognitive Decline Detection Pipeline

## 📌 Overview

This project analyzes **speech audio files** to extract **NLP and acoustic features**, applies **machine learning** to detect anomalies, and visualizes trends that may indicate **cognitive decline**.  
It's built using Whisper, Librosa, Scikit-learn, and runs fully in Python.

---

## 📂 Project Structure

```
voice-risk-detector/
├── cognitive_decline_pipeline.ipynb  # Main notebook
├── cognitive_risk_report.csv         # Output CSV with risk scores
├── your_audio_dataset.zip            # Example audio input (not uploaded here)
├── README.md                         # Project documentation
└── requirements.txt                  # All dependencies
```

---

## 🚀 Features

- 🗣️ Transcribes audio using [Whisper](https://github.com/openai/whisper)
- 📊 Extracts NLP features: hesitation, word count, sentence length
- 🎧 Extracts audio features: pitch variability, tempo, duration
- 🧪 Uses Isolation Forest to flag cognitive risk
- 📈 Visualizes patterns via Seaborn plots
- 🧰 Optional: API-ready risk scoring function

---

## 📁 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/voice-cognitive-risk-detector.git
   cd voice-cognitive-risk-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   Open `cognitive_decline_pipeline.ipynb` in Jupyter Notebook or VSCode.

4. **Provide your own `.zip` of `.wav` files** or test with provided structure.

---

## 🔬 Deliverables

✅ Clean and reproducible Python notebook  
✅ Sample visualizations of extracted features and anomalies  
✅ Optional API function to score risk from a single audio input

---

## 📈 Sample Output

| File                          | Pitch Std | Tempo | Risk Flag |
|------------------------------|-----------|-------|-----------|
| 03-01-01-01-01-01-01.wav      | 23.2      | 92.3  | -1 (Risk) |
| 03-01-01-01-01-02-01.wav      | 5.4       | 120.5 | 1 (Normal)|

> Pairplot of features shows separation of potential anomalies.

---

## 🧪 Risk Scoring Function (Optional)

You can use this function in a web app or API:

```python
def score_audio_file(file_path):
    text, _ = transcribe_audio(file_path)
    nlp = extract_nlp_features(text)
    audio = extract_audio_features(file_path)
    features = {**nlp, **audio}
    return model.predict([scaler.transform([list(features.values())])])
```

---

## 🧾 Requirements

- Python 3.9+
- whisper
- librosa
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install via:
```bash
pip install -r requirements.txt
```

---

## 💡 Future Enhancements

- Add a Streamlit web interface
- Track cognitive change over time per speaker
- Support more audio formats + voice diarization

---

## 📬 Contact

Made with ❤️ for Daksh'25 STEM Hackathon  
Project by **[Your Name]**  
[LinkedIn] | [GitHub]
