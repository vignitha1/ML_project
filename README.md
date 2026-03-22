# ArvyaX Machine Learning Internship Project

Theme : From Understanding Humans → To Guiding Them
Student Name : Vignitha
Internship : ArvyaX Machine Learning Internship  

## 🌿 Project Overview
This project builds an **AI system that understands human emotional state, reasons under imperfect signals, and guides users toward better mental states**.  

- Input: User journal reflections + lightweight contextual signals (sleep, stress, energy, time of day, previous mood)  
- Output:  
  - Emotional state prediction  
  - Intensity (1–5)  
  - Confidence & uncertainty flag  
  - Recommended action (e.g., breathing, journaling, deep work)  
  - Recommended timing (now, within 15 min, later today, tonight, tomorrow morning)  
- Optional: Supportive human-like message  

## 🛠️ Step 1: Data Loading & Understanding
- Loaded train.csv and test.csv  
- Explored features:  
  - journal_text, ambience_type, duration_min, sleep_hours, energy_level, stress_level, time_of_day, previous_day_mood, face_emotion_hint, reflection_quality  
  - Targets: emotional_state, intensity 

---

## 🧹 Step 2: Data Preprocessing
1. Text Cleaning: Lowercased, removed special characters, extra spaces  
2. Missing Values:  
   - Numerical → filled with median  
   - Categorical → filled with `'unknown'`  
3. Encoding:  
   - Label Encoding for categorical features  
4. Text to Numbers: TF-IDF (max 1000 features)  
5. Combined Features: TF-IDF + metadata  
6. Targets Prepared: emotional_state & intensity  

---

## 🤖 Step 3: Model Building

1. Emotional State Prediction  
   - Model: RandomForestClassifier 
   - Type: Classification  
2. Intensity Prediction
   - Treated as classification 
   - Reason: Discrete, easy for decision logic integration  
3. Confidence Score  
   - Computed using prediction probabilities  
   - Final confidence = avg of emotion + intensity confidence  
4. Uncertainty Flag 
   - Threshold < 0.6 → uncertain  

---

## 🎯 Step 4: Decision Engine (What + When)

- Rule-based system to decide action + timing 
- Inputs: predicted_state, predicted_intensity, stress_level, energy_level, time_of_day  
- Example rules:  
  - High stress → breathing / rest  
  - Low energy → light_planning or rest  
  - Positive + high energy → deep_work  
- Optional: Supportive human-like message  
  - e.g., “You seem slightly anxious. Let’s try breathing to calm your mind.”  

## 📊 Step 5: Feature Importance
- Text features contributed ~75%  
- Metadata features contributed ~25%  
- Most influential metadata: stress_level, energy_level, sleep_hours 
- Important words: tired, stressed, calm  

---

## 🔍 Step 6: Ablation Study

 Model : Text-only       
 Accuracy(Validation) : X (example: 0.68) 
 Model : Text + Metadata
 Accuracy(validation) : Y (example: 0.78)


Insights: Metadata significantly improves performance; shows importance of contextual signals.  

## ❌ Step 7: Error Analysis
10 real failure cases analyzed. Example:
- Text: “I don’t know… just feeling something weird today”  
- Actual: sad  
- Predicted: calm  
- Why: vague text, no strong emotional keywords  
- Improvement: use previous mood, sentiment scoring, transformer embeddings  

Common challenges: 
- Short/ambiguous text  
- Conflicting signals (text vs metadata)  
- Mixed emotions  
- Noisy labels  

---

## 📱 Step 8: Edge / Deployment Thinking

- Deployment: On-device, lightweight, offline-friendly  
- Model choice: RandomForest + TF-IDF → fast, small (~few MBs)  
- Latency: <100 ms, real-time response  
- Trade-offs: Fast & lightweight vs limited language understanding  
- Robustness:  
  - Short text → fallback to metadata  
  - Missing values → median/default  
  - Conflicting signals → decision rules + uncertainty  
  - Low confidence → flagged as uncertain  

Future Improvements:  
- Small transformer model (DistilBERT)  
- Memory of previous user states  
- Personalized recommendations  

## ⚡ Output
- predictions.csv ready with:  
  id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do  

## 📌 Libraries Required
pandas
numpy
scikit-learn
xgboost
scipy

