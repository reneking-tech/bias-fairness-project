# AI Fairness Project

![GitHub repo size](https://img.shields.io/github/repo-size/reneking-tech/bias-fairness-project)
![Last commit](https://img.shields.io/github/last-commit/reneking-tech/bias-fairness-project)
![License](https://img.shields.io/github/license/reneking-tech/bias-fairness-project)

## 🧠 Overview
This project explores bias detection and mitigation techniques using the UCI Adult dataset. It walks through a full machine learning workflow—data preprocessing, fairness evaluation, model training, and bias mitigation—with a focus on ensuring fairness in predictive systems.

We use logistic regression models and the AIF360 toolkit to evaluate and mitigate biases, especially around the `sex` attribute.

---

## 📁 Project Structure
```
AI_Fairness_Project/
├── data/                     # Raw and cleaned datasets
├── notebooks/               # Jupyter notebooks for each project stage
├── src/                     # Python scripts for modular code
├── models/                  # Trained model files
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and folders
└── README.md                # Project documentation
```

---

## ⚙️ Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/reneking-tech/bias-fairness-project.git
cd bias-fairness-project
```

2. **Set up the environment:**
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the Jupyter notebooks:**
```bash
jupyter notebook
```

Explore the notebooks in order:
- `Data_Cleaning.ipynb`
- `Bias_Detection.ipynb`
- `Bias_Mitigation.ipynb`
- `Train_Model.ipynb`
- `Evaluation.ipynb`

---

## 📊 Key Features
- 📉 Fairness metrics: Statistical Parity Difference, Disparate Impact
- 🧪 Model training: Logistic regression
- ⚖️ Bias mitigation: Reweighing algorithm (AIF360)
- 📊 Visualizations of prediction outcomes by group

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
