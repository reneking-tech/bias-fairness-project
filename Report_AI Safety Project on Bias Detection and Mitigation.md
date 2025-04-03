Report: AI Safety Project on Bias Detection and Mitigation

By René King

# 1\. Introduction

## 1.1 Motivation

Artificial Intelligence systems are increasingly deployed in high-stakes domains such as hiring, lending, and criminal justice. However, these systems can inadvertently perpetuate or amplify societal biases if trained on skewed data. Addressing these biases is a cornerstone of AI Safety, ensuring that AI systems are fair, transparent, and equitable. This project demonstrates practical steps to detect and mitigate bias in a machine learning pipeline, highlighting the importance of ethical considerations in AI development.

## 1.2 Project Objectives

This project aimed to:

1. Detect and quantify bias in the UCI Adult dataset using fairness metrics.  

2. Apply a bias mitigation technique (Reweighing) to reduce disparities.  

3. Evaluate the trade-offs between fairness and model performance.  

4. Document the methodology and results to showcase competence in AI fairness and safety.  

# 2\. Methodology and Tools

## 2.1 Tools and Libraries

- Python 3.8+ - primary programming language.  

- AIF360 (IBM AI Fairness 360) for fairness metrics and mitigation algorithms.  

- pandas, NumPy, scikit-learn for data manipulation, preprocessing, and modeling.  

- Matplotlib for visualizing fairness metrics and results.  

- Jupyter Notebooks for interactive experimentation and documentation.  

## 2.2 Dataset

- UCI Adult Dataset: A benchmark dataset for fairness research, containing demographic and income data. Key features include:  
  - Protected attributes: sex (binary: Male=1, Female=0) and race.  

  - Target variable: income_binary (binary: ≤50K=0, >50K=1).  

## 2.3 Project Workflow

1. Data cleaning and preprocessing  
    - Handled missing values and encoded categorical features.  

    - Split the data into training (70%) and testing (30%) sets.  

2. Bias detection  
    - Computed fairness metrics (Statistical Parity Difference, Disparate Impact) using AIF360.  

3. Bias mitigation  
    - Applied the Reweighing algorithm to adjust instance weights for fairness.  

4. Model training and evaluation  
    - Trained a Logistic Regression model on both original and reweighted data.  

    - Compared accuracy and fairness metrics pre- and post-mitigation.  

5. Visualisation and reporting  
    - Plotted positive prediction rates by protected groups.  

    - Summarized findings in a comprehensive report.  

# 3\. Implementation Details

## 3.1 Data Preparation

- Cleaning: removed rows with missing values and encoded categorical variables (e.g., sex, race) numerically.  

- Feature engineering: created a binary target column (income_binary) from the original income feature.  

- Splitting: divided the dataset into training and testing sets for unbiased evaluation.  

## 3.2 Bias Detection

Using AIF360, the following fairness metrics were computed on the test set:

- Statistical Parity Difference (SPD): measures the difference in positive outcome rates between privileged (Male) and unprivileged (Female) groups.  
  - Initial SPD: -0.175 (indicating bias against Females).  

- Disparate Impact (DI): ratio of positive outcomes for unprivileged vs. privileged groups.  
  - Initial DI: 0.163 (far below the fairness threshold of 0.8).
- **(Visualization)** Positive prediction rates for Females (sex=0) were significantly lower (3.4%) than for Males (sex=1, 20.9%).  

## 3.3 Bias Mitigation (Reweighing)

- Reweighing Algorithm: adjusted instance weights to balance outcomes across groups.  

- Post-Mitigation Metrics:  
  - SPD: Improved toward zero (-0.05).  

  - DI: Increased to 0.85 (closer to the ideal value of 1.0).  

## 3.4 Model Performance

- Logistic Regression Results:  
  - Original Data:  
    - Accuracy: 82.6%  

    - Fairness Metrics: SPD=-0.175, DI=0.163  

  - Reweighted Data:  
    - Accuracy: 82.4% (minimal drop)  

    - Fairness Metrics: SPD=-0.05, DI=0.85  

Trade-off: the mitigation improved fairness slightly at the cost of a negligible accuracy reduction.

# 4\. Results and Analysis

## 4.1 Key Findings

1. Bias detection  
    - The initial dataset exhibited significant bias against Females, with a Disparate Impact of 0.163.  

2. Mitigation effectiveness  
    - Reweighing improved fairness metrics (DI increased to 0.85), demonstrating its utility for bias reduction.  

3. Performance impact  
    - Model accuracy remained stable (around 82.5%), showing that fairness improvements need not drastically harm performance.  
        <br/>

## 4.2 Limitations

- Single protected attribute - focused only on sex; future work could include race or intersectional analysis.  

- Mitigation technique - reweighing is one of many methods; others (e.g., Adversarial Debiasing) could yield different results.  

# 5\. Conclusion and Future Work

## 5.1 Summary

This project successfully

- Detected bias in the UCI Adult dataset using AIF360.  

- Mitigated bias using Reweighing, improving fairness metrics without significant accuracy loss.  

- Demonstrated the importance of fairness-aware machine learning in AI Safety.  

## 5.2 Future Directions

1. Expand protected attributes - analyze bias across multiple attributes (e.g., sex and race).  

2. Advanced mitigation - experiment with techniques like Adversarial Debiasing or Post-processing methods.  

3. Real-world deployment - study how these methods perform in production systems.  

# 6\. References

1. AIF360 documentation: <https://aif360.readthedocs.io/>
2. UCI Adult Dataset: <https://archive.ics.uci.edu/ml/datasets/adult>
3. Fairness metrics: Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning. fairmlbook.org.  

**Appendix: Repository Structure  
**AI_Fairness_Project/  
├── data/  
│ ├── raw/ (Raw UCI Adult dataset)  
│ ├── train_cleaned.csv (Cleaned dataset)  
│ └── weighted_train.csv (Reweighted dataset)  
├── notebooks/  
│ ├── 01_data_cleaning.ipynb  
│ ├── 02_bias_detection.ipynb  
│ ├── 03_bias_mitigation.ipynb  
│ └── 04_evaluation.ipynb  
├── models/ (Saved models)  
└── README.md (Project overview)

GitHub Link: <https://github.com/reneking-tech/bias-fairness-project>