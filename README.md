# -Data-Analyst-Job-Market-Analysis-Salary-Prediction


# 📊 Data Analyst Jobs - Salary and Industry Analysis

This project analyzes a dataset of data analyst job postings to uncover trends in salaries, job roles, company attributes, and industry insights. It also uses machine learning to predict average salaries based on features like company rating, tech skills, and founding year.

---

## 🔍 Project Overview

This project includes:

- Cleaning and preprocessing job listing data
- Exploratory Data Analysis (EDA)
- Visualizing job and salary trends
- Feature engineering (skills, locations)
- Predicting salary using a Random Forest Regressor
- Ranking top jobs by salary and rating

---

## 📁 Dataset

- **Filename**: `data_analyst_jobs.csv`
- **Sample Columns**:
  - `Job Title`, `Company Name`, `Location`
  - `Salary Estimate`, `Rating`, `Founded`
  - `Type of ownership`, `Industry`, `Sector`, `Size`
  - `Job Description`

---

## 📊 Key Outputs & Visualizations

The following visual insights are generated and saved as PNG files:

| 📂 Insight                        | 📸 Filename                     |
|----------------------------------|----------------------------------|
| Correlation Matrix               | `correlation_matrix.png`         |
| Salary Distribution              | `salary_distribution.png`        |
| Company Ratings by Industry      | `ratings_by_industry.png`        |
| Top 10 Job Titles                | `top_10_jobs.png`                |
| Average Salary by Job Title      | `avg_salary_by_job_title.png`    |
| Top 20 Job Locations             | `top_20_locations.png`           |
| Company Size Distribution        | `company_size_distribution.png`  |
| Avg Salary by Company Size       | `avg_salary_by_size.png`         |
| Top 20 Types of Ownership        | `top_20_ownership.png`           |
| Sector Distribution              | `top_20_sectors.png`             |
| Avg Salary by Sector             | `avg_salary_by_sector.png`       |

---

## 📈 Machine Learning: Salary Prediction

- **Model**: Random Forest Regressor
- **Features Used**:
  - `Rating`
  - `Tech_Skills` (from Python & Excel keyword presence)
  - `Founded` year
- **Target Variable**: `Avg Salary`
- **Metrics**:
  - Mean Absolute Error (MAE)
  - R² Score

---

## 🏆 Top Job Insights

The script also prints the **top 10 highest paying jobs** ranked by `Avg Salary` and `Company Rating`.

---

Install Dependencies

Make sure you have Python 3.x installed. Then run:

pip install -r requirements.txt

---

 Add Dataset
 
Place your CSV file named data_analyst_jobs.csv in the root directory of the project.

---

🛠 Requirements

List of required libraries:

pandas
matplotlib
seaborn
scikit-learn

---

🧠 Technologies Used
Python

Pandas

Matplotlib

Seaborn

Scikit-learn

---

📜 License
This project is licensed under the MIT License – you are free to use, modify, and distribute with attribution.


You can **append this directly to the end** of the README file I gave earlier.

Let me know if you want a downloadable `.zip` or `.md` version of this whole file.

----
