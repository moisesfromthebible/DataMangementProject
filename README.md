# US Election Data

This project focuses on analyzing county-level demographic, socioeconomic, and electoral data to develop a predictive model for U.S. presidential election outcomes


## Dataset Sources (already inside the model section)
The project utilizes multiple publicly available datasets:
1.  **County Presidential Election Returns 2000-2020**
   - [Election Results 2000-2020](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ)
2.  **Unemployment and Median Household Income**
   - [USDA County-Level Data Sets](https://www.ers.usda.gov/data-products/county-level-data-sets/)
   - Latest data: Unemployment data until 2022.
3. **Education Data**
   - Source: [USDA County-Level Data Sets](https://www.ers.usda.gov/data-products/county-level-data-sets/)
   - Education data from 2000 to 2022
4. **Population Estimates**
   - [Intercensal Estimates (2000–2010)](https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2000-2010-counties.html)
   - [Annual Population Estimates (2010s)](https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-counties-total.html)
   - [Annual Estimates (2020s)](https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html)
   - Latest data: Annual population estimates up to July 2023.


## Setup Instructions

### 1. Clone the Repository
Start by cloning the project repository from GitHub:
```
git clone <repository-url>
cd <repository-name>
```

### 2. Install Required Libraries
Install the dependencies listed in the `requirements.txt` file:
```
pip install -r requirements.txt
```

### 3. Run the Application
Start the server and test the functionality:
```
python app.py
```

---

## Tools and Libraries Used
The project uses the following Python libraries:
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (RandomForestClassifier, GridSearchCV)
- **Evaluation Metrics:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`
- **Web Framework:** `Flask`
- **Utilities:** `joblib`, `requests`

---


## Video Demo
A demonstration video of the project is available to help you understand the workflow and functionality.

---

## Support
For any issues, questions, or suggestions, feel free to reach out to us .
