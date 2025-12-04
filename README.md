# Conjoint Analysis Implementation in Python
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-latest-blue)](https://www.statsmodels.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


This repository contains practical implementations of **Conjoint Analysis**, a statistical technique used in market research and policy-making to determine how people value different attributes (features) that make up an individual product or service.

By deconstructing a product into its constituent attributes, we can use regression analysis to calculate the **Part-Worth Utility** of each feature, allowing for data-driven decision-making.

## Repository Contents

This repository features two distinct case studies:

1.  **Urban Planning & Policy (`Conjoint-Analysis-Planning-Area.ipynb`)**:
    * **Context:** Helping development agencies (e.g., Bappenas) design sustainable residential districts.
    * **Goal:** Balance trade-offs between green space, housing density, transport, and commercial zones.
2.  **Product Market Research (`Conjoint-Analysis.ipynb`)**:
    * **Context:** Food and Beverage industry (Pizza).
    * **Goal:** Determine customer preferences regarding toppings, crust types, and pricing sensitivity.

---

## Theoretical Framework

Conjoint Analysis generally relies on the **Additive Utility Model**. It assumes that the total value (Total Utility) a respondent derives from a specific profile is the sum of the values of its separate parts plus a constant and an error term.

### The Formula

$$U_{total} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \epsilon$$

Where:
* $U_{total}$: The total utility (rating or preference score) of a profile.
* $\beta_0$: The intercept (baseline utility).
* $\beta_i$: The coefficient (Part-Worth Utility) for attribute level $i$.
* $x_i$: The dummy variable (0 or 1) representing the presence of attribute level $i$.
* $\epsilon$: Error term.

---

## Implementation Workflow

Both notebooks follow a standard data science workflow for Conjoint Analysis:

### 1. Experimental Design
We define the **Attributes** (features) and **Levels** (options within features).

**Example (Urban Planning):**
| Attribute | Levels |
| :--- | :--- |
| **Green Space** | 10%, 30%, 50% |
| **Housing Density** | Low Rise, Mid Rise, High Rise |
| **Transport** | Bus Only, Bus + Metro, None |
| **Commercial** | Mixed-use, Separate Zone, Distant |

### 2. Profile Generation (Full Factorial)
We generate all possible combinations of attributes using `itertools`.
```python
import itertools
keys, values = zip(*attributes.items())
profiles = [dict(zip(keys, v)) for v in itertools.product(*values)]
```
### 3. Data Simulation (Survey)
Since real-world survey data is not available for these demonstrations, the notebooks utilize synthetic data generation to simulate customer responses.

* **Ground Truth Definition:** We define a hidden set of preferences (utilities) representing the "true" desires of the population.
    * *Example (Pizza):* "Loves Pepperoni (+3)", "Hates Veggie (-2)".
    * *Example (Urban Planning):* "Citizens strongly prefer Metro (+6)", "Citizens like Green Space (+3)".
* **Noise Injection:** To mimic realistic variability and human error in surveys, random noise (using `np.random.normal`) is added to the calculated utility scores.
* **Respondent Generation:** The Urban Planning case study simulates 50 distinct respondents rating random subsets of profiles to create a robust dataset.

```python
# Logic example from the notebooks:
def simulate_rating(row):
    score = base_score
    score += preferences[row['Attribute']] # Add utility based on preference
    score += np.random.normal(0, sigma)    # Add random noise
    return np.clip(score, 0, 10)           # Clip to rating scale
```
### 4. Data Preparation (Dummy Coding)
To perform regression analysis, categorical attributes (e.g., text labels like "Pepperoni" or "Bus Only") must be converted into numerical format. We utilize **Dummy Coding** (One-Hot Encoding) for this purpose.

* **Avoiding Multicollinearity:** We use `drop_first=True` when generating dummy variables. This prevents the "dummy variable trap" by removing one level from each attribute, which serves as the **Baseline** or **Reference Level**.
* **Intercept:** A constant term (intercept) is added to the model to capture the baseline utility.

```python
# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(df_survey[list(attributes.keys())], drop_first=True, dtype=int)

# Add a constant term for the regression intercept
X = sm.add_constant(X)
```
### 5. Analysis: OLS Regression
We utilize **Ordinary Least Squares (OLS)** regression via the `statsmodels` library to quantify the preferences. The model treats the survey rating as the dependent variable ($Y$) and the dummy-coded attributes as independent variables ($X$).

* **Coefficients ($\beta$):** The resulting coefficients represent the **Part-Worth Utilities**. These values quantify the utility gain or loss associated with a specific attribute level compared to the baseline level (the one dropped during dummy coding).
* **Model Fit:** Metrics like R-squared are used to evaluate how well the model explains the variance in respondent preferences.

```python
import statsmodels.api as sm

# Initialize and fit the OLS model
model = sm.OLS(y, X).fit()

# Display detailed statistics, including coefficients and p-values
print(model.summary())
```
### 6. Interpretation & Visualization
The raw regression output is transformed into actionable business or policy insights through visualization and the calculation of attribute importance.

#### Interpreting Part-Worth Utilities
The regression coefficients serve as the **Part-Worth Utilities**. They explain preference relative to the *Reference Level* (the category dropped during dummy coding).

* **Positive Coefficient:** The attribute level increases total utility (preferred over the baseline).
* **Negative Coefficient:** The attribute level decreases total utility (less preferred than the baseline).
* **Magnitude:** The absolute size of the coefficient indicates the strength of the preference.

#### Calculating Relative Importance
To determine which attribute acts as the primary driver for decision-making (e.g., "Do citizens care more about Green Space or Transport?"), we calculate the **Relative Importance**:

1.  Calculate the **Range** for each attribute:
    $$\text{Range}_i = \text{Max(Utility)}_i - \text{Min(Utility)}_i$$
2.  Calculate the **Importance Percentage**:
    $$\text{Importance \%}_i = \frac{\text{Range}_i}{\sum \text{Range}_{\text{all attributes}}} \times 100$$

---

## 📊 Key Results & Policy Recommendations

Based on the synthetic data analysis performed in the included case studies, the following insights were derived:

### Case Study 1: Urban Planning (Policy Making)
The analysis suggests that citizens prioritize infrastructure functionality and livability over building density.
* **Transport is Critical:** The transition from "None" to "Bus + Metro" yielded the highest utility gain, suggesting that high-quality public transport is the most effective lever for citizen satisfaction.
* **Green Space Trade-offs:** Respondents showed a quantitative willingness to accept higher housing density (e.g., Mid-Rise) *if and only if* it is compensated with higher Green Space coverage (50%).
* **15-Minute City Concept:** There is a strong, positive preference for "Mixed-use" commercial zones compared to distant ones, supporting walkable city planning parameters.

### Case Study 2: Product Development (Pizza)
* **Price Sensitivity:** The utility scores reveal a non-linear response to pricing. There is a sharp decline in utility when the price moves to the highest tier ($15), indicating a clear price ceiling for the target market.
* **Product Formulation:** The analysis isolates specific winning ingredients (e.g., Pepperoni and Stuffed Crust) that significantly boost the product's appeal regardless of other factors.

---

## 💻 Requirements

To run the notebooks in this repository, you will need Python installed along with the following data science libraries:

* `pandas` (Data manipulation)
* `numpy` (Numerical operations)
* `statsmodels` (Statistical modeling & Regression)
* `matplotlib` & `seaborn` (Data visualization)

You can install all dependencies via pip:

```bash
pip install pandas numpy statsmodels matplotlib seaborn
```
