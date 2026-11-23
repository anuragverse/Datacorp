# Datacorp — AI-Powered Data Cleaning & Analytics

Datacorp is an interactive data cleaning and analytics application built with **Python** and **Streamlit**. It allows users to upload datasets, identify data-quality issues, clean and transform data, perform advanced analysis, generate visualizations, and download processed results.

---

## How it works

The sidebar holds your files, the sample-data switch and **Undo / Redo / Reset**. The four pages follow the natural workflow:

| Page | What you can do |
| ---- | --------------- |
| **Overview** | Quality score, detected issues, one-click **Auto-fix**, insights, data preview |
| **Clean** | Missing values, exact and fuzzy duplicates, rename/drop columns, text cleaning, find & replace, data-type conversion, outliers, merging files |
| **Analyze** | Distribution, scatter, categories, correlation, trend + seasonality, pivot table, world map, prediction (random forest), anomaly detection |
| **Export** | Cleaned data as CSV or Excel, TXT and PDF reports, before/after summary, cleaning history |

Every change is a step in the history, so it can be undone and redone. Sessions are processed in memory only and nothing is stored.

---

## Technology Stack

| Technology   | Purpose                                  |
| ------------ | ---------------------------------------- |
| Python       | Core programming language                |
| Streamlit    | Web application framework                |
| Pandas       | Data manipulation and analysis           |
| NumPy        | Numerical computing                      |
| Scikit-learn | Machine learning and predictive analysis |
| Plotly       | Interactive visualizations               |
| Statsmodels  | Time-series decomposition                |
| FPDF         | PDF report generation                    |
| RapidFuzz    | Fuzzy duplicate matching                 |

---

## Project Structure

```text
Datacorp/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── .devcontainer/
│   └── devcontainer.json
└── ...
```

---

# Installation

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/anuragverse/Datacorp.git
```

Move into the project directory:

```bash
cd Datacorp
```

---

## 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

After activation, your terminal should indicate that the `venv` environment is active.

---

## 4. Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you encounter a pip-related issue, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Open the URL in your browser to access Datacorp.


# Supported File Formats

| File Type | Extensions      | Support |
| --------- | --------------- | ------- |
| CSV       | `.csv`          | Yes     |
| Excel     | `.xlsx`         | Yes     |
| JSON      | `.json`         | Yes     |

---

# Data Privacy

Datacorp processes uploaded datasets during the active application session.

The application is designed for **session-based processing without permanent data storage**.

Users should avoid uploading confidential, sensitive, or personally identifiable information unless they understand and accept the environment in which the application is running.

---
