# Datacorp — AI-Powered Data Cleaning & Analytics

Datacorp is an interactive data cleaning and analytics application built with **Python** and **Streamlit**. It allows users to upload datasets, identify data-quality issues, clean and transform data, perform advanced analysis, generate visualizations, and download processed results.

---

## Features

* Upload **CSV, Excel, and JSON** datasets
* Automated data-quality analysis
* Missing-value detection and handling
* Duplicate detection and removal
* Data-type conversion
* Outlier detection and handling
* Fuzzy duplicate matching
* Data cleaning and transformation tools
* Interactive data visualizations
* Statistical analysis
* Correlation analysis
* Time-series analysis
* Pivot-table generation
* Geographic data visualization
* AI-powered prediction and anomaly detection
* Data-quality scoring
* Automatic data cleaning
* Undo and redo cleaning operations
* Export cleaned datasets
* Generate TXT and PDF reports
* Session-based processing without permanent data storage

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
| Matplotlib   | Data visualization                       |
| Seaborn      | Statistical visualization                |
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


# AI-Powered Analysis

Datacorp includes machine-learning-based capabilities for advanced data analysis.

These capabilities can be used for:

* Prediction
* Anomaly detection
* Pattern identification
* Advanced data analysis

The available AI-powered functionality may vary depending on the structure, size, and characteristics of the uploaded dataset.

---

# Supported File Formats

| File Type | Extensions      | Support |
| --------- | --------------- | ------- |
| CSV       | `.csv`          | Yes     |
| Excel     | `.xlsx`, `.xls` | Yes     |
| JSON      | `.json`         | Yes     |

---

# Data Privacy

Datacorp processes uploaded datasets during the active application session.

The application is designed for **session-based processing without permanent data storage**.

Users should avoid uploading confidential, sensitive, or personally identifiable information unless they understand and accept the environment in which the application is running.

---



# Future Improvements

Potential future enhancements include:

* Automated data-cleaning recommendations
* Natural-language data querying
* Automated exploratory data analysis
* Advanced feature engineering
* Additional machine-learning models
* Database connectivity
* Cloud storage integration
* Scheduled data-quality monitoring
* Additional export formats
* Role-based access control
* Advanced dashboard customization

---

# Author

**Anurag**

* GitHub: [@anuragverse](https://github.com/anuragverse)

---

# License

This project is intended for **educational, development, and demonstration purposes**.

