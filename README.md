# Financial Distress AI Agent

An AI-powered Streamlit application for analyzing financial distress across multiple companies.

## Features

### 📊 Data Panel
- Upload CSV files with company financial data
- Automatic detection of company columns
- Data preview and statistics

### 💬 Chatbot
- Interactive chat to query company status
- Ask questions like:
  - "What is the status of each company?"
  - "Show statistics per company"
  - "Which companies are at risk?"

### 📈 Visual Analysis
- Company status comparison chart
- Correlation heatmap
- Distribution plots

### 📋 Summary & Decisions
- **Individual verdict** per company:
  - 🚨 FINANCIALLY DISTRESSED (60%+ risk)
  - ⚠️ FINANCIALLY AT RISK (40-59% risk)
  - ✅ FINANCIALLY HEALTHY (<40% risk)
- Overall summary statistics
- Risk probability visualization
- Actionable recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## CSV Format

The CSV should contain:
- A column identifying companies (e.g., "company", "company_name", "firm_id")
- Numeric financial columns
- Optional "year" column for trend analysis

Example:
```csv
company,year,revenue,expenses,debt,cash_flow
Company_A,2020,1000000,800000,500000,200000
Company_A,2021,1100000,900000,450000,200000
Company_B,2020,500000,300000,100000,200000
Company_B,2021,600000,350000,150000,250000
```

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn

## License

MIT