# SlidePark-AI

> Demand forecasting and venue ops automation for an indoor slide park — built during a real internship, solving real operational problems.

Built for **Slick City Action Park** (Slide Park Georgia LLC) as part of a Data Science & AI Automation internship under CEO Mehtab Wasi.

---

## The problem

Indoor entertainment venues face highly variable demand — weekends vs weekdays, weather, school calendars, local events. Without a model, they overstaff slow days and understaff busy ones. Every wrong call costs money.

---

## What's built

### 1. Demand Forecasting Pipeline (`forecasting/`)

An ensemble forecasting model trained on synthetic-but-realistic venue data:

- **Feature engineering:** day of week, month, school calendar flags, weather proxy, local events, holiday indicators
- **Models:** Random Forest + Gradient Boosting ensemble with cross-validation
- **Results: R² = 0.91, MAPE ≈ 12%** on validation set
- Rolling validation to simulate real-world deployment

### 2. Staff Scheduling Optimizer (`scheduling/`)

Linear programming model that takes a demand forecast as input and outputs the minimum-cost staffing plan:

- Built with Python + PuLP
- Constraints: minimum coverage per hour, max shift length, staff availability
- Outputs optimal shift assignments for each role (cashier, attendant, safety)
- Reduces theoretical overstaffing by ~18% vs heuristic scheduling

### 3. Seven-Tab Excel Workbook

Delivered to the operations team for live planning:

| Tab | Content |
|-----|---------|
| Forecast | Predicted demand by day with confidence bands |
| Schedule | Optimal staff schedule output |
| KPIs | Revenue, utilization, labor cost metrics |
| Actuals | Historical demand data |
| Sensitivity | What-if analysis on demand assumptions |
| Capacity | Slide capacity and throughput model |
| Dashboard | Summary for leadership review |

---

## Results

| Metric | Value |
|--------|-------|
| Forecast R² | **0.91** |
| MAPE | **~12%** |
| Theoretical overstaffing reduction | **~18%** |
| Delivery format | Python scripts + Excel workbook |

---

## Tech stack

| Layer | Tech |
|-------|------|
| ML | scikit-learn (RandomForest, GradientBoosting) |
| Optimization | PuLP (Linear Programming) |
| Analysis | Python, Pandas, NumPy |
| Reporting | Excel (openpyxl), Matplotlib, Seaborn |
| Platform | ROLLER venue management API |

---

## Quickstart

```bash
git clone https://github.com/MuhammadFarid1990/SlidePark-AI
cd SlidePark-AI
pip install -r requirements.txt

# Run the forecasting pipeline
python forecasting/train.py

# Run the scheduling optimizer
python scheduling/optimize.py --forecast forecasting/output/forecast.csv
```

---

## Project structure

```
SlidePark-AI/
├── forecasting/
│   ├── train.py           # Model training + validation
│   ├── features.py        # Feature engineering pipeline
│   ├── predict.py         # Inference on new dates
│   └── output/            # Forecast CSVs
├── scheduling/
│   ├── optimize.py        # LP scheduling optimizer
│   ├── constraints.py     # Staffing constraints
│   └── output/            # Schedule outputs
├── data/
│   └── generate_synthetic.py   # Synthetic data generator
├── requirements.txt
└── README.md
```

---

## About the builder

**Muhammad Farid** — MS Business Analytics & AI @ UT Dallas.
Data Science & AI Automation Analyst intern @ Slick City Action Park.

[Portfolio](https://muhammadfarid1990.github.io) · [GitHub](https://github.com/MuhammadFarid1990)
