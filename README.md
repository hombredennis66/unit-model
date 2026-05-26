# Student Spending Prediction — ML Project / 学生支出预测 — 机器学习项目

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

This project provides a complete walkthrough of building a machine learning system to predict student spending. It covers data generation, feature engineering, model training, and dynamic visualization.

### Project Overview
- **Objective:** Predict total semester spending based on demographic, academic, and behavioral features.
- **Dataset:** 250 student records with 23 features.
- **Models Used:**
  - Linear Regression
  - Ridge Regression (L2 Regularization)
  - Lasso Regression (L1 Regularization)
  - Random Forest Regressor (for Feature Importance)
- **Frontend:** Dynamic dashboard built with HTML5, CSS, and Chart.js.

### Key Features
- **Financial:** Monthly Allowance.
- **Lifestyle:** Distance from campus, Accommodation type, Transport type, Meal habits.
- **Behavioral:** Outings per month, Gaming hours, Club events, Mobile data usage.
- **Academic:** Year of study, Printing frequency.

### Setup and Usage
0. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd student-spending-prediction
   ```
1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. **Generate Dataset:**
   ```bash
   python3 generate_data.py
   ```
3. **Train Models:**
   ```bash
   python3 model_training.py
   ```
   This generates `results.json` which contains performance metrics and chart data.
4. **View Dashboard:**
   Open `index.html` in any modern web browser.

---

<a name="中文"></a>
## 中文

本项目提供了一个完整的机器学习系统构建演练，用于预测学生支出。内容涵盖数据生成、特征工程、模型训练及动态可视化。

### 项目概览
- **目标:** 根据人口统计、学术和行为特征预测学期总支出。
- **数据集:** 250 条学生记录，包含 23 个特征。
- **使用模型:**
  - 线性回归 (Linear Regression)
  - 岭回归 (Ridge Regression - L2 正则化)
  - Lasso 回归 (Lasso Regression - L1 正则化)
  - 随机森林 (Random Forest - 用于特征重要性分析)
- **前端:** 使用 HTML5, CSS 和 Chart.js 构建的动态仪表板。

### 核心特征
- **财务:** 每月津贴。
- **生活方式:** 离校距离、住宿类型、交通方式、饮食习惯。
- **行为:** 每月外出次数、游戏时长、社团活动、移动数据使用量。
- **学术:** 就读年级、打印频率。

### 安装与运行
0. **克隆仓库:**
   ```bash
   git clone <repository-url>
   cd student-spending-prediction
   ```
1. **安装依赖:**
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. **生成数据:**
   ```bash
   python3 generate_data.py
   ```
3. **模型训练:**
   ```bash
   python3 model_training.py
   ```
   该脚本会生成包含模型指标和图表数据的 `results.json`。
4. **查看仪表板:**
   在浏览器中打开 `index.html`。
