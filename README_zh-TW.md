# Disease risk prediction model, deployed on NVIDIA Jetson Orin for real-time inference

## 專案概述

本專案是一套完整的機器學習系統，專為**血液透析病患的即時健康預測**而設計。系統整合三個預測模型，協助醫療人員監控及管理透析病患，並部署於 NVIDIA Jetson Orin 邊緣運算設備上進行即時推論。

### 主要功能

- **心衰竭風險預測** - 二元分類模型，識別高風險心衰竭病患
- **血紅素預測** - 迴歸模型，預測即時血紅素 (RTHGB) 數值以管理貧血
- **乾體重預測** - 迴歸模型，預估最佳乾體重以進行體液管理
- **即時儀表板** - 部署於邊緣設備的互動式視覺化介面，提供臨床決策支援
- **邊緣 AI 部署** - 針對 NVIDIA Jetson Orin 優化，使用 Docker 容器化部署

---

## 目錄

- [系統架構](#系統架構)
- [模型說明](#模型說明)
  - [心衰竭模型](#1-心衰竭預測模型)
  - [血紅素模型](#2-血紅素-rthgb-預測模型)
  - [乾體重模型](#3-乾體重預測模型)
- [儀表板](#儀表板)
- [安裝方式](#安裝方式)
- [使用方法](#使用方法)
- [部署方式](#部署方式)
- [專案結構](#專案結構)
- [技術細節](#技術細節)
- [成果展示](#成果展示)
- [授權條款](#授權條款)

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           醫療預測系統架構                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    醫院       │    │    資料      │    │    特徵       │                   │
│  │   資料庫      │───▶│   前處理      │───▶│    工程       │                   │
│  │   (私有)      │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                           │
│                    ┌────────────────────────────┼────────────────────────┐  │
│                    │                            ▼                        │  │
│                    │  ┌──────────────────────────────────────────────┐   │  │
│                    │  │            機器學習模型 (LightGBM)             │   │  │
│                    │  ├──────────────┬──────────────┬────────────────┤   │  │
│                    │  │   心衰竭     │    血紅素    │     乾體重     │  │   │  │
│                    │  │  (二元分類)  │   (迴歸)     │    (迴歸)      │  │   │  │
│                    │  │    AUC       │    RMSE      │     RMSE       │   │  │
│                    │  └──────────────┴──────────────┴────────────────┘   │  │
│                    │                            │                        │  │
│                    └────────────────────────────┼────────────────────────┘  │
│                                                 ▼                           │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │            即時儀表板 (Plotly Dash)                   │  │
│                    │  • 病患監控          • 風險視覺化                      │  │
│                    │  • 臨床指標          • 決策支援                        │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                                 │                           │
│                                                 ▼                           │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │           NVIDIA Jetson Orin (邊緣設備)              │  │
│                    │               Docker 容器化部署                       │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 模型說明

### 1. 心衰竭預測模型

**類型：** 二元分類
**演算法：** LightGBM（梯度提升決策樹）
**目標變數：** `HF_1`（1 = 高風險，0 = 低風險）

#### 目的
識別具有心衰竭風險的血液透析病患。早期偵測可實現主動介入治療，改善病患預後。

#### 關鍵技術
- SMOTE（合成少數過採樣技術）處理類別不平衡
- Yeo-Johnson 冪次轉換處理偏態特徵
- StandardScaler 標準化
- One-hot 編碼處理類別變數

#### 評估指標
| 指標 | 說明 |
|------|------|
| Accuracy（準確率）| 整體預測正確率 |
| Precision（精確率）| 真陽性 / 預測陽性 |
| Recall（召回率）| 真陽性 / 實際陽性 |
| F1-Score | 精確率與召回率的調和平均 |
| AUC-ROC | ROC 曲線下面積 |

#### 模型可解釋性
- 特徵重要性分析（split 和 gain）
- SHAP（SHapley Additive exPlanations）值解釋

---

### 2. 血紅素 (RTHGB) 預測模型

**類型：** 迴歸
**演算法：** LightGBM
**目標變數：** `RTHGB_mapDate`（即時血紅素數值）

#### 目的
預測即時血紅素濃度以支援透析病患的貧血管理。貧血是透析病患常見的併發症，需要仔細監控並使用 ESA（紅血球生成刺激劑）和鐵劑治療。

#### 臨床意義
- 血紅素監測是透析病患照護的關鍵
- 協助優化 ESA 和鐵劑治療劑量
- 預防血紅素過低或過高的併發症

#### 評估指標
| 指標 | 說明 |
|------|------|
| RMSE | 均方根誤差 |
| MSE | 均方誤差 |
| MAPE | 平均絕對百分比誤差 |

---

### 3. 乾體重預測模型

**類型：** 迴歸
**演算法：** LightGBM
**目標變數：** `DryWeight_Y`（目標乾體重，單位：公斤）

#### 目的
預估血液透析病患的最佳乾體重。乾體重是指透析後去除多餘體液時不會引起症狀的目標體重。

#### 臨床意義
- **體液過多：** 可能導致高血壓、心衰竭、肺水腫
- **體液不足：** 可能引起低血壓、抽筋、頭暈
- 準確的乾體重預估對體液管理至關重要

#### 目標標籤生成
乾體重目標使用穩定性演算法計算：
1. 追蹤連續透析療程的乾體重數值
2. 若乾體重連續 9 次以上療程保持穩定，則視為真實乾體重
3. 此穩定值作為預測目標

```python
# 穩定性判斷邏輯
stable_count = 9  # 所需連續穩定療程數
is_stable = rolling_window(dry_weight, stable_count).apply(
    lambda x: len(set(x)) == 1  # 所有數值相同
)
```

---

## 儀表板

### 即時視覺化儀表板

使用 **Plotly Dash** 建構的互動式網頁儀表板，用於即時病患監控和臨床決策支援。

### 儀表板截圖

#### 正常病患 - 低風險狀態
<p align="center">
  <img src="docs/images/dashboard_normal.png" alt="正常病患儀表板" width="900">
</p>

*低風險病患的儀表板顯示，生命徵象穩定且預測值正常。*

---

#### 心衰竭高風險警示
<p align="center">
  <img src="docs/images/dashboard_heart_failure_risk.png" alt="心衰竭風險儀表板" width="900">
</p>

*標示心衰竭風險升高的病患儀表板。系統顯示臨床介入建議。*

---

#### 透析中低血壓 (IDH) 風險警示
<p align="center">
  <img src="docs/images/dashboard_idh_risk.png" alt="低血壓風險儀表板" width="900">
</p>

*顯示透析中低血壓風險的病患儀表板。即時監控實現主動管理。*

---

#### 功能特色

| 元件 | 說明 |
|------|------|
| **病患選擇器** | 下拉選單選擇特定病患或自動輪播模式 |
| **風險指標** | 顏色標示的心衰竭和低血壓風險狀態（紅/綠） |
| **預測顯示** | 即時血紅素和乾體重預測值 |
| **儀表圖** | TSAT 和 FERRITIN 數值指示器 |
| **三角圖** | 血紅素/鐵劑/ESA 平衡視覺化 |
| **時間序列** | 6 個圖表追蹤透析療程參數 |
| **臨床建議** | 基於風險等級的 AI 生成指引 |

#### 儀表板參數

| 參數 | 圖表標題 |
|------|----------|
| A_TMP | 膜上壓 (TMP) |
| A_VENOUSPRESSURE | 靜脈壓 |
| A_ARTERIALPRESSURE | 動脈壓 |
| A_TOTALUF | 脫水量 |
| A_D_TEMPERATURE | 血液溫度 |
| A_BICARBONATEADJUSTMENT | 碳酸氫鹽調整 |

---

## 安裝方式

### 系統需求

- Python 3.10 或更高版本
- Docker（容器化部署）
- NVIDIA Jetson Orin（邊緣部署）

### 本機安裝

```bash
# 複製專案
git clone https://github.com/huang422/MedEdge-Predictor.git
cd MedEdge-Predictor

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝相依套件
pip install -r requirements.txt

# 或以套件形式安裝
pip install -e .
```

### Docker 安裝

```bash
# 建構 Docker 映像
./scripts/build_docker.sh

# 或手動建構
docker build -f docker/Dockerfile -t medical-prediction-dashboard .
```

---

## 使用方法

### 訓練模型

```python
from src.models import HeartFailureModel, HemoglobinModel, DryWeightModel
from src.data import DataPreprocessor, FeatureEngineer

# 初始化前處理器
preprocessor = DataPreprocessor()

# 資料前處理
X, y = preprocessor.preprocess(df, target_column="HF_1")

# 訓練心衰竭模型
hf_model = HeartFailureModel(use_smote=True)
hf_model.train(X_train, y_train, X_val, y_val)

# 評估
metrics = hf_model.evaluate(X_test, y_test)
print(metrics)

# 儲存模型
hf_model.save_model("weights/heart_failure_model.txt")
```

### 本機執行儀表板

```python
from src.dashboard import create_app, run_server

# 建立並執行儀表板
app = create_app(data_path="./data/Pred_all.csv")
run_server(app)

# 瀏覽 http://localhost:8050
```

### 使用 Docker

```bash
# 使用 Docker 執行
docker run -p 8050:8050 medical-prediction-dashboard

# 或使用 docker-compose
docker-compose -f docker/docker-compose.yml up

# 瀏覽 http://localhost:8050
```

---

## 部署方式

### NVIDIA Jetson Orin 部署

本系統針對 NVIDIA Jetson Orin 邊緣設備進行優化部署。

#### 快速部署

```bash
# 在開發機器上 - 建構並匯出
./scripts/deploy_jetson.sh build
./scripts/deploy_jetson.sh export

# 將 medical-prediction-dashboard.tar 傳輸至 Jetson 設備

# 在 Jetson Orin 上 - 載入並執行
./scripts/deploy_jetson.sh load
./scripts/deploy_jetson.sh start
```

#### 手動部署

```bash
# 建構 ARM64 映像
docker build --platform linux/arm64 \
    -f docker/Dockerfile.jetson \
    -t medical-prediction-dashboard .

# 匯出映像
docker save -o medical-prediction-dashboard.tar medical-prediction-dashboard

# 在 Jetson 設備上
sudo docker load -i medical-prediction-dashboard.tar
sudo docker run -d \
    --name medical-dashboard \
    --runtime nvidia \
    -p 8050:8050 \
    medical-prediction-dashboard
```

#### 部署指令參考

| 指令 | 說明 |
|------|------|
| `./scripts/deploy_jetson.sh build` | 建構 Docker 映像 |
| `./scripts/deploy_jetson.sh export` | 匯出為 tar 檔案 |
| `./scripts/deploy_jetson.sh load` | 在 Jetson 上載入映像 |
| `./scripts/deploy_jetson.sh start` | 啟動儀表板 |
| `./scripts/deploy_jetson.sh stop` | 停止儀表板 |
| `./scripts/deploy_jetson.sh status` | 查看狀態 |
| `./scripts/deploy_jetson.sh logs` | 查看日誌 |

---

## 專案結構

```
medical-prediction-system/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # 設定參數
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # 資料前處理
│   │   └── feature_engineering.py # 特徵工程
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # 基礎模型類別
│   │   ├── heart_failure.py     # 心衰竭模型
│   │   ├── hemoglobin.py        # 血紅素模型
│   │   └── dry_weight.py        # 乾體重模型
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py               # 儀表板應用程式
│   │   └── data_merger.py       # 預測資料合併
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # 評估指標
│       └── visualization.py     # 繪圖工具
├── docker/
│   ├── Dockerfile               # 標準 Docker 設定
│   ├── Dockerfile.jetson        # Jetson 優化設定
│   ├── docker-compose.yml       # 標準 compose
│   └── docker-compose.jetson.yml # Jetson compose
├── scripts/
│   ├── build_docker.sh          # 建構腳本
│   └── deploy_jetson.sh         # 部署腳本
├── docs/                        # 文件
│   └── images/                  # 截圖和圖表
├── data/                        # 資料目錄（已 gitignore）
├── weights/                     # 模型權重（已 gitignore）
├── requirements.txt             # Python 相依套件
├── .gitignore
├── README.md                    # 英文文件
└── README_zh-TW.md             # 本檔案（中文）
```

---

## 技術細節

### 資料前處理流程

1. **缺失值處理**
   - 移除缺失值超過 50% 的欄位
   - 使用 KNN 插補填補剩餘缺失值

2. **特徵轉換**
   - Yeo-Johnson 轉換處理偏態特徵（偏度 > 1）
   - StandardScaler 標準化數值特徵

3. **類別編碼**
   - One-hot 編碼處理警報狀態旗標
   - 欄位名稱特殊字元清理

4. **時間切分**
   - 訓練資料：12 月
   - 測試資料：11 月
   - 驗證集：訓練資料的 20%

### 模型設定

```python
# LightGBM 分類參數
lgbm_classification_params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "num_boost_round": 2000
}

# LightGBM 迴歸參數
lgbm_regression_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5
}
```

### 儀表板設定

| 設定項目 | 數值 |
|----------|------|
| 更新間隔 | 3 秒 |
| 時間序列窗口 | 10 個資料點 |
| 心衰竭風險閾值 | 0.5 |
| 低血壓風險閾值 | 30% |
| 伺服器埠號 | 8050 |

---

## 成果展示

### 模型效能摘要

| 模型 | 任務 | 主要指標 | 演算法 |
|------|------|----------|--------|
| 心衰竭 | 分類 | AUC-ROC | LightGBM |
| 血紅素 | 迴歸 | RMSE | LightGBM |
| 乾體重 | 迴歸 | RMSE | LightGBM |

### 關鍵發現

- **心衰竭模型：** 使用 SMOTE 處理類別不平衡，有效識別高風險病患
- **血紅素模型：** 準確預測實現主動貧血管理
- **乾體重模型：** 穩定性標籤生成提升目標品質

---

## 隱私與重構說明

**重要說明：** 本專案是與醫療機構合作期間使用醫院私有資料開發。基於**個人資料保護和資訊安全考量**，程式碼已**重構**以供公開作品集展示，同時保留所有核心功能和邏輯。

### 重構方式

| 面向 | 原始版本 | 重構版本 |
|------|----------|----------|
| **程式碼結構** | Jupyter notebooks + 腳本 | 模組化 Python 套件 |
| **資料庫連線** | 直接 PostgreSQL 查詢 | 設定檔形式（憑證已排除） |
| **病患資料** | 真實醫療紀錄 | 合成匿名資料 |
| **識別碼** | 實際 MEDICALID | 匿名 ID（A01, A02, ...） |

### 保留的內容

- **核心機器學習邏輯：** 所有模型訓練、前處理和評估程式碼維持與生產系統相同的功能
- **演算法參數：** LightGBM/XGBoost 設定與部署模型一致
- **特徵工程：** SMOTE、Yeo-Johnson 轉換、StandardScaler 維持不變
- **儀表板功能：** 所有視覺化元件、回調和布局功能等同
- **臨床邏輯：** 風險閾值、建議邏輯和醫療參數皆為真實

### 因安全考量修改的內容

- **資料庫憑證：** 移除所有連線字串和密碼
- **病患識別碼：** 替換為合成匿名 ID
- **醫療資料：** 範例資料使用真實但合成的數值
- **檔案路徑：** 從機構特定路徑通用化

### 給審閱者

此重構版本展示：
1. **技術能力：** 機器學習管線開發、儀表板建立、Docker 部署
2. **領域知識：** 醫療預測系統、血液透析參數
3. **工程實踐：** 模組化設計、設定管理、測試
4. **資安意識：** 妥善處理敏感醫療資料

原始生產程式碼（未包含）已部署於合作醫院的 NVIDIA Jetson Orin 設備上。

---

## 使用技術

| 類別 | 技術 |
|------|------|
| **機器學習框架** | LightGBM, XGBoost, scikit-learn |
| **資料處理** | Pandas, NumPy, SciPy |
| **視覺化** | Plotly, Matplotlib, Seaborn |
| **儀表板** | Plotly Dash |
| **部署** | Docker, NVIDIA Jetson |
| **可解釋性** | SHAP |

---

## 授權條款

本專案供教育和作品集展示用途。

---

## 聯絡方式
For questions, issues, or collaboration inquiries:

- Developer: Tom Huang
- Email: huang1473690@gmail.com

---

## 致謝

- 合作醫院提供臨床資料和專業知識
- NVIDIA 提供 Jetson Orin 邊緣運算平台
- 開源社群提供優秀的機器學習函式庫
