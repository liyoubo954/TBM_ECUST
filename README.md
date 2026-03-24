# TBM 风险评估与掘进参数关联分析系统

## 1. 项目概述

本项目是为隧道掘进机（TBM）开发的后端风险评估与数据分析系统。它基于 Flask 框架构建，提供了一系列 RESTful API，用于实时评估掘进过程中的多种潜在风险，并支持历史掘进参数的查询与关联分析。

核心功能包括：
- **四大核心风险评估**：实时计算结泥饼、滞排、主驱动密封失效和盾尾密封失效的风险等级。
- **历史数据查询**：支持按时间范围和参数组合查询历史掘进数据。
- **参数关联分析**：提供掘进参数之间的关联性分析，辅助操作人员优化掘进策略。
- **动态风险记录**：记录并查询所有已触发的风险事件，便于复盘和追溯。

系统通过连接 InfluxDB（时间序列数据库）和 MySQL（关系型数据库）来获取实时数据和配置信息，并利用机器学习模型进行风险概率计算。

## 2. 技术栈

- **Web 框架**: Flask
- **数据库**: 
  - InfluxDB (存储实时掘进参数)
  - MySQL (存储风险记录、参数配置等)
- **核心库**: 
  - `pandas`: 数据处理与分析
  - `scikit-learn`, `tensorflow`: 机器学习模型
  - `numpy`: 数值计算
  - `pymysql`, `influxdb`: 数据库连接

## 3. API 接口文档

以下是本系统提供的核心 API 接口说明。

### 3.1. 风险评估接口

#### `POST /getRiskLevel`

根据指定环号，评估该环掘进过程中的所有核心风险。

- **请求体 (JSON)**:
  ```json
  {
    "RING": "1234"
  }
  ```
- **响应 (JSON)**:
  返回一个包含四大风险详细评估结果的对象，每个风险对象包含风险等级、风险评分、成因分析、处置措施、预警时间、关键参数等信息。

#### `GET /getLatestRiskLevel`

获取最新一环的综合风险评估结果。

- **请求参数 (可选)**:
  - `risk_type` (string): 指定风险类型（如 `结泥饼`），如果提供，则只返回该风险的最新评估结果。
- **响应 (JSON)**:
  - 如果不带参数，返回包含四大风险最新评估结果的对象。
  - 如果带 `risk_type` 参数，返回指定风险的详细评估结果。

#### `GET /getLatestRiskLevelSimple`

获取所有风险类型的最新风险等级和对应环号的简报。

- **响应 (JSON)**:
  ```json
  {
    "结泥饼": {"risk_level": "无风险Ⅰ", "ring": 1234},
    "滞排": {"risk_level": "低风险Ⅱ", "ring": 1234}
  }
  ```

### 3.2. 历史数据与记录

#### `POST /history/query`

查询指定时间范围和参数的历史掘进数据。

- **请求体 (JSON)**:
  ```json
  {
    "start_date": "2023-01-01",
    "end_date": "2023-01-02",
    "parameters": ["刀盘转速", "推进速度"]
  }
  ```
- **响应 (JSON)**:
  返回一个数据点列表，每个点包含时间戳和所查询参数的值与单位。

#### `GET /getAllRiskRecords`

获取所有已记录的风险事件（中风险及以上）。

- **请求参数 (可选)**:
  - `risk_type` (string): 按风险类型过滤。
  - `limit` (int): 返回记录的最大数量。
- **响应 (JSON)**:
  返回一个包含风险记录列表和各等级风险计数的对象。

### 3.3. 参数关联分析

#### `GET /correlation/systems`

获取所有可供分析的掘进机子系统列表。

#### `POST /correlation/system_params`

获取指定子系统下的所有掘进参数。

- **请求体 (JSON)**:
  ```json
  {
    "system": "驱动系统"
  }
  ```

#### `POST /correlation/top5`

查询与指定掘进参数最相关的 Top 5 参数。

- **请求体 (JSON)**:
  ```json
  {
    "parameter": "刀盘扭矩"
  }
  ```

## 4. 环境配置

系统通过环境变量进行配置，以下是关键的环境变量：

- `PROJECT_NAME`: 项目名称 (例如: "通苏嘉甬")
- `INFLUXDB_HOST`, `INFLUXDB_PORT`, `INFLUXDB_USERNAME`, `INFLUXDB_PASSWORD`, `INFLUXDB_DATABASE`: InfluxDB 连接信息。
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`: MySQL 连接信息。
- `CONSECUTIVE_TRIGGER_N`: 判断风险连续触发的次数阈值。

## 6. 项目文件结构

以下是项目核心文件的结构和功能说明：

```
. (TBM_ECUST)
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    └── risk/
        ├── __init__.py
        ├── routes.py
        ├── models/
        │   └── ... (机器学习模型文件)
        └── utils/
            ├── __init__.py
            ├── clog_risk.py
            ├── mdr_seal_risk.py
            ├── mud_cake_risk.py
            └── tail_seal_risk.py
```

- **`app.py`**: 项目的入口文件，用于创建并启动 Flask 应用。

- **`config.py`**: 定义了一个 `Config` 类，用于管理应用的所有配置，如数据库连接信息、密钥等。配置信息优先从环境变量中读取。

- **`requirements.txt`**: 列出了项目运行所需的所有 Python 依赖库及其版本。

- **`app/__init__.py`**: 应用工厂函数 `create_app` 的所在地。它负责初始化 Flask 应用、加载配置、并注册蓝图（Blueprint）。

- **`app/risk/__init__.py`**: `risk` 蓝图的初始化文件，使其成为一个可被注册的 Flask 蓝图。

- **`app/risk/routes.py`**: **核心业务逻辑文件**。定义了所有与风险评估和数据查询相关的 API 接口。它处理 HTTP 请求，调用风险计算模块，连接数据库，并返回 JSON 格式的响应。

- **`app/risk/utils/`**: 存放风险计算核心算法的目录。
  - **`clog_risk.py`**: 实现了计算**滞排风险**的算法。
  - **`mdr_seal_risk.py`**: 实现了计算**主驱动密封失效风险**的算法。
  - **`tail_seal_risk.py`**: 实现了计算**盾尾密封失效风险**的算法。
  - **`mud_cake_risk.py`**: 实现了计算**结泥饼风险**的算法，通常包含更复杂的逻辑，如调用机器学习模型。

- **`app/risk/models/`**: 存放用于结泥饼等风险评估的机器学习模型文件（如 `.h5`, `.pkl` 文件）。
