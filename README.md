# 韩国人口模拟器 (Korean Population Simulator)

## 项目简介
本项目受到[SOUTH KOREA IS OVER by Kurzgesagt - In A Nutshell](https://www.youtube.com/watch?v=Ufmu1WD2TSk)启发，旨在通过模拟韩国人口的动态变化，分析不同生育率、死亡率和迁移率情景下的人口趋势。项目包含一个通用人口模拟器和一个专门针对韩国人口结构的模拟器，支持多种情景设置和可视化功能。

## 功能特性
- **人口动态模拟**：支持生育、死亡和迁移的动态变化模拟。
- **多情景分析**：支持不同生育率和迁移率情景的设置与比较。
- **可视化**：提供人口金字塔、人口趋势图、抚养比变化等多种图表。
- **动画支持**：生成人口金字塔的动态变化动画。
- **数据导出**：支持将模拟结果导出为数据表或报告。

## 文件结构
```bash
BirthRateStimulation/ 
├── README.md # 项目说明文件 
├── south_korean_society.py # 韩国人口模拟器 
├── general_society.py # 通用人口模拟器 
└── population_report.md # 模拟报告（运行后生成）
```

## 安装与运行
### 环境要求
- Python 3.8+
- 必要的依赖库：
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `IPython`

### 安装依赖
运行以下命令安装所需依赖：
```bash
pip install numpy pandas matplotlib seaborn ipython
```

## 运行示例
运行韩国人口模拟器：
```bash
python south_korean_society.py
```

运行后将生成以下文件：
- 韩国2022年和2072年人口金字塔图
- 人口趋势图
- 不同情景下的人口比较图
- 人口预测报告 (population_report.md)

`KoreaPopulationSimulator` 提供了以下主要功能：
- 设置生育率场景
```python
simulator.set_fertility_scenario('current')  # 维持当前极低生育率
simulator.set_fertility_scenario('recovery')  # 恢复至更替水平
```
- 设置迁移率场景
```python
simulator.set_migration_scenario('increased')  # 增加迁移率
```
- 运行模拟
```python
simulator.simulate()
```
- 生成报告
```python
report = simulator.generate_report("基准情景：维持极低生育率")
with open("population_report.md", "w", encoding="utf-8") as file:
    file.write(report)
```