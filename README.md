# WindSpeed-Prediction-ML-Project
 商学院机器学习课程大作业：多高度气象风速时间序列预测
 基于Hugging Face公开气象数据集，实现单步/多步风速预测，对比线性回归、LSTM、Transformer三类时序模型
 ## 项目介绍
 1. 数据来源：Hugging Face 三份风速数据集（10m/50m/100m高度传感器）
 2. 预测任务
 - 单步预测：8小时历史时序预测未来1小时风速（PRED_STEP=1）
 - 多步长预测：8小时历史时序预测未来16小时风速（修改PRED_STEP=16）
 3. 对比模型：Linear Regression、LSTM、Transformer
 4. 评价指标：MSE、RMSE、MAE、R²决定系数
 5. 项目输出：数据分布图、相关性热力图、预测拟合曲线、模型权重pth文件
 ## 环境依赖一键安装
 打开终端执行：
 ```bash
 pip install pandas numpy matplotlib seaborn scikit-learn torch transformers datasets scipy