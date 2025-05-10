# Triphorium Energy Intelligence Platform

这是一个智能建筑与数据中心通用的能耗分析平台，集成以下模块：
- ⚡ 用电趋势预测（Prophet）
- 🔍 异常检测（Isolation Forest）
- 🧠 等级预测与 GPT 节能建议（Random Forest + GPT-4）

## 本地运行方式
```bash
pip install -r requirements.txt
streamlit run Home.py
```

## GPT 使用说明
1. 注册 OpenAI 账号并获取 API Key；
2. 在页面中输入你的 Key；
3. 系统将自动调用 GPT-4 生成建议，并估算使用成本。
