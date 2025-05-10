
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 设置页面
st.set_page_config(page_title="Triphorium - 能耗等级预测 + GPT 建议", layout="wide")
st.title("Triphorium - 多类能耗预测与 GPT 优化建议")

# 设置 OpenAI Key（环境变量或页面输入）
api_key = st.text_input("请输入你的 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# 文件上传
uploaded_file = st.file_uploader("上传数据文件（含 timestamp, electricity_kwh, water_tons, gas_m3, co2_tons）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # 构造特征
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['season'] = df['month'] % 12 // 3 + 1

    # 多类标签
    low_q = df['electricity_kwh'].quantile(0.5)
    high_q = df['electricity_kwh'].quantile(0.9)
    def label_row(val):
        if val <= low_q:
            return 0
        elif val <= high_q:
            return 1
        else:
            return 2
    df['level'] = df['electricity_kwh'].apply(label_row)

    # 模型训练
    X = df[['month', 'year', 'season', 'water_tons', 'gas_m3', 'co2_tons']]
    y = df['level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 预测并合并
    df['predicted'] = clf.predict(X)

    # 分类评估
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["低", "中", "高"], output_dict=True)
    st.subheader("模型评估报告")
    st.dataframe(pd.DataFrame(report).transpose())

    # 可视化
    st.subheader("每月用电趋势（颜色代表等级预测）")
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = df['predicted'].map({0: 'green', 1: 'orange', 2: 'red'})
    ax.bar(df['timestamp'], df['electricity_kwh'], color=colors)
    ax.set_ylabel("电力使用 (kWh)")
    ax.set_title("每月电力使用量")
    st.pyplot(fig)

    # GPT建议生成
    if api_key:
        high_df = df[df['predicted'] == 2]
        st.subheader("GPT 节能建议（高能耗月份）")
        total_tokens = 0
        gpt_cost = 0.0

        for _, row in high_df.iterrows():
            prompt = f"""
你是一个智能建筑能源顾问，以下是建筑某月的数据：

- 月份：{row['timestamp'].strftime('%Y-%m')}
- 用电：{row['electricity_kwh']} kWh
- 用水：{row['water_tons']} 吨
- 用气：{row['gas_m3']} 立方米
- CO₂ 排放：{row['co2_tons']} 吨
- 季节（1=春，2=夏，3=秋，4=冬）：{int(row['season'])}

请输出两部分内容：
1. 能耗异常可能原因；
2. 针对该月提出 1-2 条优化建议。
"""

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是建筑节能优化专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=300
                )
                message = response['choices'][0]['message']['content']
                usage = response['usage']
                total_tokens += usage['total_tokens']
                gpt_cost += total_tokens * 0.06 / 1000  # GPT-4 单价

                st.markdown(f"### {row['timestamp'].strftime('%Y-%m')}")
                st.markdown(message)

            except Exception as e:
                st.error(f"生成失败：{e}")

        st.success(f"共调用 GPT {len(high_df)} 次，估计总费用：${gpt_cost:.4f}")
    else:
        st.warning("未提供 OpenAI API Key，无法生成建议。")
else:
    st.info("请上传数据以进行预测。")
