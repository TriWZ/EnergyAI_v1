
import streamlit as st
import pandas as pd
import os
import openai

st.set_page_config(page_title="Triphorium AI Hub", layout="wide")
st.title("Triphorium 智能能效总控平台（AI Hub）")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# 检查是否有上传过数据
uploaded_data = st.file_uploader("上传建筑或集群数据（可用于多个模块）", type="csv", key="hub_upload")

if uploaded_data:
    df = pd.read_csv(uploaded_data)
    st.success(f"成功加载数据，共 {df.shape[0]} 行。")

    # 展示部分数据
    with st.expander("预览上传数据"):
        st.dataframe(df.head())

    # 简易 KPI 显示
    if 'electricity_kwh' in df.columns and 'co2_tons' in df.columns:
        kwh_total = df['electricity_kwh'].sum()
        co2_total = df['co2_tons'].sum()
        st.metric("累计电力使用 (kWh)", f"{kwh_total:,.0f}")
        st.metric("累计碳排放 (吨)", f"{co2_total:,.1f}")
else:
    st.warning("未检测到数据，部分模块功能将受限。")

# 模块导航卡片
st.subheader("模块导航")
cols = st.columns(5)
modules = [
    ("Forecast", "用电趋势预测"),
    ("Anomaly", "异常检测"),
    ("Classification", "等级预测+建议"),
    ("Clustering", "聚类分析"),
    ("Strategy", "策略模拟与ROI"),
    ("Optimizer", "策略组合优化"),
    ("Controller", "AI控制模拟器"),
    ("Carbon", "碳排趋势分析"),
    ("Twin", "数字孪生模拟"),
    ("Assets", "集群资产图谱")
]

for i, (file, desc) in enumerate(modules):
    with cols[i % 5]:
        st.markdown(f"### [{file}](./{file})")
        st.caption(desc)

# GPT 总控建议
if api_key and uploaded_data is not None:
    try:
        col_str = ", ".join(df.columns[:8])
        prompt = f"""
你是一个建筑能源分析AI助手。用户上传了一份建筑数据（包含列：{col_str}）。
请基于这些数据，为平台运营人员提供建议：
1. 建议优先启动哪几个模块？（说明原因）
2. 如果要生成节能策略或ESG报告，建议使用哪些功能组合？
3. 有无提示或潜在问题值得注意？
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是Triphorium平台的能效AI顾问"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        msg = response['choices'][0]['message']['content']
        st.subheader("GPT AI 总体分析建议")
        st.markdown(msg)
    except Exception as e:
        st.error(f"GPT 生成失败：{e}")
