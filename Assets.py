
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os

st.set_page_config(page_title="Triphorium - 多建筑能源对比", layout="wide")
st.title("Triphorium - 多项目能耗与碳排分析")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded = st.file_uploader("上传包含多个项目的能源数据（字段含：project, month, electricity_kwh, co2_tons）", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df['month'] = pd.to_datetime(df['month']).dt.strftime('%Y-%m')
    projects = df['project'].unique().tolist()

    st.subheader("月度用电趋势对比")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for proj in projects:
        subset = df[df['project'] == proj]
        ax1.plot(subset['month'], subset['electricity_kwh'], marker='o', label=proj)
    ax1.set_ylabel("月用电量 (kWh)")
    ax1.set_xlabel("月份")
    ax1.set_title("电力使用趋势对比")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("月度碳排放趋势对比")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for proj in projects:
        subset = df[df['project'] == proj]
        ax2.plot(subset['month'], subset['co2_tons'], marker='o', label=proj)
    ax2.set_ylabel("碳排放（吨）")
    ax2.set_xlabel("月份")
    ax2.set_title("碳排放趋势对比")
    ax2.legend()
    st.pyplot(fig2)

    # GPT ESG 汇总分析
    if api_key:
        summary = df.groupby('project')[['electricity_kwh', 'co2_tons']].mean().round(1)
        summary_str = summary.to_markdown()

        prompt = f"""
你是一个 ESG 报告分析顾问。以下是多个建筑项目的平均月用电与碳排放数据：

{summary_str}

请为 Triphorium 提供以下内容：
1. 哪些项目最值得优先节能改造，为什么？
2. 哪些项目已较为绿色？
3. 建议如何写入 ESG 报告表述该集群的表现与改进方向？
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是 ESG 建筑分析专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=350
            )
            msg = response['choices'][0]['message']['content']
            st.subheader("GPT ESG 绩效分析与建议")
            st.markdown(msg)
        except Exception as e:
            st.error(f"GPT 生成失败：{e}")
else:
    st.info("请上传数据以查看多项目对比图与建议。")
