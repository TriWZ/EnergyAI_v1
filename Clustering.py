
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import openai
import os

st.set_page_config(page_title="Triphorium - 用能聚类分析", layout="wide")
st.title("Triphorium - 用能行为聚类分析 + GPT 解读")

# 设置 OpenAI Key
api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded_file = st.file_uploader("上传数据文件（包含 timestamp, electricity_kwh, gas_m3, water_tons, co2_tons）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1

    # 特征提取
    features = df[['electricity_kwh', 'water_tons', 'gas_m3', 'co2_tons', 'month', 'season']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 聚类中心特征
    cluster_avg = df.groupby('cluster')[['electricity_kwh', 'water_tons', 'gas_m3', 'co2_tons', 'month', 'season']].mean().round(1)
    st.subheader("各聚类中心平均特征")
    st.dataframe(cluster_avg)

    # 可视化 PCA 降维结果
    st.subheader("用能模式聚类图")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['pc1'] = components[:, 0]
    df['pc2'] = components[:, 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster_id]
        ax.scatter(subset['pc1'], subset['pc2'], label=f'Cluster {cluster_id}', s=60)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Energy Profile Clustering")
    ax.legend()
    st.pyplot(fig)

    # GPT 解释每个聚类类别
    if api_key:
        st.subheader("GPT 分析各类聚类行为及节能策略建议")
        for cluster_id, row in cluster_avg.iterrows():
            prompt = f"""
你是一个智能建筑节能专家。以下是某一类用能行为的平均特征：
- 电力使用：{row['electricity_kwh']} kWh
- 水用量：{row['water_tons']} 吨
- 气用量：{row['gas_m3']} 立方米
- CO2 排放：{row['co2_tons']} 吨
- 月份平均值：{row['month']}，季节：{row['season']}

请回答：
1. 该类行为可能代表什么类型的用能特征（例如夏季高峰？冬季加热？）？
2. 有哪些节能优化策略可以建议？
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是建筑节能顾问"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=300
                )
                message = response['choices'][0]['message']['content']
                st.markdown(f"### 聚类 {cluster_id}")
                st.markdown(message)
            except Exception as e:
                st.error(f"GPT 生成失败：{e}")
    else:
        st.warning("请提供 OpenAI API Key 以启用 GPT 建议。")
else:
    st.info("请上传数据进行聚类分析。")
