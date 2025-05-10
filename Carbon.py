
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import seaborn as sns
import openai
import os

st.set_page_config(page_title="Triphorium - 碳排趋势与异常分析", layout="wide")
st.title("Triphorium - CO₂ 排放预测与对比分析")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded_file = st.file_uploader("上传建筑碳排数据（含 timestamp, co2_tons）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # 月度汇总
    df_month = df.set_index('timestamp').resample('M')['co2_tons'].mean().dropna()

    # ARIMA 趋势预测
    model = ARIMA(df_month, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)

    st.subheader("CO₂ 趋势预测（未来6个月）")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_month.index, df_month.values, label="历史")
    future_index = pd.date_range(df_month.index[-1], periods=7, freq='M')[1:]
    ax1.plot(future_index, forecast, label="预测", linestyle='--')
    ax1.set_ylabel("CO₂ 排放（吨）")
    ax1.set_title("CO₂ 月度排放趋势")
    ax1.legend()
    st.pyplot(fig1)

    # 异常检测（Z-score）
    st.subheader("异常月份识别（Z-score > 2）")
    df_month_z = (df_month - df_month.mean()) / df_month.std()
    outliers = df_month_z[df_month_z.abs() > 2]
    if not outliers.empty:
        st.dataframe(outliers.rename("Z-score").round(2))
    else:
        st.success("无显著异常排放月份。")

    # 多项目对比（可选）
    st.subheader("CO₂ 多项目排放对比（可选）")
    group_col = st.selectbox("选择项目分组列（可为空）", [""] + list(df.columns))
    if group_col and group_col in df.columns:
        df_group = df.copy()
        df_group['month'] = df_group['timestamp'].dt.to_period('M')
        df_summary = df_group.groupby(['month', group_col])['co2_tons'].mean().reset_index()
        df_summary['month'] = df_summary['month'].astype(str)
        fig2 = plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_summary, x='month', y='co2_tons', hue=group_col, marker='o')
        plt.xticks(rotation=45)
        plt.ylabel("CO₂ 排放（吨）")
        plt.title("不同项目碳排对比")
        plt.tight_layout()
        st.pyplot(fig2)

    # GPT 分析
    if api_key:
        avg_recent = df_month[-6:].mean()
        avg_all = df_month.mean()
        trend_desc = "上升" if avg_recent > avg_all * 1.05 else "下降" if avg_recent < avg_all * 0.95 else "平稳"

        prompt = f"""
你是一个绿色建筑碳排分析专家。该建筑的月度 CO₂ 排放趋势最近 6 个月为：{trend_desc}趋势，
平均排放约 {avg_recent:.2f} 吨/月。历史平均为 {avg_all:.2f} 吨/月。
请简要分析：
1. 可能的变化原因；
2. 有哪些改进方向；
3. 若用于 ESG 报告，应如何表述。
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是碳排与能源分析专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=350
            )
            msg = response['choices'][0]['message']['content']
            st.subheader("GPT 趋势解读与 ESG 建议")
            st.markdown(msg)
        except Exception as e:
            st.error(f"GPT 生成失败：{e}")
else:
    st.info("请上传数据后查看分析结果。")
