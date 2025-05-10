
import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(page_title="Triphorium - 节能策略模拟", layout="wide")
st.title("Triphorium - 节能策略模拟器与 ROI 分析")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded_file = st.file_uploader("上传数据文件（含 timestamp, electricity_kwh 等）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    avg_kwh = df['electricity_kwh'].mean()
    annual_kwh = avg_kwh * 12
    electricity_price = 0.12

    strategies = [
        {"name": "高效 HVAC 替换", "savings_pct": 0.15, "cost": 30000},
        {"name": "智能控制系统", "savings_pct": 0.10, "cost": 10000},
        {"name": "夜间负载转移", "savings_pct": 0.05, "cost": 0},
        {"name": "冷却水系统优化", "savings_pct": 0.08, "cost": 15000}
    ]

    selected = st.multiselect("请选择要模拟的节能策略", [s["name"] for s in strategies])

    if selected:
        results = []
        total_savings = 0
        total_cost = 0

        for s in strategies:
            if s['name'] in selected:
                savings_kwh = annual_kwh * s['savings_pct']
                savings_dollars = savings_kwh * electricity_price
                roi = (savings_dollars - s['cost']) / s['cost'] if s['cost'] > 0 else float('inf')
                payback = s['cost'] / savings_dollars if savings_dollars > 0 else float('inf')
                results.append({
                    "策略": s["name"],
                    "预估节能 (%)": f"{s['savings_pct']*100:.1f}%",
                    "年节省电费 ($)": round(savings_dollars, 2),
                    "一次性投入成本 ($)": s["cost"],
                    "投资回报率 (ROI)": f"{roi*100:.1f}%" if roi != float('inf') else "无限",
                    "回本周期 (年)": round(payback, 2) if payback != float('inf') else "无需投入"
                })
                total_savings += savings_dollars
                total_cost += s['cost']

        df_result = pd.DataFrame(results)
        st.subheader("策略模拟结果")
        st.dataframe(df_result)

        if api_key:
            strategy_list = ", ".join(selected)
            prompt = f"""
你是一个绿色建筑节能分析师。用户选择了以下策略组合：{strategy_list}。
每年预估节能约 ${total_savings:.2f}，总投入为 ${total_cost:.2f}。

请根据常识与建筑运营经验，输出：
1. 为什么这组策略有效；
2. 有无进一步优化建议（例如先后顺序、替代策略等）。
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是建筑节能顾问"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=350
                )
                msg = response['choices'][0]['message']['content']
                st.subheader("GPT 分析建议")
                st.markdown(msg)
            except Exception as e:
                st.error(f"GPT 生成失败：{e}")
    else:
        st.info("请选择策略以进行模拟分析。")
else:
    st.info("请上传能耗数据进行模拟。")
