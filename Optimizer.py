
import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(page_title="Triphorium - 策略优化器", layout="wide")
st.title("Triphorium - 策略组合优化推荐引擎")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded_file = st.file_uploader("上传建筑数据（含 electricity_kwh 列）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # 默认参数
    avg_kwh = df['electricity_kwh'].mean()
    annual_kwh = avg_kwh * 12
    electricity_price = 0.12
    budget = st.slider("设置节能预算 ($)", min_value=5000, max_value=50000, value=30000, step=5000)

    # 策略池
    strategies = [
        {"name": "高效 HVAC 替换", "savings_pct": 0.15, "cost": 30000},
        {"name": "智能控制系统", "savings_pct": 0.10, "cost": 10000},
        {"name": "夜间负载转移", "savings_pct": 0.05, "cost": 0},
        {"name": "冷却水系统优化", "savings_pct": 0.08, "cost": 15000},
        {"name": "窗户遮阳与隔热膜", "savings_pct": 0.03, "cost": 5000},
        {"name": "照明系统升级", "savings_pct": 0.04, "cost": 8000}
    ]

    # ROI 排序
    def calculate_roi(s):
        savings_kwh = annual_kwh * s["savings_pct"]
        savings_dollars = savings_kwh * electricity_price
        roi = (savings_dollars - s["cost"]) / s["cost"] if s["cost"] > 0 else float("inf")
        s["roi"] = roi
        s["savings_dollars"] = savings_dollars
        return s

    ranked = sorted([calculate_roi(s) for s in strategies], key=lambda x: -x["roi"])

    # 贪婪选择
    selected = []
    total_cost = 0
    total_savings = 0

    for s in ranked:
        if total_cost + s["cost"] <= budget:
            selected.append({
                "策略": s["name"],
                "年节能收益 ($)": round(s["savings_dollars"], 2),
                "成本 ($)": s["cost"],
                "ROI": f"{s['roi']*100:.1f}%" if s['roi'] != float("inf") else "无限"
            })
            total_cost += s["cost"]
            total_savings += s["savings_dollars"]

    st.subheader("推荐策略组合（贪婪算法）")
    df_selected = pd.DataFrame(selected)
    st.dataframe(df_selected)

    # GPT 分析
    if api_key:
        strategy_names = ", ".join(df_selected['策略'].tolist())
        prompt = f"""
你是一个节能策略优化顾问。用户当前建筑年电量为 {annual_kwh:.0f} kWh，电价 $0.12/kWh，节能预算为 ${budget}。
推荐的策略组合包括：{strategy_names}，总年节省约 ${total_savings:.2f}。

请输出以下内容：
1. 为何这组策略性价比高？
2. 有无其他建议或替代组合？
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是建筑节能策略优化专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=350
            )
            message = response['choices'][0]['message']['content']
            st.subheader("GPT 节能策略分析建议")
            st.markdown(message)
        except Exception as e:
            st.error(f"GPT 生成失败：{e}")
else:
    st.info("请上传包含 electricity_kwh 的数据以开始优化。")
