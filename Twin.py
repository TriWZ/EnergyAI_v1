
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import os

st.set_page_config(page_title="Triphorium - 建筑数字孪生模拟器", layout="wide")
st.title("Triphorium - 数字孪生与节能策略模拟")

api_key = st.text_input("请输入 OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# 用户输入建筑参数
st.subheader("输入建筑参数")
name = st.text_input("建筑名称", value="8 Hedges Ave")
area = st.number_input("建筑面积（m²）", min_value=50, max_value=100000, value=3500)
base_kwh = st.number_input("基准每日电力消耗（kWh）", value=1200)

# 各系统占比输入
st.subheader("用能系统构成（总和应为100%）")
hvac = st.slider("HVAC（空调）", 0, 100, 45)
lighting = st.slider("Lighting（照明）", 0, 100, 20)
plug = st.slider("Plug Load（插座）", 0, 100, 15)
elevator = st.slider("Elevator（电梯）", 0, 100, 10)
other = 100 - (hvac + lighting + plug + elevator)
systems = {"HVAC": hvac/100, "Lighting": lighting/100, "Plug": plug/100, "Elevator": elevator/100, "Other": other/100}

# 策略模拟
st.subheader("节能策略模拟参数")
efficiency = st.slider("整体节能系数（越小越节能）", 0.5, 1.0, 0.85, step=0.01)

def simulate_energy(base, days=30, efficiency=1.0, noise_std=50):
    np.random.seed(42)
    base_daily = base * efficiency
    values = base_daily + np.random.normal(0, noise_std, size=days)
    return pd.Series(np.clip(values, 0, None), index=pd.date_range('2024-01-01', periods=days), name='kWh')

original = simulate_energy(base_kwh, efficiency=1.0)
adjusted = simulate_energy(base_kwh, efficiency=efficiency)

# 可视化对比
st.subheader("未来30天能耗预测对比")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(original.index, original.values, label="原始能耗")
ax.plot(adjusted.index, adjusted.values, label="节能策略", linestyle='--')
ax.set_ylabel("电力使用（kWh）")
ax.set_title("数字孪生能耗预测模拟")
ax.legend()
st.pyplot(fig)

# GPT 分析
if api_key:
    saving = original.sum() - adjusted.sum()
    prompt = f"""
你是一个智能建筑节能模拟专家。当前建筑名称为 {name}，面积 {area} 平方米。
每日基准电力消耗约 {base_kwh} kWh，系统结构如下：
HVAC {hvac}%，Lighting {lighting}%，Plug Load {plug}%，Elevator {elevator}%，Other {round(other)}%。

当前实施了一个策略，将整体效率调整为 {efficiency:.2f}，未来30天节省约 {saving:.1f} kWh。
请输出以下内容：
1. 该策略节能潜力的意义；
2. 是否还有进一步可优化的系统建议；
3. 若该建筑用于申请绿色建筑认证（如LEED），如何表述策略价值。
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是智能建筑数字孪生专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=350
        )
        msg = response['choices'][0]['message']['content']
        st.subheader("GPT 节能策略解读")
        st.markdown(msg)
    except Exception as e:
        st.error(f"GPT 生成失败：{e}")
