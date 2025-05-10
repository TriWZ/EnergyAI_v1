
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Triphorium - AI 控制模拟器", layout="wide")
st.title("Triphorium - Q-learning HVAC 控制策略模拟器")

st.markdown("本模块训练一个 AI 控制器，在 24 小时内自动学习如何最优控制室温与电力消耗。")

# 参数设定
initial_temp = st.slider("初始温度", min_value=18, max_value=30, value=26)
target_temp = 22.5
comfort_min, comfort_max = 21, 24
episodes = st.slider("训练回合数", 1000, 20000, 10000, step=1000)

# 状态空间
temp_range = range(-8, 9)
actions = [-1, 0, 1]
q_table = np.zeros((len(temp_range), 24, len(actions)))
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.2

# 奖励函数
def get_reward(temp, action):
    power_penalty = abs(action) * -1
    if comfort_min <= temp <= comfort_max:
        comfort_reward = 2
    else:
        comfort_reward = -abs(temp - target_temp)
    return comfort_reward + power_penalty

# 训练 Q 表
for _ in range(episodes):
    temp = initial_temp
    for h in range(24):
        temp_diff = int(round(temp - target_temp))
        temp_diff = np.clip(temp_diff, -8, 8)
        temp_idx = temp_diff + 8

        if np.random.rand() < exploration_rate:
            a_idx = np.random.randint(len(actions))
        else:
            a_idx = np.argmax(q_table[temp_idx, h])

        action = actions[a_idx]
        next_temp = temp + action + np.random.normal(0, 0.5)
        next_temp_diff = int(round(next_temp - target_temp))
        next_temp_diff = np.clip(next_temp_diff, -8, 8)
        next_temp_idx = next_temp_diff + 8

        reward = get_reward(temp, action)
        best_future_q = np.max(q_table[next_temp_idx, (h+1)%24])
        q_table[temp_idx, h, a_idx] += learning_rate * (reward + discount_factor * best_future_q - q_table[temp_idx, h, a_idx])
        temp = next_temp

# 策略图
policy = np.argmax(q_table, axis=2) - 1
fig, ax = plt.subplots(figsize=(12, 5))
cax = ax.imshow(policy, cmap='coolwarm', aspect='auto', extent=[0, 23, -8, 8])
ax.set_xlabel("小时")
ax.set_ylabel("温度差（当前 - 目标）")
ax.set_title("AI 控制策略热图（动作：-1=降温, 0=保持, +1=升温）")
fig.colorbar(cax, label="控制动作")
st.pyplot(fig)

st.markdown("该策略图展示了在不同温度差和时间下，AI 控制器学习的最优行为。")
