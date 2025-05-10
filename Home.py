import streamlit as st

st.set_page_config(page_title="Triphorium AI 能耗平台", layout="wide")
st.title("Triphorium 智能建筑平台")
st.markdown("""
欢迎使用 Triphorium 多功能能耗分析平台！

左侧导航可进入：
- **能耗预测**（Forecast）：使用 Prophet 预测未来 90 天用电趋势；
- **异常检测**（Anomaly）：识别突发性能耗异常，辅助运维排查；
- **等级预测与建议**（Classification）：预测能耗等级，并由 GPT 提供节能建议。
""")
