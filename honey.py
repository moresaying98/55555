import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的模型
model = joblib.load('ET.pkl')

# 训练时的特征顺序（根据新的顺序）
model_feature_names = ['WC', 'Insulin', 'TG', 'PLT', 'Height.cm.', 'GLU', 'WBC', 'TC', 'RBC', 'HDL']

# 特征范围定义（根据提供的特征范围和数据类型，已按新的顺序调整）
feature_ranges = {
    "WC": {"type": "numerical", "min": 50, "max": 170, "default": 80},
    "Insulin": {"type": "numerical", "min": 0.5, "max": 250, "default": 10},
    "TG": {"type": "numerical", "min": 10, "max": 400, "default": 50},
    "PLT": {"type": "numerical", "min": 150, "max": 450, "default": 250},  # 补充PLT范围
    "Height.cm.": {"type": "numerical", "min": 130, "max": 210, "default": 150},
    "GLU": {"type": "numerical", "min": 50, "max": 550, "default": 150},
    "WBC": {"type": "numerical", "min": 0, "max": 20, "default": 5},
    "TC": {"type": "numerical", "min": 100, "max": 300, "default": 200},  # 补充TC范围
    "RBC": {"type": "numerical", "min": 3.5, "max": 7.5, "default": 5.5},
    "HDL": {"type": "numerical", "min": 20, "max": 120, "default": 50},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of NAFLD is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
