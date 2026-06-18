import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration Setup
st.set_page_config(page_title="Enterprise AI HR Dashboard", page_icon="🏢", layout="wide")

# 2. Cached Intelligence Core Loader
@st.cache_resource
def load_components():
    # Logic to handle both Localhost and Cloud paths automatically
    if os.path.exists("Model"):
        # Cloud/Root deployment scenario
        base_path = "Model"
    else:
        # Localhost/VS Code scenario
        base_path = os.path.join("..", "Model")
    
    model_path = os.path.join(base_path, "rf_model.pkl")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    features_path = os.path.join(base_path, "features.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
        
    return model, scaler, features

try:
    model, scaler, model_features = load_components()
except FileNotFoundError:
    st.error("Critical Error: Core ML pipeline files are missing in the 'Model' directory.")
    st.stop()

# 3. Component: Gauge Chart for Single Assessment
def create_gauge_chart(probability):
    prob_percent = probability * 100
    level_color = "#32CD32" if prob_percent < 30 else ("#FFA500" if prob_percent < 70 else "#DC143C")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Attrition Probability (%)", 'font': {'size': 16}},
        number = {'font': {'color': level_color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': '#e2fbe2'},
                {'range': [30, 70], 'color': "#090705"},
                {'range': [70, 100], 'color': '#fde8eb'}
            ],
            'threshold': {'line': {'color': level_color, 'width': 4}, 'value': prob_percent}
        }))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# 4. Dashboard Branding Header
st.title("📊 Enterprise AI - Human Capital Predictive Suite")
st.markdown("""
<div style="background-color: #1e1e24; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 25px; color: #ffffff;">
<strong>Predictive Analytics Engine Operational:</strong> Machine Learning algorithms are deployed to identify structural attrition risks. Switch workspace modules to screen individual profiles or run batch operations.
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["👤 Single Profile Evaluation", "📂 Batch Analytics Module (CSV Upload)"])

# ==========================================
# WORKSPACE MODULE 1: SINGLE EMPLOYEE CORE
# ==========================================
with tab1:
    st.subheader("Individual Risk Assessment Parameters")
    
    st.sidebar.header("👤 Employee Profile Parameters")
    age = st.sidebar.slider("Age Domain", 18, 60, 32)
    monthly_income = st.sidebar.number_input("Monthly Salary (USD Equivalent)", 1000, 20000, 6200)
    num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 10, 2)
    years_at_company = st.sidebar.slider("Tenure at Current Firm", 0, 40, 4)
    years_since_last_promotion = st.sidebar.slider("Years Post Last Promotion", 0, 15, 2)

    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ Environmental & Subjective Vectors")
    env_satisfaction = st.sidebar.selectbox("Workplace Environment Rating (1-4)", [1, 2, 3, 4], index=3)
    job_involvement = st.sidebar.selectbox("Job Involvement Rating (1-4)", [1, 2, 3, 4], index=2)
    overtime = st.sidebar.radio("System Logs OverTime Regularity?", ["Yes", "No"], index=1)
    business_travel = st.sidebar.selectbox("Corporate Travel Frequencies", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    # Developer Information Section (Stylish UI)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background-color: #2b2b36; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;">
        <h4 style="color: #ffffff; margin-top: 0px; margin-bottom: 10px;">👨‍💻 Developed By</h4>
        <div style="color: #e0e0e0; font-size: 14px; margin-bottom: 8px;">
            <strong>👤 Name:</strong><br>
            <span style="color: #00a8ff; font-weight: 900; font-size: 18px; letter-spacing: 1px;">VISHAL KASHYAP</span>
        </div>
        <div style="color: #e0e0e0; font-size: 13px; margin-bottom: 15px;">
            <strong>🎓 College:</strong><br>
            <i>United College of Engineering and Research</i>
        </div>
        <a href="https://www.linkedin.com/in/vishal-kashyap-10a16b395?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" style="text-decoration: none;">
            <div style="background-color: #0077b5; color: white; text-align: center; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 14px;">
                🔗 Connect on LinkedIn
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🔮 RUN PREDICTIVE CORRELATION", use_container_width=True):
        # Matching exactly the 9 inputs trained in model_training.py
        input_dict = {
            'Age': [age],
            'MonthlyIncome': [monthly_income],
            'NumCompaniesWorked': [num_companies_worked],
            'YearsAtCompany': [years_at_company],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'EnvironmentSatisfaction': [env_satisfaction],
            'JobInvolvement': [job_involvement],
            'OverTime': [overtime],
            'BusinessTravel': [business_travel]
        }
        
        raw_input_df = pd.DataFrame(input_dict)
        encoded_input = pd.get_dummies(raw_input_df)
        
        # Match exactly with the Model's memory (47 columns)
        aligned_input = pd.DataFrame(0, index=[0], columns=model_features)
        for col in model_features:
            if col in encoded_input.columns:
                aligned_input[col] = encoded_input[col].astype(int)
                
        scaled_input = scaler.transform(aligned_input)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]
        
        col_m, col_c = st.columns([1.3, 1])
        with col_m:
            if prediction[0] == 1:
                st.error("### ⚠️ Attrition Risk Alert Flagged")
                st.write("The internal neural path logic categorizes this user profile within the high-turnover boundary classification.")
            else:
                st.success("### ✅ Low Risk Retention Profile")
                st.write("Operational thresholds are balanced. High probability of continued structural retention.")
            st.metric(label="Precise Mathematical Risk Metric", value=f"{probability * 100:.2f}%")
        with col_c:
            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

# ==========================================
# WORKSPACE MODULE 2: BATCH DATA ENGINE WITH REAL GRAPH ANALYTICS
# ==========================================
with tab2:
    st.subheader("Dynamic Enterprise File Screener")
    st.write("Drop active HR tabular matrices (CSV) below to dynamically extract predictive trends and visualize real-time statistical breakdowns.")
    
    uploaded_file = st.file_uploader("Upload Target Document (CSV Format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            raw_batch = pd.read_csv(uploaded_file)
            st.info(f"Data ingestion successful. Found {raw_batch.shape[0]} active structural rows.")
            
            if st.button("🚀 INITIATE DISTRIBUTED BATCH PROCESS", use_container_width=True):
                
                # Encode the entire uploaded file just like Kaggle
                encoded_batch = pd.get_dummies(raw_batch)
                
                # Match exactly with the Model's memory
                aligned_matrix = pd.DataFrame(0, index=range(len(raw_batch)), columns=model_features)
                for col in model_features:
                    if col in encoded_batch.columns:
                        aligned_matrix[col] = encoded_batch[col].astype(int)
                
                # Scale and Predict
                scaled_matrix = scaler.transform(aligned_matrix)
                predictions = model.predict(scaled_matrix)
                probabilities = model.predict_proba(scaled_matrix)[:, 1]
                
                # Engineering output
                compiled_results = raw_batch.copy()
                compiled_results['Risk Probability (%)'] = (probabilities * 100).round(2)
                compiled_results['AI Decision Metric'] = ['⚠️ High Risk' if p >= 0.50 else '✅ Low Risk' for p in probabilities]
                compiled_results = compiled_results.sort_values(by='Risk Probability (%)', ascending=False)
                
                st.divider()
                st.subheader("📊 Dynamic Executive Analytics Dashboard (100% Real-Time)")
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("##### Attrition Risk Share Classification")
                    fig_pie = px.pie(compiled_results, names='AI Decision Metric', 
                                     color='AI Decision Metric',
                                     color_discrete_map={'⚠️ High Risk': '#dc3545', '✅ Low Risk': '#28a745'},
                                     hole=0.4)
                    fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_g2:
                    if 'OverTime' in compiled_results.columns:
                        st.markdown("##### Risk Impact Correlation: OverTime vs Avg Risk Probability")
                        fig_bar = px.bar(compiled_results.groupby('OverTime')['Risk Probability (%)'].mean().reset_index(),
                                         x='OverTime', y='Risk Probability (%)',
                                         labels={'Risk Probability (%)': 'Avg Risk Score (%)'},
                                         color='OverTime', color_discrete_sequence=['#ffc107', '#17a2b8'])
                        fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300, showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.write("OverTime data not available for bar chart.")
                
                st.divider()
                st.subheader("🎯 Computed Risk Evaluation Table")
                
                high_risk_total = int((predictions == 1).sum())
                st.metric(label="Total Profiles Flagged for Immediate Structural Retention Isolation", value=f"{high_risk_total} / {len(raw_batch)}")
                
                render_cols = ['EmployeeNumber', 'Age', 'MonthlyIncome', 'OverTime', 'Risk Probability (%)', 'AI Decision Metric']
                available_render = [c for c in render_cols if c in compiled_results.columns]
                st.dataframe(compiled_results[available_render] if available_render else compiled_results, use_container_width=True)
                
                csv_stream = compiled_results.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 DOWNLOAD EXECUTIVE SYSTEM RETENTION REPORT (CSV)", data=csv_stream, file_name="Enterprise_AI_Risk_Matrix_Report.csv", mime="text/csv", use_container_width=True)
                
        except Exception as error_exception:
            st.error(f"Operational stream interruption: {error_exception}")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Developed for Capstone Project | Machine Learning Random Forest Core Architecture | Production Stable v2.2")