from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit page configuration
st.set_page_config(
    page_title="Sleep Health and Lifestyle Analysis",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Dell\\Downloads\\Sleep_health_and_lifestyle_dataset.csv")

data = load_data()

# Sidebar options
st.sidebar.header("Select Analysis")
options = [
    "Introduction",
    "Dataset Overview",
    "Age Distribution and Sleep Duration",
    "Gender and Sleep Duration",
    "Stress Levels and Sleep Duration",
    "Physical Activity and Sleep Quality",
    "Sleep Disorders Prevalence",
    "Relationship Between BMI and Sleep Quality",
    "Sleep Quality Across Age Groups",
    "Correlation Between Sleep Duration and Sleep Quality",
    "Predict Sleep Disorder Using Random Forest"
]
selected_option = st.sidebar.radio("Choose an analysis:", options)

# Function to preprocess data and train the model
def predict_sleep_disorder():
    st.title("Sleep Disorder Prediction Using Random Forest")
    
    st.markdown("### Data Preprocessing")
    st.write("Converting categorical columns into numerical values.")
    
    # Data preprocessing
    df = data.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes
    
    st.write("**Preview of Processed Data:**")
    st.write(df.head())
    
    # Define features and target
    X = df.drop(["Sleep Disorder"], axis=1)
    y = df["Sleep Disorder"]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.markdown("### Model Results")
    st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
    
    # Display classification report
    st.markdown("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Feature importance
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    st.markdown("**Feature Importance:**")
    st.bar_chart(feature_importances.set_index("Feature"))

# Analysis sections
if selected_option == "Introduction":
    st.title("Introduction to the Sleep Health and Lifestyle Dataset")
    st.markdown(
        """
        <div style="text-align: center;">
            <span style="font-size: 50px;">🌙</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("Project Overview")
    st.markdown(
        """
        This project aims to analyze the **Sleep Health and Lifestyle Dataset**, which explores the relationships between lifestyle 
        factors, health conditions, and sleep patterns. By using this data, we aim to uncover actionable insights that can improve 
        individuals' sleep quality and overall well-being.
        """
    )
    # Additional content for the "Introduction" section...
    # Introduction to the Data
    st.header("Introduction to the Data")
    st.markdown(
        """
        The dataset used in this project contains information about individuals and their sleep habits, alongside lifestyle and 
        health factors. Key features of the dataset include:
        - **Age**: The age of the individual.
        - **Gender**: Gender identity of the individual.
        - **Sleep Duration**: The average number of hours an individual sleeps per night.
        - **Quality of Sleep**: A subjective rating of sleep quality.
        - **Stress Levels**: The level of stress experienced by the individual.
        - **Physical Activity**: The individual's level of physical activity.
        - **BMI (Body Mass Index)**: A measure of body fat based on height and weight.
        - **Sleep Disorders**: Whether the individual has been diagnosed with a sleep disorder.

        The data serves as the foundation for uncovering patterns and correlations between these variables and sleep health.
        """
    )
    
    # Insights and Usefulness
    st.header("Insights and Usefulness")
    st.markdown(
        """
        Through this analysis, we aim to answer several key questions:
        - How does **age** impact sleep duration and quality?
        - Are there any differences in sleep patterns based on **gender**?
        - What is the relationship between **stress levels** and sleep health?
        - How does **physical activity** influence the quality of sleep?
        - What is the prevalence of **sleep disorders**, and how does it relate to BMI and other factors?
        - What is the **optimal sleep duration** for achieving better sleep quality?

        The insights derived from this analysis can:
        - Help individuals understand how lifestyle changes may improve their sleep health.
        - Assist healthcare professionals in identifying patterns to recommend personalized interventions.
        - Raise awareness about the importance of addressing stress, maintaining a healthy BMI, and engaging in physical activity 
          to improve sleep.
        """
    )
elif selected_option == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(data.head())
    st.write("**Dataset Info:**")
    st.text(data.info())
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())

elif selected_option == "Age Distribution and Sleep Duration":
    st.header("Age Distribution and Sleep Duration")
    fig1 = px.histogram(data, x="Age", title="Age Distribution", nbins=20, color_discrete_sequence=["skyblue"])
    st.plotly_chart(fig1)
    fig2 = px.box(data, x="Age", y="Sleep Duration", title="Age vs. Sleep Duration", color_discrete_sequence=["orange"])
    st.plotly_chart(fig2)
    st.markdown(
        "This analysis shows how sleep duration varies across different age groups. It helps identify if certain age groups are more prone to inadequate sleep, which can inform targeted interventions."
    )

elif selected_option == "Gender and Sleep Duration":
    st.header("Gender and Sleep Duration")
    fig = px.box(data, x="Gender", y="Sleep Duration", title="Gender vs. Sleep Duration", color="Gender")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis explores how sleep duration differs between genders, providing insights into whether gender-specific strategies are needed for improving sleep health."
    )

elif selected_option == "Stress Levels and Sleep Duration":
    st.header("Stress Levels and Sleep Duration")
    fig = px.scatter(data, x="Stress Level", y="Sleep Duration", 
                     title="Stress Levels vs. Sleep Duration", 
                     color="Stress Level", 
                     color_continuous_scale="Viridis")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This scatter plot explores the relationship between stress levels and sleep duration. High stress levels are often associated with shorter sleep durations, which can impact overall health."
    )


elif selected_option == "Physical Activity and Sleep Quality":
    st.header("Physical Activity and Sleep Quality")
    fig = px.box(data, x="Physical Activity Level", y="Quality of Sleep", 
                 title="Physical Activity vs. Sleep Quality", 
                 color="Physical Activity Level")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis examines the impact of physical activity levels on sleep quality. Regular physical activity is often linked to better sleep quality, providing actionable insights for improving sleep health."
    )

elif selected_option == "Sleep Disorders Prevalence":
    st.header("Sleep Disorders Prevalence")
    disorder_counts = data["Sleep Disorder"].value_counts()
    fig = px.pie(names=disorder_counts.index, values=disorder_counts.values, title="Sleep Disorders Prevalence")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis highlights the prevalence of different sleep disorders in the dataset, helping to identify common issues that may require attention."
    )

elif selected_option == "Relationship Between BMI and Sleep Quality":
    st.header("Relationship Between BMI and Sleep Quality")
    fig = px.scatter(data, x="BMI Category", y="Quality of Sleep", 
                     title="BMI vs. Sleep Quality", 
                     color="BMI Category")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis explores how different BMI categories (e.g., underweight, normal, overweight) correlate with sleep quality. Maintaining a healthy BMI can play a significant role in achieving good sleep."
    )


elif selected_option == "Sleep Quality Across Age Groups":
    st.header("Sleep Quality Across Age Groups")
    data["AgeGroup"] = pd.cut(data["Age"], bins=[0, 18, 35, 50, 65, 100], labels=["0-18", "19-35", "36-50", "51-65", "65+"])
    fig = px.box(data, x="AgeGroup", y="Quality of Sleep", title="Sleep Quality Across Age Groups", color="AgeGroup")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis categorizes individuals into age groups to examine how sleep quality varies across the lifespan. It provides insights into which age groups may need targeted sleep interventions."
    )

elif selected_option == "Correlation Between Sleep Duration and Sleep Quality":
    st.header("Correlation Between Sleep Duration and Sleep Quality")
    fig = px.scatter(data, x="Sleep Duration", y="Quality of Sleep", 
                     title="Sleep Duration vs. Sleep Quality", 
                     color="Quality of Sleep", 
                     color_continuous_scale="Plasma")
    st.plotly_chart(fig)
    st.write("### Explanation")
    st.markdown(
        "This analysis investigates the relationship between the duration of sleep and the perceived quality of sleep. It helps to identify the optimal sleep duration for better sleep quality."
    )

elif selected_option == "Predict Sleep Disorder Using Random Forest":
    # Run the machine learning model only when this option is selected
    predict_sleep_disorder()

# Footer
st.markdown(
    """
    <hr style='border: 1px solid #e0e0e0;'>
    <footer style='text-align: center;'>
        🌟 <b>Sleep Health Analysis App</b> - Created by Abeer Zafar
    </footer>
    """,
    unsafe_allow_html=True
)
