import streamlit as st
import pandas as pd
import altair as alt
import joblib

# Load model
model = joblib.load("user_prediction_model.pkl")

st.title("LinkedIn User Prediction App")

st.subheader("User Guide")
st.write("""
This app uses a logistic regression model trained on survey research data 
to estimate the probability that a person uses LinkedIn based on 
demographic and lifestyle characteristics.

Adjust the inputs to generate a personalized LinkedIn usage prediction.
""")

st.subheader("Select Your Characteristics")

# Layout: User Inputs on left, chart and results on right
left, right = st.columns([1, 1])

# LEFT COLUMN: SELECTIONS
with left:
#Dictionaries that convert numeric income and education codes into readable labels
    income_guide = {
        "Less than $10k": 1,
        "$10k to $20k": 2,
        "$20k to $30k": 3,
        "$30k to $40k": 4,
        "$40k to $50k": 5,
        "$50k to $75k": 6,
        "$75k to $100k": 7,
        "$100k to $150k": 8,
        "$150k+": 9
    }

    education_guide = {
        "Less than high school": 1,
        "HS incomplete": 2,
        "HS diploma / GED": 3,
        "Some college": 4,
        "Associate degree": 5,
        "Bachelor's degree": 6,
        "Some graduate school": 7,
        "Graduate/Professional degree": 8
    }

#Creating questions
    income_selection = st.selectbox(
        "Select your income level:",
        list(income_guide.keys())
    )
    income_numeric = income_guide[income_selection]

    education_options = st.selectbox(
        "Select your education level:",
        list(education_guide.keys())
    )
    education_numeric = education_guide[education_options]

    parent_input = st.radio(
        "Are you a parent of a child under 18 living in your home?",
        ["Yes", "No"]
    )
    married_input = st.selectbox(
        "Select your marital status:",
        ["Not Married", "Married"]
    )
    female_input = st.selectbox(
        "Select your gender:",
        ["Male", "Female"]
    )

    age = st.slider("Select your age", 18, 98, 28)

    # Converting answers to numeric
    parent = 1 if parent_input == "Yes" else 0
    married = 1 if married_input == "Married" else 0
    female = 1 if female_input == "Female" else 0

    # Preparing user input dataframe
    user_info = pd.DataFrame({
        "income": [income_numeric],
        "education": [education_numeric],
        "parent": [parent],
        "married": [married],
        "female": [female],
        "age": [age]
    })


# RIGHT COLUMN: FEATURE INFLUENCE STRENGTH
with right:

    st.subheader("Top Factors Affecting Your Prediction")

    coeffs = model.coef_[0]
    features = ["Income", "Education", "Parent", "Married", "Female", "Age"]

    user_values = [
        income_numeric,
        education_numeric,
        parent,
        married,
        female,
        age
    ]

    # Computing contribution strength
    effect_strength = [abs(c * v) for c, v in zip(coeffs, user_values)]

    # Creating dataframe
    strength_df = pd.DataFrame({
        "Feature": features,
        "Strength": effect_strength
    })

    # Hiding features that contribute close to nothing
    strength_df = strength_df[strength_df["Strength"] > 0.01]

    # Sorting features in order of strength for better readability
    strength_df = strength_df.sort_values("Strength", ascending=False)

    # Bar chart
    chart = (
        alt.Chart(strength_df)
        .mark_bar(color="#1E90FF")
        .encode(
            x=alt.X("Strength:Q", title="Strength of Input"),
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature", "Strength"]
        )
        .properties(
            width="container",
            height=300,
            title="This Chart Updates as You Adjust Your Inputs"
        )
    )

    st.altair_chart(chart, use_container_width=True)

   # Prediction Button
    button_column, result_column = st.columns([1, 3])

    with button_column:
        run_prediction = st.button("Predict")

    with result_column:
        if run_prediction:
            prediction = model.predict(user_info)[0]
            probability = model.predict_proba(user_info)[0][1]

            prediction_text = "LinkedIn User" if prediction == 1 else "Not a User"

            st.markdown(f"**Predicted Class:** {prediction_text}")
            st.markdown(f"**Probability of Using LinkedIn:** {probability:.3f}")

#MAIN SECTION
# Show the Learn More section only after a prediction is made
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# activate after user clicks Predict
if run_prediction:
    st.session_state.predicted = True

# show Learn More only when predicted = True
if st.session_state.predicted:

    st.markdown(" ")

    show_insights = st.button("Click Here to Learn More About Your Results!")

    if show_insights:
        
        st.subheader("Patterns Observed Based on LinkedIn Usage")
        ss = pd.read_csv("ss_cleaned.csv")
        
        
        # Displaying LinkedIn usage by income level
        st.write("### Income Level")
        
        income_usage = ss.groupby("income")["sm_li"].mean().reset_index()
        income_usage["usage_pct"] = income_usage["sm_li"] * 100
        
        chart_income = (
            alt.Chart(income_usage)
            .mark_bar(color="#4C72B0")
            .encode(
                x=alt.X("income:O", title="Income Level", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["income", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_income, use_container_width=True)
        st.write("The data suggests higher income levels are showing a greater likelihood of LinkedIn usage.")
        
        
        # Displaying LinkedIn usage by education level
        st.write("### Education Level")
        
        educ = ss.groupby("education")["sm_li"].mean().reset_index()
        educ["usage_pct"] = educ["sm_li"] * 100
        
        chart_educ = (
            alt.Chart(educ)
            .mark_bar(color="#55A868")
            .encode(
                x=alt.X("education:O", title="Education Level", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["education", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_educ , use_container_width=True)
        st.write("The data shows LinkedIn usage increases as education levels increase.")
        
        
        # Displaying LinkedIn usage by age
        st.write("### Age")
        
        age_data = ss.groupby("age")["sm_li"].mean().reset_index()
        age_data["usage_pct"] = age_data["sm_li"] * 100
        
        chart_age = (
            alt.Chart(age_data)
            .mark_line(color="#C44E52")
            .encode(
                x=alt.X("age:Q", title="Age"),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["age", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_age, use_container_width=True)
        st.write("The data shows LinkedIn usage is highest among individuals in early and mid career stages and decreases at older ages.")
        
        
        # Displaying LinkedIn usage by parent status
        st.write("### Parent Status")
        
        parent_usage = ss.groupby("parent")["sm_li"].mean().reset_index()
        parent_usage["parent"] = parent_usage["parent"].map({0: "Non Parent", 1: "Parent"})
        parent_usage["usage_pct"] = parent_usage["sm_li"] * 100
        
        chart_parent = (
            alt.Chart(parent_usage)
            .mark_bar(color="#8172B2")
            .encode(
                x=alt.X("parent:N", title="Parent Status", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["parent", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_parent, use_container_width=True)
        st.write("Parent status shows only small differences in LinkedIn usage.")
        
        
        # Displaying LinkedIn usage by marital status
        st.write("### LinkedIn Usage by Marital Status")
        
        married_usage = ss.groupby("married")["sm_li"].mean().reset_index()
        married_usage["married"] = married_usage["married"].map({0: "Not Married", 1: "Married"})
        married_usage["usage_pct"] = married_usage["sm_li"] * 100
        
        chart_married = (
            alt.Chart(married_usage)
            .mark_bar(color="#CCB974")
            .encode(
                x=alt.X("married:N", title="Marital Status", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["married", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_married, use_container_width=True)
        st.write("Marital status shows minimal variation in LinkedIn usage behavior.")
        
        
        # Displaying LinkedIn usage by gender
        st.write("### Gender")
        
        gender_usage = ss.groupby("female")["sm_li"].mean().reset_index()
        gender_usage["female"] = gender_usage["female"].map({0: "Male", 1: "Female"})
        gender_usage["usage_pct"] = gender_usage["sm_li"] * 100
        
        chart_gender = (
            alt.Chart(gender_usage)
            .mark_bar(color="#64B5CD")
            .encode(
                x=alt.X("female:N", title="Gender", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("usage_pct:Q", title="Predicted Likelihood (%)"),
                tooltip=["female", "usage_pct"]
            )
            .properties(height=250)
        )
        
        st.altair_chart(chart_gender, use_container_width=True)
        st.write("Gender does not show a strong influence on LinkedIn usage patterns in this dataset.")
