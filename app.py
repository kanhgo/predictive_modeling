# Dependencies
import streamlit as st
import pandas as pd
import numpy as np

# A function to load the model coefficients and intercept.
def get_model_parameters():
    """Loads the model coefficients and intercept as a dictionary."""
    coefficients = {
        'HighBP': 0.69850,
        'HighChol': 0.57242,
        'CholCheck': 1.31883,
        'BMI': 6.62451,
        'Stroke': 0.17279,
        'HeartDiseaseorAttack': 0.22394,
        'HvyAlcoholConsump': -0.63673,
        'NoDocbcCost': 0.05890,
        'GenHlth': 0.54642,
        'MentHlth': -0.07747,
        'PhysHlth': -0.18691,
        'DiffWalk': 0.06530,
        'Sex': -0.04751,
        'Age': 0.14884,
        'Education': -0.04751,
        'Income': -0.06777
    }
    intercept = -5.67339
    return coefficients, intercept

# The main function to run the app
def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

    st.title("Diabetes Risk Predictor")
    st.markdown("Upload a CSV file with patient data to get predictions of diabetes status.")
    st.empty()
    with st.expander("Click to review required data parameters"):
        st.markdown("*Required data columns: (i) HighBP, (ii) HighChol, (iii) CholCheck (iv) BMI, (v) Stroke, (vi) (HeartDiseaseorAttack, (vii) HvyAlcoholConsump, (viii) NoDocbcCost (ix) GenHlth, (x) MenHlth (xi) PhysHlth, (xii) DiffWalk, (xiii) Sex, (xiv) Age, (xv) Education, and (xvi) Income.*")
        st.markdown(":grey[Click [here](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) for feature details (from page 13)]")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded file into a DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("File uploaded successfully. \nPreview:")
            st.dataframe(df.head())

            # Get model parameters
            coefficients, intercept = get_model_parameters()
            features = list(coefficients.keys())

            # Ensure all required features are in the DataFrame
            if not all(feature in df.columns for feature in features):
                missing_features = [f for f in features if f not in df.columns]
                st.error(f"Error: Missing required columns in the CSV file: {', '.join(missing_features)}")
                return

            # # --- Multi-step explanation of the above check for missing features ---
            # missing_features = []
            # for feature in features:
            #     if feature not in df.columns:
            #         missing_features.append(feature)

            # if len(missing_features) > 0:
            #     st.error(f"Error: Missing required columns in the CSV file: {', '.join(missing_features)}")
            #     return
            
            # # --- End of multi-step check ---

            # Create a progress bar
            st.markdown("Predicting diabetes risk...")
            progress_bar = st.progress(0)

            # Calculate log odds and predictions
            predictions = []
            probabilities = []
            total_rows = len(df)
            
            for i, row in df.iterrows():
                log_odds = intercept
                for feature, coeff in coefficients.items():
                    log_odds += row[feature] * coeff
                
                probability = 1 / (1 + np.exp(-log_odds))
                
                predictions.append('Diabetic / Pre-Diabetic' if probability >= 0.5 else 'Not Diabetic')
                probabilities.append(probability)
                
                # Update progress bar
                progress_bar.progress((i + 1) / total_rows)

            # Add the new columns to the DataFrame
            df['Predicted_Diabetes_Status'] = predictions
            df['Prediction_Probability'] = probabilities
            
            st.markdown("---")
            st.success("Predictions complete!")
            st.markdown("***Note: Not diagnostic. For informational purposes only.***")
            st.empty()
            st.write("Preview of predicted data:")
            st.dataframe(df.head(10))

            # Provide a download button for the new file
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download updated CSV file",
                data=csv,
                file_name=f'predictions_{uploaded_file.name}',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()