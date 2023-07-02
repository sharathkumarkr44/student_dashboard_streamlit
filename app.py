import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Student Assignment Submission Data")

    # Load the data
    data = pd.read_csv('c:/Users/shara/OneDrive/Documents/OVGU/Projects/DBSE_ProvenanceProj-Victor/final_data.csv')

    # Get unique eq_UserID values
    unique_user_ids = data['eq_UserID'].unique()
    # Create a dropdown menu to select eq_UserID
    selected_user_id = st.selectbox('Select eq_UserID', unique_user_ids)

    # Filter the DataFrame based on the selected eq_UserID
    filtered_df = data[data['eq_UserID'] == selected_user_id]

    # Calculate value counts of SQL_category in the filtered DataFrame
    category_counts = filtered_df['SQL_category'].value_counts()

    # Exclude the "Unknown" category if present
    if 'Unknown' in category_counts.index:
        category_counts = category_counts.drop('Unknown')

    # Display the value counts as a bar graph
    if not category_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        category_counts.plot(kind='bar')
        plt.xlabel('SQL_category')
        plt.ylabel('Count')
        plt.title('Query Category Counts')
        st.pyplot(fig)
    else:
        st.write("No data available for the selected eq_UserID.")

# Run the app
if __name__ == '__main__':
    main()