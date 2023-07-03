import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def main():
    st.title("Student Assignment Submission Data")
    tab1,tab2,tab3 = st.tabs(["Home","Error Frequency", "Summary Table"])
    # Load the data
    data = pd.read_csv('c:/Users/shara/OneDrive/Documents/OVGU/Projects/DBSE_ProvenanceProj-Victor/final_data.csv')
    # Get unique eq_UserID values
    unique_user_ids = data['eq_UserID'].unique()
    # Get unique task_id values
    unique_task_ids = data['eq_taskid'].unique()
    # Get different semester batches
    unique_sem_ids = data['sem_passphrase'].unique()
    # Get instructor names
    unique_instructor_names = data['egrp_instructor'].unique()

    with tab1:
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


    # Convert eq_ts column to datetime format
    #data["eq_ts"] = pd.to_datetime(data["eq_ts"])
    # data["eq_ts"] = pd.to_datetime(data["eq_ts"])

    # submission on same hour
    # st.title("submission on same hour")
    # Create the initial selectbox for egrp_instructor
    # selected_instructor = st.selectbox("Select Instructor", data["egrp_instructor"].unique())

    # Filter the data based on the selected instructor
    # filtered_data = data[data["egrp_instructor"] == selected_instructor]

    # Create the selectbox for eq_taskid
    # selected_taskid = st.selectbox("Select Task ID", filtered_data["eq_taskid"].unique())

    # Filter the data based on the selected task ID
    # filtered_data = filtered_data[filtered_data["eq_taskid"] == selected_taskid]

    # Find eq_UserID, SQL_category, count(eq_ts), first_submission_time, and last_submission_time
    # filtered_data = filtered_data.groupby(["eq_UserID", "SQL_category"]).agg(
    #     Total_Attempt=("eq_ts", "count"),
    #     first_submission_time=("eq_ts", "min"),
    #     last_submission_time=("eq_ts", "max")
    # )

    # Filter the data where last_submission_time of minimum 2 eq_UserID lies in the same hour of the same date
    # filtered_data = filtered_data.groupby(pd.Grouper(freq="H", key="last_submission_time")).filter(lambda x: x.shape[0] >= 2)

    # Show the resulting table
    # st.table(filtered_data)


    #error frequency per task
    with tab2:
        # specify the user ID
        selected_userid = st.selectbox('Select user_id', unique_user_ids)
        #df_sample = data[data["eq_UserID"] == selected_userid]
        # Specify the task ID to filter on
        selected_taskid = st.selectbox('Select task_id', unique_task_ids)

        # Filter the dataframe based on the task ID and userID
        filtered_df = data.loc[(data.eq_UserID == selected_userid) & (data.eq_taskid == selected_taskid)]

        # Get the value counts for the eq_errors column of the filtered dataframe
        value_counts = filtered_df.eq_errors.value_counts()

        # Check if value_counts is empty
        if not value_counts.empty:
            # Define a list of colors for each category
            colors = ['blue', 'green', 'yellow', 'orange', 'red']
            # Create a bar graph with custom colors and figure size
            fig, ax = plt.subplots(figsize=(8,4))
            ax = value_counts.plot(kind='bar', color=colors)
            # Add labels and title
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.set_title(f'Error Frequency for Task ID = {selected_taskid}')
            # Show the plot
            st.pyplot(fig)
        else:
            st.write("No data available for the selected user ID and task ID.")

    with tab3:
        def process_data(df):
            df['eq_ts'] = pd.to_datetime(df['eq_ts'])
            df['first_submission_time'] = df.groupby('eq_UserID')['eq_ts'].transform('min')
            df['last_submission_time'] = df.groupby('eq_UserID')['eq_ts'].transform('max')
            df['Time_taken_To_Complete_Assignment'] = (df['last_submission_time'] - df['first_submission_time']).astype(str)
            df['Time_taken_To_Complete_Assignment'] = df['Time_taken_To_Complete_Assignment'].apply(lambda x: str(pd.Timedelta(x)))
            df['minutes'] = round(df['Time_taken_To_Complete_Assignment'] / pd.Timedelta(minutes=1))
            df['Total_Attempt'] = df.groupby('eq_UserID')['eq_ts'].transform('count')
            # Replace unknown values in SQL_category with the most frequent value for each eq_taskid
            df['SQL_category'] = df.groupby('eq_taskid')['SQL_category'].transform(lambda x: x.fillna(x.value_counts().idxmax()))
            df = df.drop_duplicates(subset='eq_UserID', keep='first')
            return df[['eq_UserID', 'eq_taskid', 'SQL_category', 'first_submission_time', 'last_submission_time', 'Time_taken_To_Complete_Assignment', 'Total_Attempt', 'minutes']]

        def cluster_and_plot(eq_taskid, max_clusters):
            # Select the features for clustering
            features = ['minutes', 'Total_Attempt']

            # Determine the range of clusters to try
            cluster_range = range(1, max_clusters + 1)

            # Calculate inertia for each number of clusters
            inertias = []
            for num_clusters in cluster_range:
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(output[features])
                inertias.append(kmeans.inertia_)

            # Plot the elbow graph
            fig, ax = plt.subplots(figsize=(8,4))
            plt.plot(cluster_range, inertias, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title(f'Elbow Graph for eq_taskid {eq_taskid}')
            st.pyplot(fig)

            # Choose the optimal number of clusters based on the elbow graph
            optimal_clusters = st.number_input("Enter the optimal number of clusters based on the elbow graph:", min_value=1, max_value=max_clusters, value=1, step=1)

            if optimal_clusters != 1:
                # Perform clustering with the chosen number of clusters
                kmeans = KMeans(n_clusters=optimal_clusters)
                cluster_labels = kmeans.fit_predict(output[features])

                # Plot the clusters
                fig, ax = plt.subplots(figsize=(8,4))
                plt.scatter(output['minutes'], output['Total_Attempt'], c=cluster_labels)

                # Annotate the data points with eq_UserID
                for i, eq_userid in enumerate(output['eq_UserID']):
                    plt.annotate(eq_userid, (output['minutes'].iloc[i], output['Total_Attempt'].iloc[i]))

                plt.xlabel('Minutes')
                plt.ylabel('Total Attempt')
                plt.title(f'Clustering for eq_taskid {eq_taskid} (Optimal Clusters: {optimal_clusters})')
                st.pyplot(fig)


        #specify semester
        selected_sem = st.selectbox('Select Semester', unique_sem_ids)
        selected_instructor = st.selectbox('Select Instructor', unique_instructor_names)
        selected_taskid_summary = st.selectbox('Select Task-id', unique_task_ids)
        filtered_data = data.loc[(data.sem_passphrase == selected_sem) & (data.egrp_instructor == selected_instructor) & (data.eq_taskid == selected_taskid_summary)]
        output = process_data(filtered_data)
        #output = output.set_index('eq_UserID')
        st.dataframe(output)
        #for elbow graph and clustering graph
        cluster_and_plot(selected_taskid_summary, 3)


# Run the app
if __name__ == '__main__':
    main()