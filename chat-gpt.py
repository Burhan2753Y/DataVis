import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import os, json

# Set page configuration and title
st.set_page_config(page_title="DataVis", layout="wide")
st.title("DataVis")

# =============================================================================
# Sidebar – CSV File Upload and MySQL Database Option
# =============================================================================
st.sidebar.header("Upload CSV Files")
upload_label = "Upload CSV file" if "dfs" not in st.session_state or not st.session_state.dfs else "Upload another CSV file"
uploaded_files = st.sidebar.file_uploader(upload_label, type="csv", accept_multiple_files=True)

if "dfs" not in st.session_state:
    st.session_state.dfs = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.dfs:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dfs[uploaded_file.name] = df
                st.sidebar.success(f"Loaded {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error reading {uploaded_file.name}: {e}")

# ---------------------------
# MySQL Database Option
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Load Data from MySQL")

# Load saved connection details if the file exists.
if os.path.exists("mysql_connection.json"):
    try:
        with open("mysql_connection.json", "r") as f:
            saved_conns = json.load(f)
        if not isinstance(saved_conns, list):
            saved_conns = [saved_conns]
    except Exception as e:
        st.sidebar.error("Error reading saved connection details.")
        saved_conns = []
else:
    saved_conns = []

if saved_conns:
    saved_conn_options = [f"{conn['host']}:{conn['port']} ({conn['username']})" for conn in saved_conns]
    selected_saved_conn = st.sidebar.selectbox("Load saved connection", options=["None"] + saved_conn_options, key="saved_conn_select")
    if selected_saved_conn != "None":
        index = saved_conn_options.index(selected_saved_conn)
        loaded_conn = saved_conns[index]
        host_default = loaded_conn.get("host", "localhost")
        port_default = loaded_conn.get("port", 3306)
        username_default = loaded_conn.get("username", "root")
        password_default = loaded_conn.get("password", "")
    else:
        host_default = "localhost"
        port_default = 3306
        username_default = "root"
        password_default = ""
else:
    host_default = "localhost"
    port_default = 3306
    username_default = "root"
    password_default = ""

use_mysql = st.sidebar.checkbox("Load data from MySQL database?")
if use_mysql:
    host = st.sidebar.text_input("MySQL Host", value=host_default, key="mysql_host")
    port = st.sidebar.number_input("MySQL Port", value=port_default, key="mysql_port")
    username = st.sidebar.text_input("Username", value=username_default, key="mysql_username")
    password = st.sidebar.text_input("Password", type="password", value=password_default, key="mysql_password")
    
    save_conn = st.sidebar.checkbox("Save connection details for future sessions", key="save_mysql_details")
    
    if st.sidebar.button("Fetch Databases", key="fetch_databases"):
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=host,
                port=int(port),
                user=username,
                password=password
            )
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            conn.close()
            st.session_state["mysql_databases"] = databases
            st.sidebar.success("Fetched databases successfully!")
            
            if save_conn:
                if os.path.exists("mysql_connection.json"):
                    try:
                        with open("mysql_connection.json", "r") as f:
                            existing_conns = json.load(f)
                        if not isinstance(existing_conns, list):
                            existing_conns = [existing_conns]
                    except Exception as e:
                        existing_conns = []
                else:
                    existing_conns = []
                current_conn = {"host": host, "port": port, "username": username, "password": password}
                if current_conn not in existing_conns:
                    existing_conns.append(current_conn)
                with open("mysql_connection.json", "w") as f:
                    json.dump(existing_conns, f)
                st.sidebar.info("Connection details saved!")
        except Exception as e:
            st.sidebar.error(f"Error fetching databases: {e}")
    
    if "mysql_databases" in st.session_state:
        selected_db = st.sidebar.selectbox("Select Database", st.session_state["mysql_databases"], key="mysql_selected_db")
        
        if st.sidebar.button("Fetch Tables", key="fetch_tables"):
            try:
                import mysql.connector
                conn = mysql.connector.connect(
                    host=host,
                    port=int(port),
                    user=username,
                    password=password,
                    database=selected_db
                )
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                conn.close()
                st.session_state["mysql_tables"] = tables
                st.sidebar.success("Fetched tables successfully!")
            except Exception as e:
                st.sidebar.error(f"Error fetching tables: {e}")
        
        if "mysql_tables" in st.session_state:
            selected_table = st.sidebar.selectbox("Select Table", st.session_state["mysql_tables"], key="mysql_selected_table")
            if st.sidebar.button("Fetch Data", key="fetch_data"):
                try:
                    import mysql.connector
                    conn = mysql.connector.connect(
                        host=host,
                        port=int(port),
                        user=username,
                        password=password,
                        database=selected_db
                    )
                    query = f"SELECT * FROM {selected_table}"
                    df_db = pd.read_sql(query, conn)
                    conn.close()
                    st.sidebar.success("Data loaded from MySQL successfully!")
                    st.session_state.dfs[f"mysql_{selected_table}"] = df_db
                except Exception as e:
                    st.sidebar.error(f"Error loading data from MySQL: {e}")

# =============================================================================
# Create Tabs for Different Functionalities
# =============================================================================
tabs = st.tabs([ "Datasets","Data Cleaning", "Visualization", "Machine Learning", "Raw Data"])

# =============================================================================
# Tab 1: Datasets
# =============================================================================
with tabs[0]:
    st.header("List of Datasets")
    if st.session_state.dfs:
        for ds_name in list(st.session_state.dfs.keys()):
            new_name = st.text_input("Rename", value=ds_name, key=f"rename_ds_{ds_name}")

            col3, col4 = st.columns([1, 1])
            with col3:
                if st.button("Rename", key=f"rename_btn_{ds_name}"):
                    if new_name and new_name != ds_name:
                        st.session_state.dfs[new_name] = st.session_state.dfs.pop(ds_name)
                        st.success(f"Renamed dataset {ds_name} to {new_name}")
            with col4:
                if st.button("Remove", key=f"remove_btn_{ds_name}"):
                    st.session_state.dfs.pop(ds_name)
                    st.success(f"Removed dataset {ds_name}")
            
    
            
    else:
        st.info("No datasets loaded.")


# =============================================================================
# Tab 2: Data Cleaning and Transformation
# =============================================================================
with tabs[1]:
    st.header("Data Cleaning and Transformation")
    
    if st.session_state.dfs:
        file_choice = st.selectbox("Select a Dataset", list(st.session_state.dfs.keys()))
        df = st.session_state.dfs[file_choice].copy()
        
        with st.expander("Column Renaming (Click to Expand/Collapse)", expanded=True):
            new_columns = {}
            for col in df.columns:
                new_name = st.text_input(f"Rename '{col}'", value=col, key=f"rename_{file_choice}_{col}")
                new_columns[col] = new_name
            if st.button("Apply Renaming", key=f"apply_rename_{file_choice}"):
                df.rename(columns=new_columns, inplace=True)
                st.session_state.dfs[file_choice] = df
                st.success("Columns renamed successfully!")
        
        st.subheader("Delete Columns")
        cols_to_delete = st.multiselect("Select columns to delete", df.columns, key=f"delete_cols_{file_choice}")
        if st.button("Delete Selected Columns", key=f"delete_cols_button_{file_choice}"):
            if cols_to_delete:
                df.drop(columns=cols_to_delete, inplace=True)
                st.session_state.dfs[file_choice] = df
                st.success(f"Deleted columns: {', '.join(cols_to_delete)}")
            else:
                st.info("No columns selected for deletion.")
        
        st.subheader("Missing Value Treatment")
        st.write("Missing values count per column:")
        missing_counts = df.isna().sum()
        st.write(missing_counts)
        
        cleaning_options = {}
        for col in df.columns:
            option = st.selectbox(
                f"Cleaning option for column '{col}'",
                ["No Action", "Drop rows", "Replace with Mean", "Replace with Median", "Replace with Mode", "Replace with Custom Value"],
                key=f"cleaning_{file_choice}_{col}"
            )
            if option == "Replace with Custom Value":
                custom_val = st.text_input(f"Custom value for '{col}'", key=f"custom_{file_choice}_{col}")
                cleaning_options[col] = (option, custom_val)
            else:
                cleaning_options[col] = option

        if st.button("Apply Missing Value Treatment", key=f"apply_missing_{file_choice}"):
            for col, option in cleaning_options.items():
                if option == "Drop rows":
                    df = df.dropna(subset=[col])
                elif option == "Replace with Mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                elif option == "Replace with Median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                elif option == "Replace with Mode":
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
                elif isinstance(option, tuple) and option[0] == "Replace with Custom Value":
                    try:
                        custom_val = float(option[1])
                    except:
                        custom_val = option[1]
                    df[col].fillna(custom_val, inplace=True)
            st.session_state.dfs[file_choice] = df
            st.success("Missing value treatment applied.")
        
        st.subheader("Outlier Removal")
        outlier_column = st.selectbox("Select column for outlier detection", df.columns, key=f"outlier_col_{file_choice}")
        if pd.api.types.is_numeric_dtype(df[outlier_column]):
            Q1 = df[outlier_column].quantile(0.25)
            Q3 = df[outlier_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound)]
            st.write(f"Detected {len(outliers)} outlier rows in '{outlier_column}'.")
            st.dataframe(outliers)
            remove = st.checkbox("Remove outlier rows?", key=f"remove_outliers_{file_choice}")
            if remove:
                df = df[(df[outlier_column] >= lower_bound) & (df[outlier_column] <= upper_bound)]
                st.session_state.dfs[file_choice] = df
                st.success("Outliers removed.")
        else:
            st.info("Selected column is not numeric; outlier detection is not applicable.")
    else:
        st.info("Please upload a Dataset from the sidebar.")

# =============================================================================
# Tab 3: Data Visualization
# =============================================================================
with tabs[2]:
    st.header("Data Visualization")
    if st.session_state.dfs:
        file_choice_viz = st.selectbox("Select a Dataset for visualization", list(st.session_state.dfs.keys()), key="viz_file")
        df_viz = st.session_state.dfs[file_choice_viz]
        
        st.subheader("Single Chart Visualization")
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Scatter", "Line", "Bar"], key="chart_type")
        x_axis = st.selectbox("Select x-axis column", df_viz.columns, key="x_axis")
        if chart_type == "Histogram":
            y_axis = st.selectbox("Select y-axis column (optional)", ["None"] + list(df_viz.columns), key="y_axis")
        else:
            y_axis = st.selectbox("Select y-axis column", df_viz.columns, key="y_axis")
        
        if st.button("Generate Chart", key="generate_chart"):
            if chart_type == "Histogram":
                if y_axis == "None":
                    fig = px.histogram(df_viz, x=x_axis)
                else:
                    fig = px.histogram(df_viz, x=x_axis, y=y_axis)
            elif chart_type == "Scatter":
                fig = px.scatter(df_viz, x=x_axis, y=y_axis)
            elif chart_type == "Line":
                fig = px.line(df_viz, x=x_axis, y=y_axis)
            elif chart_type == "Bar":
                fig = px.bar(df_viz, x=x_axis, y=y_axis)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Compare Two Charts")
        if len(st.session_state.dfs) < 2:
            st.info("Please upload at least two Datasets to use this comparison feature.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                file_choice1 = st.selectbox("Select first Dataset", list(st.session_state.dfs.keys()), key="compare_file1")
                df1 = st.session_state.dfs[file_choice1]
                chart_type1 = st.selectbox("Select Chart Type", ["Histogram", "Scatter", "Line", "Bar"], key="chart_type1")
                x_axis1 = st.selectbox("Select x-axis column for File 1", df1.columns, key="x_axis1")
                if chart_type1 == "Histogram":
                    y_axis1 = st.selectbox("Select y-axis column (optional) for File 1", ["None"] + list(df1.columns), key="y_axis1")
                else:
                    y_axis1 = st.selectbox("Select y-axis column for File 1", df1.columns, key="y_axis1")
            with col2:
                file_choice2 = st.selectbox("Select second Dataset", list(st.session_state.dfs.keys()), key="compare_file2")
                df2 = st.session_state.dfs[file_choice2]
                chart_type2 = st.selectbox("Select Chart Type", ["Histogram", "Scatter", "Line", "Bar"], key="chart_type2")
                x_axis2 = st.selectbox("Select x-axis column for File 2", df2.columns, key="x_axis2")
                if chart_type2 == "Histogram":
                    y_axis2 = st.selectbox("Select y-axis column (optional) for File 2", ["None"] + list(df2.columns), key="y_axis2")
                else:
                    y_axis2 = st.selectbox("Select y-axis column for File 2", df2.columns, key="y_axis2")
            
            if st.button("Generate Comparison Charts", key="generate_compare_chart"):
                fig = make_subplots(rows=1, cols=2, subplot_titles=(file_choice1, file_choice2))
                
                if chart_type1 == "Histogram":
                    if y_axis1 == "None":
                        fig1 = px.histogram(df1, x=x_axis1)
                    else:
                        fig1 = px.histogram(df1, x=x_axis1, y=y_axis1)
                elif chart_type1 == "Scatter":
                    fig1 = px.scatter(df1, x=x_axis1, y=y_axis1)
                elif chart_type1 == "Line":
                    fig1 = px.line(df1, x=x_axis1, y=y_axis1)
                elif chart_type1 == "Bar":
                    fig1 = px.bar(df1, x=x_axis1, y=y_axis1)
                for trace in fig1.data:
                    fig.add_trace(trace, row=1, col=1)
                
                if chart_type2 == "Histogram":
                    if y_axis2 == "None":
                        fig2 = px.histogram(df2, x=x_axis2)
                    else:
                        fig2 = px.histogram(df2, x=x_axis2, y=y_axis2)
                elif chart_type2 == "Scatter":
                    fig2 = px.scatter(df2, x=x_axis2, y=y_axis2)
                elif chart_type2 == "Line":
                    fig2 = px.line(df2, x=x_axis2, y=y_axis2)
                elif chart_type2 == "Bar":
                    fig2 = px.bar(df2, x=x_axis2, y=y_axis2)
                for trace in fig2.data:
                    fig.add_trace(trace, row=1, col=2)
                
                fig.update_layout(title_text="Comparison of Charts")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a Dataset from the sidebar.")

# =============================================================================
# Tab 4: Machine Learning
# =============================================================================
with tabs[3]:
    st.header("Machine Learning")
    if st.session_state.dfs:
        ml_file = st.selectbox("Select a Dataset file for ML", list(st.session_state.dfs.keys()), key="ml_file")
        df_ml = st.session_state.dfs[ml_file]
        st.write("Data preview:")
        st.dataframe(df_ml.head())
        
        target_col = st.selectbox("Select target column", df_ml.columns, key="target_col")
        feature_cols = st.multiselect("Select feature columns", [col for col in df_ml.columns if col != target_col], key="feature_cols")
        test_size = st.slider("Test set size (%)", 10, 50, 20, key="test_size")
        
        if st.button("Train Models"):
            if target_col and feature_cols:
                X = df_ml[feature_cols]
                y = df_ml[target_col]
                X_processed = pd.get_dummies(X, drop_first=True)
                
                if pd.api.types.is_numeric_dtype(y):
                    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size/100, random_state=42)
                    
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    pred_lr = lr.predict(X_test)
                    mse_lr = mean_squared_error(y_test, pred_lr)
                    
                    knn = KNeighborsRegressor()
                    knn.fit(X_train, y_train)
                    pred_knn = knn.predict(X_test)
                    mse_knn = mean_squared_error(y_test, pred_knn)
                    
                    mlp = MLPRegressor(max_iter=500)
                    mlp.fit(X_train, y_train)
                    pred_mlp = mlp.predict(X_test)
                    mse_mlp = mean_squared_error(y_test, pred_mlp)
                    
                    st.subheader("Regression Model Performance (MSE)")
                    st.write(f"Linear Regression MSE: {mse_lr:.4f}")
                    st.write(f"KNN Regression MSE: {mse_knn:.4f}")
                    st.write(f"MLP Regression MSE: {mse_mlp:.4f}")
                else:
                    y_encoded = pd.factorize(y)[0]
                    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=test_size/100, random_state=42)
                    
                    logreg = LogisticRegression(max_iter=500)
                    logreg.fit(X_train, y_train)
                    pred_logreg = logreg.predict(X_test)
                    acc_logreg = accuracy_score(y_test, pred_logreg)
                    
                    knn_clf = KNeighborsClassifier()
                    knn_clf.fit(X_train, y_train)
                    pred_knn_clf = knn_clf.predict(X_test)
                    acc_knn = accuracy_score(y_test, pred_knn_clf)
                    
                    mlp_clf = MLPClassifier(max_iter=500)
                    mlp_clf.fit(X_train, y_train)
                    pred_mlp_clf = mlp_clf.predict(X_test)
                    acc_mlp = accuracy_score(y_test, pred_mlp_clf)
                    
                    st.subheader("Classification Model Performance (Accuracy)")
                    st.write(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
                    st.write(f"KNN Classifier Accuracy: {acc_knn:.4f}")
                    st.write(f"MLP Classifier Accuracy: {acc_mlp:.4f}")
            else:
                st.error("Please select a target column and at least one feature column.")
    else:
        st.info("Please upload a Dataset from the sidebar.")

# =============================================================================
# Tab 5: Raw Data Editor and Export
# =============================================================================
with tabs[4]:
    st.header("Raw Data and Data Editor")
    if st.session_state.dfs:
        for file_name, df_raw in st.session_state.dfs.items():
            st.subheader(f"Data from {file_name}")
            
            if hasattr(st, 'experimental_data_editor'):
                edited_df = st.experimental_data_editor(df_raw, num_rows="dynamic", key=f"editor_{file_name}")
            else:
                st.write("Data preview:")
                st.dataframe(df_raw)
                csv_str = df_raw.to_csv(index=False)
                new_csv_str = st.text_area("Edit CSV data (make changes to CSV text)", csv_str, height=200, key=f"text_editor_{file_name}")
                try:
                    edited_df = pd.read_csv(StringIO(new_csv_str))
                except Exception as e:
                    st.error("Error parsing CSV. Reverting to original data.")
                    edited_df = df_raw
            
            st.session_state.dfs[file_name] = edited_df
            
            if st.button(f"Save {file_name}", key=f"save_{file_name}"):
                edited_df.to_csv(file_name, index=False)
                st.success(f"Saved {file_name} to disk.")
    else:
        st.info("Please upload a Dataset from the sidebar.")

