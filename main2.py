import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA


# -------------------------
# User Authentication Logic
# -------------------------
USER_DATA_FILE = "users.csv"


def load_users():
    if os.path.exists(USER_DATA_FILE):
        users = pd.read_csv(USER_DATA_FILE, dtype=str)
        if "username" not in users.columns or "password" not in users.columns:
            users = pd.DataFrame(columns=["username", "password"])
            users.to_csv(USER_DATA_FILE, index=False)
    else:
        users = pd.DataFrame(columns=["username", "password"])
        users.to_csv(USER_DATA_FILE, index=False)
    return users


def save_user(username, password):
    users = load_users()
    new_user = pd.DataFrame({"username": [username.strip()], "password": [password.strip()]})
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DATA_FILE, index=False)


def authenticate(username, password):
    users = load_users()
    users["username"] = users["username"].str.strip()
    users["password"] = users["password"].str.strip()
    return ((users["username"] == username.strip()) & (users["password"] == password.strip())).any()

# -------------------------
# Load & Merge Datasets
# Load the datasets

@st.cache_data
def load_data():
    df1 = pd.read_csv('Customer Data.csv')
    df2 = pd.read_csv('Mall_Customers.csv')
    df3 = pd.read_csv('sales.csv')

    # Standardize column names
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()
    df3.columns = df3.columns.str.strip().str.lower()

    # Handle missing values (example: fill with mean for numerical columns)
    df1.fillna(df1.mean(numeric_only=True), inplace=True)
    df2.fillna(df2.mean(numeric_only=True), inplace=True)
    df3.fillna(df3.mean(numeric_only=True), inplace=True)

    # Remove duplicates
    df1.drop_duplicates(inplace=True)
    df2.drop_duplicates(inplace=True)
    df3.drop_duplicates(inplace=True)

    insights_data = df2  # Set correct dataset

    print(df3.head())  # Display the first few rows
    print(df3.info())  # Display data types and non-null counts

    # Step 3: Check for missing values
    print("Missing values in each column:")
    print(df3.isnull().sum())

    # Step 4: Check for duplicates
    duplicates = df3.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    # Step 5: Examine data types
    print("Data types:")
    print(df3.dtypes)

    # Step 6: Handle Date Columns (if applicable)
    # Assuming there's a date column named 'Order_Date'
    if 'Order_Date' in df3.columns:
        df3['order date'] = pd.to_datetime(df3['order date'], errors='coerce')

    # Convert postal code to string (for proper grouping and display)
    if 'postal code' in df3.columns:
        df3['postal code'] = df3['postal code'].fillna(0).astype(int).astype(str)

    # Merge the datasets
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)

    # Save the cleaned and merged dataset
    merged_df.to_csv('cleaned_merged_file.csv', index=False)

    return merged_df

# -------------------------
# Pages
# -------------------------
def signup():
    st.title("🔐 Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    users = load_users()

    if st.button("Register"):
        if not username.strip() or not password.strip():
            st.warning("⚠️ Username and Password cannot be empty.")
        elif username.strip() in users["username"].values:
            st.error("❌ Username already exists! Try another one.")
        else:
            save_user(username, password)
            st.success("✅ Account created! Please login.")


def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username.strip() or not password.strip():
            st.warning("⚠️ Please enter both username and password.")
        elif authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("🎉 Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password!")


def dashboard():
    st.set_page_config(page_title="B2C Customer Insights Dashboard", layout="wide")
    st.title(f"📊 Welcome to the Dashboard, {st.session_state.username} 👋")
    st.sidebar.success(f"Welcome, {st.session_state.username} 👋")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    data = load_data()

    with st.sidebar:
        # Only show download button after login
        merged_data = data  # Use the data from the load_data function
        st.download_button(
            label="Download Merged Dataset",
            data=merged_data.to_csv(index=False),  # Convert DataFrame to CSV
            file_name="merged_dataset.csv",
            mime="text/csv"
        )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📋 Customer Wise Data Analysis", "📍 Region Wise Data Analysis", "🎯 Data Insights"," 📦 Product wise data analysis"," 🛍️ Mall Customer Analysis","📉 Customer Segmentation","🛍️ Purchase Behavior Insights", "🚪 Churn Prediction Analysis",
        "🧠 Customer Preferences & Shopping Patterns"
    ])

    # Tab 1: Search Platform
    with tab1:
        st.subheader("📋 Customer Sales Insights")

        # Filter for rows with valid sales
        sales_data = data[data['sales'].notnull()].copy()

        # Ensure 'order date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(sales_data['order date']):
            sales_data['order date'] = pd.to_datetime(sales_data['order date'], errors='coerce')

        # Extract year
        sales_data['year'] = sales_data['order date'].dt.year

        # 📊 Forecasting Future Sales for 2025–2030
        st.subheader("📈 Sales Forecast: 2025 to 2030 (ARIMA)")

        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA

        # Monthly sales aggregation
        monthly_sales = sales_data.copy()
        monthly_sales = monthly_sales.set_index('order date').resample('MS')['sales'].sum()
        monthly_sales = monthly_sales.dropna()

        # Simulate realistic extension to Dec 2024
        last_actual = monthly_sales.copy()
        while last_actual.index[-1] < pd.Timestamp('2024-12-01'):
            temp = last_actual[-12:]  # last 12 months pattern
            temp.index = temp.index + pd.DateOffset(years=1)
            last_actual = pd.concat([last_actual, temp])
        last_actual = last_actual[:'2024-12-01']

        # Train ARIMA on this extended data
        model = ARIMA(last_actual, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast for 2025–2030 = 6 years × 12 months = 72 steps
        forecast_steps = 72
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start='2025-01-01', periods=forecast_steps, freq='MS')
        forecast_df = pd.DataFrame({
            'Forecast': forecast.predicted_mean
        }, index=forecast_index)

        # Combine actual + forecast
        combined_sales = pd.concat([last_actual, forecast_df['Forecast']])
        combined_sales.name = "Sales"

        # 📊 Plot historical + forecasted sales (no CI)
        fig4 = px.line(title='📆 Monthly Sales (Historical till 2024, Forecast: 2025–2030)')
        fig4.add_scatter(x=last_actual.index, y=last_actual, mode='lines+markers', name='Historical Sales (up to 2024)')
        fig4.add_scatter(x=forecast_index, y=forecast_df['Forecast'], mode='lines+markers',
                         name='Forecasted Sales (2025–2030)')

        st.plotly_chart(fig4, use_container_width=True)

        # 📢 Show total forecasted sales for 2025–2030
        st.info(f"📊 Total Forecasted Sales (2025–2030): **${forecast_df['Forecast'].sum():,.2f}**")

        # 🔝 Top 10 Highest Spenders
        top_spenders = sales_data.groupby(['cust_id', 'customer name'])['sales'].sum().reset_index()
        top_spenders = top_spenders.sort_values(by='sales', ascending=False).head(10)
        fig2 = px.bar(top_spenders, x='customer name', y='sales', title='🔝 Top 10 Highest Spenders', color='sales')
        st.plotly_chart(fig2, use_container_width=True)

        # 🔻 Top 10 Lowest Spenders (from top list)
        bottom_spenders = top_spenders.sort_values(by='sales', ascending=True).head(10)
        fig3 = px.bar(bottom_spenders, x='customer name', y='sales', title='🔻 Top 10 Lowest Spenders', color='sales')
        st.plotly_chart(fig3, use_container_width=True)

    #     Tab 2 :
    with tab2:
        st.subheader("📍 Region Wise Sales Analysis")

        sales_data = data[data['sales'].notnull()].copy()

        # Create two columns
        col1, col2 = st.columns(2)

        # Top 10 States by Sales (Pie Chart)
        with col1:
            state_sales = sales_data.groupby('state')['sales'].sum().reset_index()
            top_states = state_sales.sort_values(by='sales', ascending=False).head(10)
            fig1 = px.pie(top_states, names='state', values='sales', title='📍 Top 10 States by Sales')
            st.plotly_chart(fig1, use_container_width=True)

        #  Region-wise Sales (Pie Chart)
        with col2:
            region_sales = sales_data.groupby('region')['sales'].sum().reset_index()
            fig2 = px.pie(region_sales, names='region', values='sales', title='🌍 Region-wise Sales')
            st.plotly_chart(fig2, use_container_width=True)

        # Spacer
        st.markdown("---")

        #  Top 10 Cities by Sales (Bar Chart)
        city_sales = sales_data.groupby('city')['sales'].sum().reset_index()
        top_cities = city_sales.sort_values(by='sales', ascending=False).head(10)
        fig3 = px.bar(top_cities, x='city', y='sales', title='🏙️ Top 10 Cities by Sales', color='sales')
        st.plotly_chart(fig3, use_container_width=True)

        # Top 40 Cities Countplot (Bar Chart)
        top_40_cities = sales_data['city'].value_counts().head(40).reset_index()
        top_40_cities.columns = ['city', 'count']
        fig4 = px.bar(top_40_cities, x='city', y='count', title='🌆 Top 40 Cities Countplot', color='count')
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("📊 Data Insights")

        insights_df = data.copy()

        # Drop rows with missing values in relevant columns
        insights_df = insights_df.dropna(subset=['age', 'annual income (k$)', 'spending score (1-100)', 'gender'])

        # Histogram of Age
        st.plotly_chart(
            px.histogram(insights_df, x='age', nbins=30, title='🎂 Age Distribution of Customers'),
            use_container_width=True
        )

        # Scatter plot: Annual Income vs Spending Score
        st.plotly_chart(
            px.scatter(
                insights_df, x='annual income (k$)', y='spending score (1-100)', color='gender',
                title='💸 Annual Income vs Spending Score by Gender',
                hover_data=['age']
            ),
            use_container_width=True
        )

        # Create two columns for side-by-side charts
        col1, col2 = st.columns(2)

        # Gender vs Average Spending Score (Bar Chart)
        with col1:
            gender_spend = insights_df.groupby('gender')['spending score (1-100)'].mean().reset_index()
            fig1 = px.bar(gender_spend, x='gender', y='spending score (1-100)',
                          title='👩‍🦰 Gender-wise Average Spending Score', color='gender')
            st.plotly_chart(fig1, use_container_width=True)

        # Top 10 Highest Income Customers (Bar Chart)
        with col2:
            top_income = insights_df.sort_values(by='annual income (k$)', ascending=False).head(10)
            fig2 = px.bar(top_income, x='annual income (k$)', y='cust_id', orientation='h',
                          title='💼 Top 10 Highest Income Customers', color='annual income (k$)')
            st.plotly_chart(fig2, use_container_width=True)

        # Create two more columns for side-by-side charts
        col3, col4 = st.columns(2)

        # Pie Chart: Top 10 Spending Customers
        with col3:
            top_spend = insights_df.sort_values(by='spending score (1-100)', ascending=False).head(10)
            fig3 = px.pie(top_spend, names='cust_id', values='spending score (1-100)',
                          title='🧁 Top 10 Spending Customers')
            st.plotly_chart(fig3, use_container_width=True)

        # Bar Chart: Age Groups vs Spending Score
        with col4:
            # Define age groups
            bins = [18, 25, 35, 45, 55, 65, 100]
            labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            insights_df['age_group'] = pd.cut(insights_df['age'], bins=bins, labels=labels, right=False)

            age_spending = insights_df.groupby('age_group')['spending score (1-100)'].mean().reset_index()
            fig4 = px.bar(age_spending, x='age_group', y='spending score (1-100)',
                          title='📊 Age Groups vs Average Spending Score', color='spending score (1-100)')
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.subheader("📦 Product Wise Data Analysis")

        # reload dataset
        df3 = pd.read_csv('sales.csv')

        # 1. Orders per Category
        st.markdown("### 🗂️ 1. Orders by Category")
        category_orders = df3['Category'].value_counts().reset_index()
        category_orders.columns = ['Category', 'Order Count']
        fig1 = px.bar(category_orders, x='Category', y='Order Count', color='Category',
                      title="Number of Orders per Category")
        st.plotly_chart(fig1)

        # 2. Top 10 Highest Selling Products
        st.markdown("### 🏆 2. Top 10 Highest Selling Products")
        top_products = df3.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(top_products, x=top_products.values, y=top_products.index, orientation='h',
                      title="Top 10 Best Selling Products", labels={'x': 'Sales', 'y': 'Product Name'})
        st.plotly_chart(fig2)

        # 3. Highest Grossing Categories (by Revenue)
        st.markdown("### 💰 3. Highest Grossing Categories (Revenue)")
        grossing = df3.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig3 = px.pie(grossing, names='Category', values='Sales', title="Revenue Distribution by Category")
        st.plotly_chart(fig3)

        # 4. Top Rated Categories (Estimated via Quantity or Count)
        st.markdown("### 🌟 4. Top Rated Categories (Quantity Sold or Order Volume)")
        if 'Quantity' in df3.columns:
            rated = df3.groupby("Category")["Quantity"].sum().sort_values(ascending=False).reset_index()
            quantity_label = "Quantity"
        else:
            rated = df3['Category'].value_counts().reset_index()
            rated.columns = ['Category', 'Quantity']
            quantity_label = "Estimated Quantity"
        fig4 = px.bar(rated, x='Category', y='Quantity', color='Category', title="Top Rated Categories",
                      labels={"Quantity": quantity_label})
        st.plotly_chart(fig4)

        # 5. Sub-Category Performance (Total Sales)
        st.markdown("### 🔍 5. Sub-Category Sales Performance")
        subcat_sales = df3.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True)
        fig5 = px.bar(subcat_sales, x=subcat_sales.values, y=subcat_sales.index, orientation='h',
                      title="Sales by Sub-Category", labels={'x': 'Sales', 'y': 'Sub-Category'})
        st.plotly_chart(fig5)

        # 6. Regional Product Sales (Category by Region Heatmap)
        st.markdown("### 🌍 6. Regional Product Demand (Category vs Region)")
        regional = df3.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
        pivot = regional.pivot(index='Region', columns='Category', values='Sales').fillna(0)
        fig6 = px.imshow(pivot, text_auto=True, title="Sales Heatmap: Category vs Region")
        st.plotly_chart(fig6)

    with tab5:
        st.subheader("🛍️ Mall Customer Analysis")

        # Re-load the original datasets
        df1 = pd.read_csv('Customer Data.csv')
        df2 = pd.read_csv('Mall_Customers.csv')

        # Standardize and clean
        df2.columns = df2.columns.str.lower().str.strip()
        df2.fillna(df2.mean(numeric_only=True), inplace=True)

        # Preview top 10 rows
        st.markdown("### 🔍 Top 10 Records from Mall Customer Data")
        st.dataframe(df2.head(10))

        # Preprocess for clustering
        X = df2[['age', 'annual income (k$)', 'spending score (1-100)']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit KMeans model
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df2['cluster'] = clusters

        # PCA for 2D Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df2['PCA1'] = X_pca[:, 0]
        df2['PCA2'] = X_pca[:, 1]

        # Plot 2D PCA with clusters
        st.markdown("### 🎨 2D PCA Visualization of Clusters")
        fig = px.scatter(
            df2,
            x='PCA1',
            y='PCA2',
            color=df2['cluster'].astype(str),
            title="Customer Segments (2D PCA)",
            labels={'color': 'Cluster'},
            template="plotly_dark",
            hover_data=['age', 'annual income (k$)', 'spending score (1-100)']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interactive form
        st.markdown("### 🧮 Income Wise Spending Cluster (Insert Your Details)")
        age_input = st.number_input("Age", min_value=10, max_value=100, value=30)
        income_input = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
        spending_input = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

        # Predict user cluster
        user_data = scaler.transform([[age_input, income_input, spending_input]])
        user_cluster = kmeans.predict(user_data)[0]

        st.markdown("### 📌 Your Cluster")
        st.write(f"You belong to **Cluster {user_cluster}**")

        # Cluster Meanings
        cluster_meanings = {
            0: "Cluster 0: Low Income, Low Spending",
            1: "Cluster 1: High Income, High Spending",
            2: "Cluster 2: High Income, Low Spending",
            3: "Cluster 3: Low Income, High Spending",
            4: "Cluster 4: Moderate Income, Moderate Spending",
        }

        st.markdown("### 🧠 Cluster Meanings")
        for key, value in cluster_meanings.items():
            st.write(value)

    with tab6:
        st.subheader("📉 Customer Segmentation (KMeans Clusters)")

        # Load and preprocess
        customer_df = pd.read_csv("Customer Data.csv")
        customer_df.columns = customer_df.columns.str.lower().str.strip()
        customer_df.dropna(inplace=True)

        # Select features for clustering
        features = ['purchases', 'credit_limit', 'payments']
        df_seg = customer_df[features]

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_seg)

        # Choose number of clusters
        k = st.slider("Select number of clusters (K)", 2, 10, 4)

        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        customer_df['Cluster'] = clusters
        st.session_state.df_clustered = customer_df.copy()

        # 📊 3D Cluster Visualization using Plotly
        fig = px.scatter_3d(
            customer_df,
            x='purchases',
            y='credit_limit',
            z='payments',
            color=customer_df['Cluster'].astype(str),
            title="📊 Customer Segmentation Clusters (3D)",
            labels={'Cluster': 'Cluster Group'},
            opacity=0.8
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Show Cluster Summary Count
        cluster_counts = customer_df['Cluster'].value_counts().sort_index()
        fig_cluster_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            title="🧮 Number of Customers in Each Cluster",
            color=cluster_counts.index.astype(str)
        )
        st.plotly_chart(fig_cluster_bar, use_container_width=True)

        # Show segmented data preview
        st.subheader("📄 Top 10 Customers by Cluster")
        st.dataframe(customer_df.head(10))

    with tab7:
        st.subheader("🛍️ Purchase Behavior Insights")

        if "df_clustered" not in st.session_state:
            st.warning("Please run Customer Segmentation in Tab 5 first.")
            st.stop()

        df = st.session_state.df_clustered

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                px.box(df, x="Cluster", y="purchases", color="Cluster", title="Purchases Distribution by Cluster"),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                px.box(df, x="Cluster", y="payments", color="Cluster", title="Payments Distribution by Cluster"),
                use_container_width=True
            )

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                px.histogram(df, x="purchases", color="Cluster", nbins=30, barmode="overlay",
                             title="Histogram of Purchases by Cluster"),
                use_container_width=True
            )

        with col4:
            st.plotly_chart(
                px.histogram(df, x="payments", color="Cluster", nbins=30, barmode="overlay",
                             title="Histogram of Payments by Cluster"),
                use_container_width=True
            )

        st.markdown("### 🧾 Average Purchase vs Payment by Cluster")
        avg_df = df.groupby("Cluster")[["purchases", "payments"]].mean().reset_index()
        st.plotly_chart(
            px.bar(avg_df, x="Cluster", y=["purchases", "payments"], barmode="group",
                   title="Average Purchases and Payments per Cluster"),
            use_container_width=True
        )

    with tab8:
        st.subheader("🚪 Churn Prediction Analysis")

        # Load mall dataset (df2)
        df2 = pd.read_csv('Mall_Customers.csv')

        # Preview the dataset
        st.write("Mall Customer Dataset:", df2.head())

        # Rename columns for clarity
        df = df2.rename(columns={
            'Annual Income (k$)': 'annual_income',
            'Spending Score (1-100)': 'spending_score',
            'Gender': 'gender',
            'Age': 'age'
        })

        # Add a synthetic churn column (for demonstration)
        # Let's assume people with low spending and high income churn
        df['churn'] = ((df['spending_score'] < 40) & (df['annual_income'] > 70)).astype(int)

        # Encode gender
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

        # Features & target
        features = ['age', 'annual_income', 'spending_score', 'gender']
        target = 'churn'

        X = df[features]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        with st.expander("📊 Churn Prediction Results"):
            st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
            st.text(cr)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        st.markdown("### 🔍 Predict Churn for a New Customer")

        # Inputs
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income (k$)", 0, 200, 60)
        spending = st.slider("Spending Score (1-100)", 1, 100, 50)
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        gender_encoded = 0 if gender_input == "Male" else 1

        user_input = pd.DataFrame([[age, income, spending, gender_encoded]], columns=features)
        user_scaled = scaler.transform(user_input)
        prediction = model.predict(user_scaled)

        if prediction[0] == 1:
            st.success("🚨 This customer is likely to churn.")
        else:
            st.info("✅ This customer is not likely to churn.")

        st.write("### 📊 Feature Importance")
        feat_importance = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': features, 'Importance': feat_importance})
        st.bar_chart(feat_df.set_index('Feature'))

    with tab9:
        st.subheader("🧠 Customer Preferences & Shopping Patterns")

        # reload dataset
        df3 = pd.read_csv('sales.csv')

        st.markdown("Analyzing customer behavior to drive strategic inventory and marketing decisions.")

        # Subcategory-wise analysis of sales, top products
        st.markdown("### 📊 Sales Insights by Subcategory")

        # Total sales by subcategory using Plotly bar chart
        subcategory_sales = df3.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        fig_sales = px.bar(subcategory_sales, x=subcategory_sales.index, y=subcategory_sales.values,
                           labels={'x': 'Subcategory', 'y': 'Total Sales'},
                           title='Total Sales by Subcategory')
        st.plotly_chart(fig_sales)

        # Top products in each subcategory using Plotly bar chart
        st.markdown("### 🏅 Top Products by Category")
        top_products_category = df3.groupby(["Category", "Product Name"])["Sales"].sum().sort_values(
            ascending=False).groupby("Category").head(5).reset_index()

        fig_top_products_cat = px.bar(top_products_category, x="Product Name", y="Sales", color="Category",
                                      title="Top 5 Products by Category",
                                      labels={"Sales": "Sales", "Product Name": "Product"})
        st.plotly_chart(fig_top_products_cat)

        # Subcategory-wise shipping mode preference (e.g., which shipping modes are most popular for different subcategories)
        st.markdown("### 🚚 Shipping Mode Preference by Subcategory")
        shipping_pref_subcategory = df3.groupby(['Sub-Category', 'Ship Mode']).size().unstack().fillna(0)
        fig_shipping_pref = go.Figure(
            data=[go.Bar(name=ship_mode, x=shipping_pref_subcategory.index, y=shipping_pref_subcategory[ship_mode])
                  for ship_mode in shipping_pref_subcategory.columns])
        fig_shipping_pref.update_layout(barmode='stack', title='Shipping Mode Preference by Subcategory',
                                        xaxis_title='Subcategory', yaxis_title='Count')
        st.plotly_chart(fig_shipping_pref)

        # Add meaning for each shipping mode
        st.markdown("### 📘 Shipping Mode Meanings")
        ship_meanings = {
            "First Class": "Faster delivery than standard, more expensive than second class",
            "Second Class": "Mid-tier shipping — slower than first class, but cheaper",
            "Standard Class": "Regular delivery time, cost-effective, most common",
            "Same Day": "Delivered the same day — fastest and most expensive option"
        }

        ship_df = pd.DataFrame(list(ship_meanings.items()), columns=["Ship Mode", "Meaning"])
        st.table(ship_df)


#
# # -------------------------
# # App Flow
# # -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Main App Logic
if not st.session_state.logged_in:
    choice = st.sidebar.radio("Choose an option", ["Login", "Signup"])
    if choice == "Login":
        login()
    else:
        signup()
else:
    dashboard()
