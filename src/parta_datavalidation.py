from pathlib import Path
import re
import pandas as pd
import sqlite3
import re
import numpy as np

# Folder containing the original CSV files
RAW = Path("data/raw")


def audit_data():
    # Load the raw tables
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    # Important: preserve empty ticket text as an empty string
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

    tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "tickets": tickets,
    }

    # --------------------------------------------------
    # 1. Shape, data types and missing-value audit
    # --------------------------------------------------

    for name, df in tables.items():
        print(f"\n{'=' * 50}")
        print(f"TABLE: {name.upper()}")
        print(f"{'=' * 50}")

        print("\nShape (rows, columns):")
        print(df.shape)

        print("\nData types:")
        print(df.dtypes)

        # pandas isna() counts NaN/None, not empty strings
        missing = pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_percentage": (df.isna().mean() * 100).round(2),
            }
        )

        print("\nMissing values by column:")
        print(missing.to_string())

    # --------------------------------------------------
    # 2. Audit figures for orders.csv
    # --------------------------------------------------

    # Convert amounts temporarily to numeric for checking.
    # Invalid entries become NaN; the raw DataFrame is unchanged.
    amount_text = orders["order_amount"].astype("string").str.strip()
    numeric_amount = pd.to_numeric(orders["order_amount"], errors="coerce")

    non_numeric_amounts = (
        numeric_amount.isna() & amount_text.notna() & amount_text.ne("")
    ).sum()

    delivery_text = orders["delivery_minutes"].astype("string").str.strip()

    numeric_delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    negative_delivery = (numeric_delivery < 0).sum()

    blank_delivery = (delivery_text.eq("") | delivery_text.isna()).sum()

    # --------------------------------------------------
    # 3. Audit figures for customers.csv
    # --------------------------------------------------

    age_text = customers["age"].astype("string").str.strip()
    numeric_age = pd.to_numeric(customers["age"], errors="coerce")

    blank_age = (age_text.eq("") | age_text.isna()).sum()

    age_outside_range = (numeric_age.notna() & ~numeric_age.between(10, 100)).sum()

    # Keep original spellings when counting city variations
    distinct_city_spellings = customers["city"].nunique(dropna=True)

    gender_text = customers["gender"].astype("string").str.strip()

    blank_gender = (gender_text.eq("") | gender_text.isna()).sum()

    distinct_gender_spellings = customers.loc[
        gender_text.notna() & gender_text.ne(""), "gender"
    ].nunique()

    # --------------------------------------------------
    # 4. Print the requested audit summary
    # --------------------------------------------------

    audit_results = pd.DataFrame(
        {
            "Audit measure": [
                "Raw orders rows",
                "Exact duplicate order rows",
                "Pending order rows",
                "Distinct order_id values",
                "Non-numeric order_amount entries",
                "Negative delivery_minutes",
                "Blank delivery_minutes",
                "Blank customer ages",
                "Ages outside 10-100",
                "Distinct raw city spellings",
                "Distinct non-blank raw gender spellings",
                "Blank customer genders",
            ],
            "Observed value": [
                len(orders),
                orders.duplicated().sum(),
                orders["status"].eq("pending").sum(),
                orders["order_id"].nunique(),
                int(non_numeric_amounts),
                int(negative_delivery),
                int(blank_delivery),
                int(blank_age),
                int(age_outside_range),
                distinct_city_spellings,
                distinct_gender_spellings,
                int(blank_gender),
            ],
        }
    )

    print("\n\nRAW DATA AUDIT SUMMARY")
    print(audit_results.to_string(index=False))

    # Save the summary for reference
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    audit_results.to_csv(output_dir / "raw_audit_summary.csv", index=False)

    print(f"\nAudit summary saved to {output_dir / 'raw_audit_summary.csv'}")


def clean_data():
    # 1. Load the original customer data
    customers = pd.read_csv("data/raw/customers.csv")

    print("Original shape:", customers.shape)

    # 2. Standardise city names
    customers["city"] = customers["city"].astype("string").str.strip().str.title()

    # 3. Standardise gender values
    gender = customers["gender"].astype("string").str.strip().str.lower()

    gender_mapping = {"m": "male", "male": "male", "f": "female", "female": "female"}

    customers["gender"] = gender.map(gender_mapping).fillna("unknown")

    # 4. Convert age to numeric
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

    # Treat impossible ages as missing.
    # The supplied reference expects all final ages to be 18–60.
    customers.loc[~customers["age"].between(18, 60), "age"] = float("nan")

    # Calculate each city's median age using groupby + transform
    city_median_age = customers.groupby("city")["age"].transform("median")

    # Fill all missing ages with the median of their city
    customers["age"] = customers["age"].fillna(city_median_age)

    # 5. Convert signup_date to datetime
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")

    # 6. Validate the cleaned data
    print("\nCity counts:")
    print(customers["city"].value_counts().sort_values(ascending=False))

    print("\nGender counts:")
    print(customers["gender"].value_counts())

    print("\nMissing ages:", customers["age"].isna().sum())
    print("Age range:", customers["age"].min(), "to", customers["age"].max())
    print("Missing signup dates:", customers["signup_date"].isna().sum())

    # 7. Save the cleaned customer data
    output_dir = Path("data/clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    customers.to_csv(output_dir / "customers.csv", index=False, date_format="%Y-%m-%d")

    print("\nCleaned customers saved to data/clean/customers.csv")

    # --------------------------------------------------
    # 1. Load raw orders and products
    # --------------------------------------------------
    orders = pd.read_csv("data/raw/orders.csv")
    products = pd.read_csv("data/raw/products.csv")

    print("Original orders shape:", orders.shape)

    # --------------------------------------------------
    # 2. Convert order_ts to datetime
    # --------------------------------------------------
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

    # --------------------------------------------------
    # 3. Remove exact duplicate rows
    # --------------------------------------------------
    before_duplicates = len(orders)
    orders = orders.drop_duplicates()

    duplicates_removed = before_duplicates - len(orders)

    print("Exact duplicate rows removed:", duplicates_removed)

    # --------------------------------------------------
    # 4. Keep the latest record for each order_id
    # --------------------------------------------------
    before_latest = len(orders)

    # Put invalid/missing timestamps first so valid timestamps
    # are preferred when selecting the latest record.
    orders = orders.sort_values("order_ts", na_position="first")

    orders = orders.drop_duplicates(subset="order_id", keep="last")

    print("Repeated order records removed:", before_latest - len(orders))

    # --------------------------------------------------
    # 5. Convert order_amount to numeric
    # --------------------------------------------------
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

    # Add product price and category to each order.
    orders = orders.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Calculate the amount for rows where it is missing.
    calculated_amount = (
        orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

    # --------------------------------------------------
    # 6. Convert delivery_minutes to numeric
    # --------------------------------------------------
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Negative delivery times are invalid, so make them missing.
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

    # --------------------------------------------------
    # 7. Fill missing delivery times for completed orders only
    # --------------------------------------------------
    completed = orders["status"].eq("completed")

    # Calculate each completed order's product-category median.
    category_median = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    # Fill missing completed-order delivery times using the
    # corresponding category median.
    completed_delivery = orders.loc[completed, "delivery_minutes"]

    orders.loc[completed, "delivery_minutes"] = completed_delivery.fillna(
        category_median
    )

    # Do not fill delivery times for cancelled orders.
    cancelled = orders["status"].eq("cancelled")

    orders.loc[cancelled, "delivery_minutes"] = float("nan")

    # --------------------------------------------------
    # 8. Remove helper columns and save cleaned orders
    # --------------------------------------------------
    orders = orders.drop(columns=["price_inr", "category"])

    output_dir = Path("data/clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    orders.to_csv(
        output_dir / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------
    # 9. Validate the cleaned data
    # --------------------------------------------------
    print("\n--- Cleaned Orders Summary ---")
    print("Final shape:", orders.shape)
    print("\nOrders by status:")
    print(orders["status"].value_counts())

    print("\nDistinct order IDs:", orders["order_id"].nunique())
    print("Duplicate order IDs:", orders["order_id"].duplicated().sum())
    print("Missing order amounts:", orders["order_amount"].isna().sum())

    print(
        "Missing delivery times among completed orders:",
        orders.loc[completed, "delivery_minutes"].isna().sum(),
    )

    print(
        "Missing delivery times among cancelled orders:",
        orders.loc[cancelled, "delivery_minutes"].isna().sum(),
    )

    print(
        "Completed-order revenue:",
        round(orders.loc[completed, "order_amount"].sum(), 2),
    )

    print("\nCleaned orders saved to data/clean/orders.csv")

    # 1. Load raw products
    products = pd.read_csv("data/raw/products.csv")

    print("Original shape:", products.shape)
    print("\nOriginal data types:")
    print(products.dtypes)

    # 2. Remove exact duplicate rows
    before = len(products)
    products = products.drop_duplicates()
    print("\nExact duplicate rows removed:", before - len(products))

    # 3. Clean text columns
    text_columns = ["product_id", "product_name", "category"]

    for column in text_columns:
        if column in products.columns:
            products[column] = products[column].astype("string").str.strip()

    # Standardize product names and categories
    if "product_name" in products.columns:
        products["product_name"] = products["product_name"].str.title()

    if "category" in products.columns:
        products["category"] = products["category"].str.strip().str.title()

    # 4. Convert price to numeric
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")

    # 5. Check missing values and invalid prices
    print("\nMissing values before validation:")
    print(products.isna().sum())

    invalid_price = products["price_inr"].isna() | (products["price_inr"] <= 0)

    print("\nRows with missing or non-positive prices:")
    print(products.loc[invalid_price])

    # 6. Remove rows with missing essential values or invalid prices
    products = products.dropna(
        subset=["product_id", "product_name", "category", "price_inr"]
    )

    products = products.loc[products["price_inr"] > 0].copy()

    # 7. Ensure product_id is unique
    duplicate_ids = products["product_id"].duplicated(keep=False)

    if duplicate_ids.any():
        print("\nWarning: duplicate product IDs found:")
        print(products.loc[duplicate_ids].sort_values("product_id"))

        # Keep the first record only if duplicate IDs remain.
        products = products.drop_duplicates(subset="product_id", keep="first")

    # 8. Round prices to two decimal places
    products["price_inr"] = products["price_inr"].round(2)

    # 9. Reset index
    products = products.reset_index(drop=True)

    # 10. Save cleaned products
    output_dir = Path("data/clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    products.to_csv(output_dir / "products.csv", index=False)

    # 11. Final summary
    print("\n--- Cleaned Products Summary ---")
    print("Final shape:", products.shape)
    print("Unique product IDs:", products["product_id"].nunique())
    print("Missing values:")
    print(products.isna().sum())
    print("\nProducts by category:")
    print(products["category"].value_counts())
    print("\nCleaned products saved to data/clean/products.csv")


def validate_data():

    # 1. Load cleaned data

    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Read the cleaned CSV files
    customers = pd.read_csv(clean_dir / "customers.csv")
    products = pd.read_csv(clean_dir / "products.csv")
    orders = pd.read_csv(clean_dir / "orders.csv")

    # Store order_ts as text in YYYY-MM-DD HH:MM:SS format.
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure the three clean CSV files exist.
    customers.to_csv(clean_dir / "customers.csv", index=False)
    products.to_csv(clean_dir / "products.csv", index=False)
    orders.to_csv(clean_dir / "orders.csv", index=False)

    # Write all three tables to SQLite.
    db_path = data_dir / "instamart.db"

    with sqlite3.connect(db_path) as connection:
        customers.to_sql("customers", connection, if_exists="replace", index=False)
        products.to_sql("products", connection, if_exists="replace", index=False)
        orders.to_sql("orders", connection, if_exists="replace", index=False)

    print("Saved clean CSV files to:", clean_dir)
    print("Saved SQLite database to:", db_path)
    print("\n--- Data Validation ---")
    print("Customers shape:", customers.shape)
    print("Orders shape:", orders.shape)
    print("Products shape:", products.shape)

    # Assertion of data
    db_path = Path("data/instamart.db")

    # Load the clean CSV files.
    customers = pd.read_csv(clean_dir / "customers.csv")
    products = pd.read_csv(clean_dir / "products.csv")
    orders = pd.read_csv(clean_dir / "orders.csv")

    # Convert columns used in the checks to numeric values.
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # ---------------- Customer checkpoints ----------------

    assert len(customers) == 1200

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    assert customers["city"].value_counts().to_dict() == expected_cities

    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    assert customers["gender"].value_counts().to_dict() == expected_genders

    assert customers["age"].isna().sum() == 0
    assert customers["age"].between(18, 60).all()

    # ---------------- Order checkpoints ----------------

    assert len(orders) == 12309
    assert orders["order_id"].nunique() == 12309
    assert orders["order_id"].duplicated().sum() == 0

    expected_statuses = {
        "completed": 11110,
        "cancelled": 1199,
    }
    assert orders["status"].value_counts().to_dict() == expected_statuses

    assert orders["order_amount"].isna().sum() == 0

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")

    assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0
    assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199

    completed_revenue = round(orders.loc[completed, "order_amount"].sum(), 2)
    assert (
        completed_revenue == 2998668.80
    ), f"Unexpected completed-order revenue: {completed_revenue}"

    # ---------------- SQLite checkpoints ----------------

    assert db_path.exists(), "SQLite database does not exist"

    with sqlite3.connect(db_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        assert {"customers", "products", "orders"}.issubset(table_names)

        assert pd.read_sql_query("SELECT * FROM customers", connection).shape[0] == 1200

        assert pd.read_sql_query("SELECT * FROM products", connection).shape[0] == len(
            products
        )

        db_orders = pd.read_sql_query("SELECT * FROM orders", connection)

        assert len(db_orders) == 12309

        # Check that timestamps are stored as text in the requested format.
        timestamp_type = connection.execute(
            "SELECT typeof(order_ts) FROM orders " "WHERE order_ts IS NOT NULL LIMIT 1"
        ).fetchone()

        assert timestamp_type is not None
        assert timestamp_type[0] == "text"

        timestamp_sample = connection.execute(
            "SELECT order_ts FROM orders " "WHERE order_ts IS NOT NULL LIMIT 1"
        ).fetchone()[0]

        assert len(timestamp_sample) == 19
        assert timestamp_sample[4] == "-"
        assert timestamp_sample[7] == "-"
        assert timestamp_sample[10] == " "
        assert timestamp_sample[13] == ":"
        assert timestamp_sample[16] == ":"

    # Print this line only if every assertion passed.
    print("\nALL CHECKPOINTS PASSED")
    run_sql_analysis()
def run_sql_analysis():
    db_path = Path("data/instamart.db")
    sql_path = Path("sql/queries.sql")

    # Read the SQL file and split it into Q1–Q7 sections.
    sql_text = sql_path.read_text(encoding="utf-8")

    sections = re.split(r"(?m)^-- Q[1-7]:.*$", sql_text)
    headers = re.findall(r"(?m)^-- (Q[1-7]):.*$", sql_text)

    queries = {}
    for header, section in zip(headers, sections[1:]):
        query = section.strip()
        if query:
            queries[header] = query

    if len(queries) != 7:
        raise ValueError(f"Expected 7 SQL queries, but found {len(queries)}")

    with sqlite3.connect(db_path) as connection:
        results = {}

        for name, query in queries.items():
            df = pd.read_sql_query(query, connection)
            results[name] = df

            print(f"\n--- {name} ---")
            print("Rows:", len(df))
            print(df.head(10).to_string(index=False))

    # ---------------- Reference checks ----------------

    q1 = results["Q1"]
    q2 = results["Q2"]
    q3 = results["Q3"]
    q4 = results["Q4"]
    q5 = results["Q5"]
    q6 = results["Q6"]
    q7 = results["Q7"]

    assert len(q1) == 30, f"Q1 expected 30, got {len(q1)}"
    assert len(q2) == 30, f"Q2 expected 30, got {len(q2)}"
    assert round(q2["revenue"].sum(), 2) == 2998668.80
    assert len(q3) == 11, f"Q3 expected 11, got {len(q3)}"
    assert len(q4) == 25, f"Q4 expected 25, got {len(q4)}"
    assert len(q5) == 550, f"Q5 expected 550, got {len(q5)}"
    assert round(q5["average_spend"].iloc[0], 2) == 2552.06
    assert (q6["spend_rank"] != q6["spend_row_number"]).sum() == 8
    assert len(q7) == 183, f"Q7 expected 183, got {len(q7)}"
    assert round(q7["cumulative_revenue"].iloc[-1], 2) == 2998668.80

    # Print the largest city/category revenue cell.
    largest_cell = q2.loc[q2["revenue"].idxmax()]
    print("\nLargest revenue cell:")
    print(largest_cell.to_dict())

    # Print the first-ranked customer in each city.
    top_customers = q6[q6["spend_rank"] == 1][
        ["city", "customer_id", "total_spend"]
    ].sort_values("city")
    print("\nTop customer in each city:")
    print(top_customers.to_string(index=False))

    # Save query outputs for inspection.
    output_dir = Path("outputs/sql")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        df.to_csv(output_dir / f"{name.lower()}_results.csv", index=False)

    print("\nAll seven SQL queries ran and reference checks passed.")


def pandas_crosscheck_eda():
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")

    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Keep completed orders only for revenue and EDA.
    completed = orders.loc[orders["status"].eq("completed")].copy()

    # Add city and category for the cross-check and EDA.
    completed = completed.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    completed = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------
    # 2. Part A: Pivot table with grand totals
    # --------------------------------------------------
    pivot = pd.pivot_table(
        completed,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        margins=True,
        margins_name="All",
        fill_value=0,
    )

    print("\n--- Q2 Pandas Pivot Table (INR) ---")
    print(pivot.round(2).to_string())

    pivot_total = round(float(pivot.loc["All", "All"]), 2)
    print("\nPivot grand total:", pivot_total)

    # Compare with SQL Q2 total from SQLite.
    with sqlite3.connect("data/instamart.db") as connection:
        sql_total = pd.read_sql_query(
            """
            SELECT SUM(order_amount) AS total_revenue
            FROM orders
            WHERE status = 'completed'
            """,
            connection,
        ).iloc[0]["total_revenue"]

    sql_total = round(float(sql_total), 2)

    print("SQL grand total:", sql_total)
    print("Totals match:", pivot_total == sql_total)

    assert pivot_total == sql_total

    # --------------------------------------------------
    # 3. City-level sum, mean, and count
    # --------------------------------------------------
    city_summary = (
        completed.groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            order_count=("order_amount", "count"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
    )

    print("\n--- Completed Orders by City ---")
    print(city_summary.to_string())

    # --------------------------------------------------
    # 4. Part B: Descriptive statistics
    # --------------------------------------------------
    amount = completed["order_amount"].dropna()

    mean_amount = amount.mean()
    median_amount = amount.median()
    skewness = amount.skew()

    q1 = amount.quantile(0.25)
    q2 = amount.quantile(0.50)
    q3 = amount.quantile(0.75)

    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    below_fence = int((amount < lower_fence).sum())
    above_fence = int((amount > upper_fence).sum())

    print("\n--- Order Amount Statistics (Completed Orders) ---")
    print("Count:", len(amount))
    print(f"Mean: INR {mean_amount:.2f}")
    print(f"Median: INR {median_amount:.2f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Q1 (25th percentile): INR {q1:.2f}")
    print(f"Q2 (50th percentile): INR {q2:.2f}")
    print(f"Q3 (75th percentile): INR {q3:.2f}")
    print(f"IQR: INR {iqr:.2f}")
    print(f"Lower fence: INR {lower_fence:.2f}")
    print(f"Upper fence: INR {upper_fence:.2f}")
    print("Outliers below lower fence:", below_fence)
    print("Outliers above upper fence:", above_fence)

    # --------------------------------------------------
    # 5. Correlation matrix
    # --------------------------------------------------
    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation = completed[correlation_columns].corr()

    print("\n--- Correlation Matrix ---")
    print(correlation.round(4).to_string())

    # Find the pair with the largest absolute off-diagonal correlation.
    absolute_corr = correlation.abs().where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )

    largest_pair = absolute_corr.stack().idxmax()
    largest_value = correlation.loc[largest_pair[0], largest_pair[1]]

    print(
        "\nLargest absolute correlation:",
        f"{largest_pair[0]} and {largest_pair[1]} = {largest_value:.4f}",
    )


if __name__ == "__main__":
    audit_data()
    clean_data()
    validate_data()
    pandas_crosscheck_eda()
