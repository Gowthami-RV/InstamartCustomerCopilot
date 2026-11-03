"""Part B: RFM segmentation and a leakage-aware lapse model."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

def rfm_and_segments():
    Path("outputs").mkdir(exist_ok=True)

    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Add product category to orders
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    completed = orders[
        orders["status"].str.lower() == "completed"
    ].copy()


    # =========================================================
    # TASK 1: BUILD RFM TABLE
    # =========================================================

    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum")
        )
        .reset_index()
    )

    rfm["recency"] = (
        snapshot - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    rfm.to_csv("data/clean/rfm.csv", index=False)

    print("\nTASK 1: RFM SUMMARY")
    print(rfm[["recency", "frequency", "monetary"]].describe())

    assert len(rfm) == 1175
    assert rfm["customer_id"].is_unique
    assert rfm["recency"].min() == 0
    assert rfm["recency"].max() == 180


    # =========================================================
    # TASK 2: K-MEANS CLUSTERING
    # =========================================================

    rfm_features = ["recency", "frequency", "monetary"]

    scaler_rfm = StandardScaler()
    X_scaled = scaler_rfm.fit_transform(rfm[rfm_features])

    evaluation = []

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        evaluation.append({
            "K": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X_scaled, labels)
        })

    evaluation_df = pd.DataFrame(evaluation)

    print("\nTASK 2: K-MEANS EVALUATION")
    print(evaluation_df.round(3))

    evaluation_df.to_csv(
        "outputs/kmeans_evaluation.csv",
        index=False
    )

    # Choose K=4 based on the elbow, silhouette and business usability
    FINAL_K = 4

    final_kmeans = KMeans(
        n_clusters=FINAL_K,
        n_init=10,
        random_state=42
    )

    rfm["segment_id"] = final_kmeans.fit_predict(X_scaled)


    # =========================================================
    # TASK 3: PROFILE AND NAME SEGMENTS
    # =========================================================

    profile = (
        rfm.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean")
        )
        .reset_index()
    )

    # Premium is always the segment with the highest mean monetary value
    premium_id = profile.loc[
        profile["mean_monetary"].idxmax(), "segment_id"
    ]

    other_ids = [
        sid for sid in profile["segment_id"]
        if sid != premium_id
    ]

    # Give the remaining segments descriptive business names
    remaining = profile[
        profile["segment_id"].isin(other_ids)
    ].copy()

    loyal_id = remaining.loc[
        remaining["mean_frequency"].idxmax(), "segment_id"
    ]

    remaining = remaining[
        remaining["segment_id"] != loyal_id
    ]

    at_risk_id = remaining.loc[
        remaining["mean_recency"].idxmax(), "segment_id"
    ]

    other_id = remaining.loc[
        remaining["segment_id"] != at_risk_id, "segment_id"
    ].iloc[0]

    segment_names = {
        premium_id: "Premium High-Value Customers",
        loyal_id: "Loyal Active Customers",
        at_risk_id: "At-Risk Customers",
        other_id: "Occasional Customers"
    }

    segment_actions = {
        premium_id: "Offer VIP rewards, exclusive deals and personalised benefits.",
        loyal_id: "Encourage repeat orders through loyalty points and subscriptions.",
        at_risk_id: "Run win-back campaigns with personalised offers.",
        other_id: "Use product recommendations and promotions to increase frequency."
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    profile["segment_name"] = profile["segment_id"].map(segment_names)
    profile["recommended_action"] = profile["segment_id"].map(segment_actions)
    profile["is_premium"] = profile["segment_id"].eq(premium_id)

    print("\nTASK 3: SEGMENT PROFILE")
    print(profile.round(2))

    profile.to_csv("outputs/segment_profile.csv", index=False)

    # PCA on the scaled RFM matrix
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    print("\nPCA EXPLAINED VARIANCE")
    print(pca.explained_variance_ratio_)
    print("Total:", pca.explained_variance_ratio_.sum())

    # Required segment output
    customer_segments = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium"
        ]
    ].copy()

    customer_segments.to_csv(
        "outputs/customer_segments.csv",
        index=False
    )
    return rfm,pd.DataFrame(evaluation),profile,pca.explained_variance_ratio_

def build_lapse_model():
    # =========================================================
    # TASK 4: DEFINE LAPSE PROBLEM
    # =========================================================
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    customer_segments_path = Path("outputs/customer_segments.csv")
    if not customer_segments_path.exists():
        raise FileNotFoundError(
            "Missing outputs/customer_segments.csv. Run rfm_and_segments() first."
        )
    customer_segments = pd.read_csv(customer_segments_path)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    completed = orders[
        orders["status"].str.lower() == "completed"
    ].copy()

    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    label_end = pd.Timestamp("2025-10-01 00:00:00")

    # Only orders strictly before cutoff are used for features
    before_cutoff = orders[
        orders["order_ts"] < cutoff
    ].copy()

    completed_before = before_cutoff[
        before_cutoff["status"].str.lower() == "completed"
    ].copy()

    # Cohort: customers with at least one completed order before cutoff
    cohort_ids = completed_before["customer_id"].unique()

    # Label window: September completed orders only
    september_completed = completed[
        (completed["order_ts"] >= cutoff)
        & (completed["order_ts"] < label_end)
    ]

    september_customer_ids = set(
        september_completed["customer_id"].unique()
    )

    labels = pd.DataFrame({"customer_id": cohort_ids})

    labels["lapsed"] = (
        ~labels["customer_id"].isin(september_customer_ids)
    ).astype(int)


    # =========================================================
    # TASK 4: CREATE PRE-CUTOFF FEATURES
    # =========================================================

    # Completed-order behaviour
    completed_features = (
        completed_before.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean")
        )
        .reset_index()
    )

    completed_features["recency"] = (
        cutoff - completed_features["last_order_ts"]
    ).dt.days

    # Most frequently purchased product category
    def favourite_category(series):
        modes = series.dropna().mode()
        if len(modes) == 0:
            return "Unknown"
        return modes.iloc[0]


    favourite = (
        completed_before.groupby("customer_id")["category"]
        .agg(favourite_category)
        .reset_index(name="favourite_category")
    )

    # Cancellation rate among all final-status orders before cutoff
    final_before = before_cutoff[
        before_cutoff["status"].str.lower().isin(
            ["completed", "cancelled"]
        )
    ].copy()

    final_before["is_cancelled"] = (
        final_before["status"].str.lower() == "cancelled"
    ).astype(int)

    cancellation = (
        final_before.groupby("customer_id")
        .agg(
            final_order_count=("order_id", "nunique"),
            cancelled_count=("is_cancelled", "sum")
        )
        .reset_index()
    )

    cancellation["cancellation_rate"] = (
        cancellation["cancelled_count"]
        / cancellation["final_order_count"]
    )

    # Combine all features with customer information
    features = (
        completed_features
        .merge(favourite, on="customer_id", how="left")
        .merge(
            cancellation[["customer_id", "cancellation_rate"]],
            on="customer_id",
            how="left"
        )
        .merge(
            customers[
                ["customer_id", "city", "age", "gender", "signup_date"]
            ],
            on="customer_id",
            how="left"
        )
    )

    features["tenure_days"] = (
        cutoff - features["signup_date"]
    ).dt.days

    # Remove timestamp columns that are not model features
    features = features.drop(
        columns=["last_order_ts", "signup_date"]
    )

    # Combine features and labels
    model_data = features.merge(
        labels,
        on="customer_id",
        how="inner"
    )

    # Do not include segment_id, segment_name or is_premium
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days"
    ]

    categorical_features = [
        "city",
        "gender",
        "favourite_category"
    ]

    feature_columns = numeric_features + categorical_features

    X = model_data[feature_columns].copy()
    y = model_data["lapsed"].copy()

    print("\nTASK 4: LAPSE COHORT")
    print("Cohort size:", len(model_data))
    print("Lapsed customers:", int(y.sum()))
    print("Lapsed percentage:", round(y.mean() * 100, 2), "%")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    assert len(model_data) == 1156
    assert int(y.sum()) == 443
    assert len(X_train) == 924
    assert len(X_test) == 232


    # =========================================================
    # TASK 5: PREPROCESS WITHOUT LEAKAGE
    # =========================================================

    # All imputers, encoders and scalers are inside the pipeline.
    # GridSearchCV fits them only on each training fold.

    def make_preprocessor():

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ))
        ])

        return ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])


    # =========================================================
    # TASK 6A: BASELINE
    # =========================================================

    baseline_predictions = np.zeros(len(y_test), dtype=int)

    print("\nTASK 6A: BASELINE")
    print("Accuracy:", accuracy_score(y_test, baseline_predictions))
    print(
        "Lapsed F1:",
        f1_score(y_test, baseline_predictions, zero_division=0)
    )


    # =========================================================
    # TASK 6B: TUNE LOGISTIC REGRESSION
    # =========================================================

    logistic_pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_grid = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    logistic_grid.fit(X_train, y_train)

    best_logistic = logistic_grid.best_estimator_

    print("\nLOGISTIC REGRESSION")
    print("Best parameters:", logistic_grid.best_params_)
    print("Best CV F1:", logistic_grid.best_score_)


    # =========================================================
    # TASK 6C: TUNE RANDOM FOREST
    # =========================================================

    rf_pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_

    print("\nRANDOM FOREST")
    print("Best parameters:", rf_grid.best_params_)
    print("Best CV F1:", rf_grid.best_score_)


    # =========================================================
    # TASK 6D: TEST EVALUATION
    # =========================================================

    def evaluate_model(name, model):

        predictions = model.predict(X_test)

        print(f"\n{name}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Precision:", precision_score(
            y_test, predictions, zero_division=0
        ))
        print("Recall:", recall_score(
            y_test, predictions, zero_division=0
        ))
        print("F1:", f1_score(
            y_test, predictions, zero_division=0
        ))

        return predictions


    logistic_predictions = evaluate_model(
        "TUNED LOGISTIC REGRESSION",
        best_logistic
    )

    rf_predictions = evaluate_model(
        "TUNED RANDOM FOREST",
        best_rf
    )

    logistic_test_f1 = f1_score(
        y_test, logistic_predictions, zero_division=0
    )

    rf_test_f1 = f1_score(
        y_test, rf_predictions, zero_division=0
    )

    # Select the model with the better test F1, after tuning is complete
    selected_model = (
        best_rf if rf_test_f1 >= logistic_test_f1
        else best_logistic
    )

    selected_name = (
        "Random Forest" if selected_model is best_rf
        else "Logistic Regression"
    )

    print("\nSELECTED MODEL:", selected_name)

    assert max(logistic_test_f1, rf_test_f1) >= 0.55


    # =========================================================
    # TASK 7: CLASS-BALANCE EXPERIMENT
    # =========================================================

    # Keep all original training rows.
    # Add replacement samples from the lapsed training class.

    train_data = X_train.copy()
    train_data["lapsed"] = y_train.values

    not_lapsed_train = train_data[
        train_data["lapsed"] == 0
    ]

    lapsed_train = train_data[
        train_data["lapsed"] == 1
    ]

    extra_lapsed = resample(
        lapsed_train,
        replace=True,
        n_samples=len(not_lapsed_train) - len(lapsed_train),
        random_state=42
    )

    balanced_train = pd.concat(
        [train_data, extra_lapsed],
        ignore_index=True
    )

    X_balanced = balanced_train[feature_columns]
    y_balanced = balanced_train["lapsed"]

    print("\nBALANCED TRAINING COUNTS")
    print(y_balanced.value_counts())

    # Refit best logistic regression configuration on balanced data
    balanced_logistic = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(
            max_iter=1000,
            C=logistic_grid.best_params_["model__C"]
        ))
    ])

    balanced_logistic.fit(X_balanced, y_balanced)

    balanced_predictions = balanced_logistic.predict(X_test)

    print("\nCLASS BALANCE COMPARISON")

    print("Original logistic:")
    print("Precision:", precision_score(
        y_test, logistic_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, logistic_predictions, zero_division=0
    ))
    print("F1:", f1_score(
        y_test, logistic_predictions, zero_division=0
    ))

    print("\nBalanced logistic:")
    print("Precision:", precision_score(
        y_test, balanced_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, balanced_predictions, zero_division=0
    ))
    print("F1:", f1_score(
        y_test, balanced_predictions, zero_division=0
    ))


    # =========================================================
    # TASK 8A: FEATURE IMPORTANCE AND COEFFICIENTS
    # =========================================================

    # Random forest feature importance
    rf_preprocessor = best_rf.named_steps["preprocess"]

    rf_feature_names = rf_preprocessor.get_feature_names_out()
    rf_importances = best_rf.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": rf_feature_names,
        "importance": rf_importances
    }).sort_values(
        "importance", ascending=False
    )

    print("\nTOP 5 RANDOM FOREST FEATURES")
    print(importance_df.head(5))

    importance_df.to_csv(
        "outputs/rf_feature_importance.csv",
        index=False
    )

    # Logistic regression coefficients
    log_preprocessor = best_logistic.named_steps["preprocess"]

    log_feature_names = log_preprocessor.get_feature_names_out()
    log_coefficients = best_logistic.named_steps["model"].coef_[0]

    coefficient_df = pd.DataFrame({
        "feature": log_feature_names,
        "coefficient": log_coefficients
    })

    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] > 0,
        "Increases lapse likelihood",
        "Decreases lapse likelihood"
    )

    coefficient_df["absolute_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient", ascending=False
    )

    print("\nLOGISTIC REGRESSION COEFFICIENTS")
    print(coefficient_df.head(10))

    coefficient_df.to_csv(
        "outputs/logistic_coefficients.csv",
        index=False
    )


    # =========================================================
    # TASK 8B: SAVE AND RELOAD SELECTED MODEL
    # =========================================================

    joblib.dump(
        selected_model,
        "outputs/lapse_model.joblib"
    )

    reloaded_model = joblib.load(
        "outputs/lapse_model.joblib"
    )

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions,
        reloaded_predictions
    )

    print("\nModel saved and reloaded successfully.")
    print("Reloaded predictions match original predictions.")


    # =========================================================
    # TASK 8C: PREMIUM RETENTION LIST
    # =========================================================

    # Use only test customers for this list.
    # Premium is joined after predictions; it is not a model feature.

    test_results = pd.DataFrame({
        "customer_id": model_data.loc[
            X_test.index, "customer_id"
        ].values,
        "actually_lapsed": y_test.values,
        "lapse_probability": selected_model.predict_proba(
            X_test
        )[:, list(selected_model.classes_).index(1)]
    })

    premium_customers = customer_segments[
        ["customer_id", "is_premium"]
    ].copy()

    test_results = test_results.merge(
        premium_customers,
        on="customer_id",
        how="left"
    )

    test_results["is_premium"] = (
        test_results["is_premium"].fillna(False).astype(bool)
    )

    retention_list = (
        test_results[test_results["is_premium"]]
        .sort_values("lapse_probability", ascending=False)
        .head(10)
    )

    retention_list = retention_list[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed"
        ]
    ]

    retention_list.to_csv(
        "outputs/retention_list.csv",
        index=False
    )

    print("\nPREMIUM RETENTION LIST")
    print(retention_list)

    print(
        "Number of listed customers who actually lapsed:",
        int(retention_list["actually_lapsed"].sum())
    )






