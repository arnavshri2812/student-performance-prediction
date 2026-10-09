import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "student_performance.csv"
FEATURES = ["Attendance", "Internal_Marks", "Assignment_Marks", "Study_Hours", "Previous_Marks"]
CLASSES = ["Poor", "Average", "Good", "Excellent"]

def load_data():
    df = pd.read_csv(DATA_PATH)
    needed = FEATURES + ["Performance"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    df = df[needed].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Performance"] = df["Performance"].astype(str).str.strip()
    df = df.dropna(subset=["Performance"])
    df = df[df["Performance"].isin(CLASSES)]
    return df

def train_python_model(df):
    # A transparent local baseline. This is a CART-style decision tree,
    # not WEKA's J48/C4.5 implementation.
    X = df[FEATURES]
    y = df["Performance"]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model

def try_weka_prediction(values, weka_jar, arff_path):
    """
    Optional genuine WEKA route. Requires a WEKA jar and a WEKA-compatible
    ARFF dataset whose final attribute is Performance. For robust beginner
    use, the website defaults to its local Python tree unless configured.
    """
    if not weka_jar or not Path(weka_jar).exists():
        raise RuntimeError("WEKA jar path was not found.")
    if not arff_path or not Path(arff_path).exists():
        raise RuntimeError("WEKA ARFF dataset path was not found.")
    # WEKA CLI prediction input is a separate ARFF row with matching attributes.
    # We generate a temporary test ARFF with class marked missing.
    header = []
    in_data = False
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().lower() == "@data":
                in_data = True
                break
            header.append(line.rstrip("\n"))
    if not header or not in_data:
        raise RuntimeError("The ARFF file does not contain a valid @data section.")
    row = ",".join(str(values[k]) for k in FEATURES) + ",?"
    # WEKA test file must retain all attributes. This works with the supplied
    # numeric-feature dataset structure and nominal Performance class.
    with tempfile.TemporaryDirectory() as td:
        test_file = Path(td) / "test.arff"
        model_file = Path(td) / "j48.model"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n\n@data\n" + row + "\n")

        java_exe = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
        if not Path(java_exe).exists():
            java_exe = r"C:\Program Files\Weka-3-8\jre\bin\java.exe"
        if not Path(java_exe).exists():
            java_exe = "java"

        train_cmd = [
    r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "-cp", str(Path(weka_jar).resolve()),
    "weka.classifiers.trees.J48",
    "-t", str(Path(arff_path).resolve()),
    "-c", "last",
    "-d", str(model_file)
]
        trained = subprocess.run(train_cmd, capture_output=True, text=True, timeout=60)
        if trained.returncode != 0 or not model_file.exists():
            raise RuntimeError("WEKA training failed: " + (trained.stderr or trained.stdout)[-1200:])
        pred_cmd = [
    r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "-cp", str(Path(weka_jar).resolve()),
    "weka.classifiers.trees.J48",
    "-l", str(model_file),
    "-T", str(test_file),
    "-c", "last",
    "-p", "0"
]
        pred = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=60)
        if pred.returncode != 0:
            raise RuntimeError("WEKA prediction failed: " + (pred.stderr or pred.stdout)[-1200:])
        # WEKA's prediction output commonly has a line containing "1:label".
        for line in pred.stdout.splitlines():
            stripped = line.strip()
            if ":" in stripped and any(c in stripped for c in CLASSES):
                for label in CLASSES:
                    if label in stripped:
                        return label, pred.stdout
        raise RuntimeError("Could not parse WEKA prediction output. See the WEKA output below.")

try:
    df = load_data()
    model = train_python_model(df)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

st.sidebar.title("🎓 Student Predictor")
page = st.sidebar.radio("Navigate", ["Home & Predict", "Analytics", "Model Evaluation", "About"])
st.sidebar.caption("Educational mini project • Sample data")

if page == "Home & Predict":
    st.title("🎓 Student Academic Performance Predictor")
    st.write("Enter academic indicators to get a model-generated performance category.")
    st.info("This starter version uses a local Python decision tree. To use genuine WEKA J48 prediction, configure the optional WEKA settings below. Predictions are for demonstration only.")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Student details")
        with st.form("student_form"):
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            internal = st.slider("Internal marks (out of 100)", 0, 100, 70)
            assignment = st.slider("Assignment marks (out of 100)", 0, 100, 75)
            hours = st.slider("Average study hours per day", 0.0, 12.0, 3.0, 0.5)
            previous = st.slider("Previous academic marks (out of 100)", 0, 100, 70)
            use_weka = st.checkbox("Use WEKA J48 instead (requires Java, WEKA JAR and ARFF file)")
            weka_jar = st.text_input("Path to weka.jar", placeholder=r"C:\Program Files\Weka-3-8\weka.jar", disabled=not use_weka)
            arff_path = st.text_input("Path to student_performance.arff", placeholder=r"C:\project\student_performance.arff", disabled=not use_weka)
            submitted = st.form_submit_button("Predict performance", type="primary", use_container_width=True)
    with right:
        st.subheader("Prediction result")
        if submitted:
            values = {
                "Attendance": attendance,
                "Internal_Marks": internal,
                "Assignment_Marks": assignment,
                "Study_Hours": hours,
                "Previous_Marks": previous,
            }
            if use_weka:
                try:
                    prediction, raw = try_weka_prediction(values, weka_jar, arff_path)
                    st.success(f"Predicted category: {prediction}")
                    st.caption("Prediction produced by WEKA J48.")
                    with st.expander("View WEKA command output"):
                        st.code(raw)
                except Exception as e:
                    st.error(f"WEKA mode could not run: {e}")
                    st.warning("Try the local Python model by unticking the WEKA option, or check the setup guide.")
            else:
                sample = pd.DataFrame([values], columns=FEATURES)
                prediction = model.predict(sample)[0]
                st.success(f"Predicted category: {prediction}")
                st.caption("Prediction produced by the local Python decision tree.")
            st.markdown("**Input summary**")
            st.dataframe(pd.DataFrame([values]).T.rename(columns={0: "Value"}), use_container_width=True)
            st.warning("This result is a learning aid, not a final judgment about a student. Confirm concerns with a teacher and consider context not represented in the data.")
        else:
            st.markdown("### Ready to predict")
            st.write("Adjust the inputs and select **Predict performance**.")
            st.metric("Records in sample dataset", len(df))
            st.metric("Performance categories", df["Performance"].nunique())
            st.caption("The included dataset is synthetic sample data, not real student records.")

elif page == "Analytics":
    st.title("📊 Dataset Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample records", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Categories", df["Performance"].nunique())
    st.subheader("Performance category distribution")
    counts = df["Performance"].value_counts().reindex(CLASSES, fill_value=0)
    st.bar_chart(counts)
    st.subheader("Attendance versus internal marks")
    st.scatter_chart(df, x="Attendance", y="Internal_Marks", color="Performance")
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("Charts reflect the included synthetic dataset. Replace it with an approved dataset for measured results.")

elif page == "Model Evaluation":
    st.title("🧪 Model Evaluation")
    X = df[FEATURES]
    y = df["Performance"]
    counts = y.value_counts()
    can_stratify = counts.min() >= 2 and len(df) >= 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if can_stratify else None
    )
    eval_model = train_python_model(df.iloc[X_train.index] if False else df)
    # Evaluate using a separate split-trained model to avoid evaluating on its training data.
    eval_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))
    ])
    eval_model.fit(X_train, y_train)
    pred = eval_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    st.metric("Hold-out accuracy (Python model)", f"{acc:.1%}")
    st.caption("This is a hold-out estimate on the sample dataset, not WEKA J48 accuracy and not a guarantee of future performance.")
    st.subheader("Classification report")
    report = classification_report(y_test, pred, labels=CLASSES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    st.dataframe(pd.DataFrame(cm, index=CLASSES, columns=CLASSES), use_container_width=True)
    st.subheader("Decision-tree rules (Python baseline)")
    st.code(export_text(eval_model.named_steps["tree"], feature_names=FEATURES), language="text")
    st.info("For your assignment, separately run WEKA Explorer → Classify → trees → J48 → 10-fold cross-validation and record the actual WEKA results.")

else:
    st.title("ℹ️ About this project")
    st.markdown("""
**Project title:** Predicting Student Academic Performance Using Data Mining and WEKA

**Problem:** Academic records contain multiple variables. A classification model can explore patterns and estimate a performance category.

**Input attributes:** Attendance, internal marks, assignment marks, study hours and previous marks.

**Target:** Performance — Poor, Average, Good or Excellent.

**WEKA algorithm:** J48, WEKA's implementation of the C4.5 decision-tree approach.

**Privacy and responsible use:** Use anonymized data, limit access, and treat predictions as decision support only. Never use a model output as the sole basis for academic decisions.
""")
    st.subheader("Technology stack")
    st.write("- Streamlit: website interface")
    st.write("- pandas: dataset handling")
    st.write("- scikit-learn: local baseline model and evaluation")
    st.write("- WEKA J48: optional genuine WEKA prediction route")
    st.subheader("Dataset columns")
    st.code(", ".join(FEATURES + ["Performance"]))
    st.caption("This project includes synthetic records created for a classroom demo. Do not describe them as real student data.")
