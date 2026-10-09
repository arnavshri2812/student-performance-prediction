# Student Academic Performance Predictor

A beginner-friendly Streamlit mini project based on Case Study 1: Predicting Student Academic Performance Using Data Mining and WEKA.

## Important model note
The website's default model is a Python decision tree so it runs easily. This is **not** WEKA J48. The app includes an optional WEKA CLI path for genuine J48 prediction, but it requires Java, a WEKA JAR, and a matching ARFF dataset. For the university assignment, run J48 in WEKA Explorer and report the actual output from that run.

The included dataset is synthetic demonstration data. Do not present its metrics as real-world performance.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/. During Windows installation, enable "Add Python to PATH".

## 2. Open this folder in VS Code
Open the folder containing `app.py`.

## 3. Create and activate a virtual environment (Windows)
```bat
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt, or run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 4. Install packages
```bat
pip install -r requirements.txt
```

## 5. Run the site
```bat
streamlit run app.py
```
The browser should open at `http://localhost:8501`.

## 6. Dataset format
`student_performance.csv` must include:
- Student_ID (optional identifier; not used as a feature)
- Attendance
- Internal_Marks
- Assignment_Marks
- Study_Hours
- Previous_Marks
- Performance

Allowed labels are Poor, Average, Good, Excellent. Replace sample data only with data you are permitted to use.

## 7. Genuine WEKA J48 steps for the case study
1. Install WEKA from https://www.cs.waikato.ac.nz/ml/weka/.
2. Open WEKA Explorer → Preprocess → Open file, and load `student_performance.arff` (you can make an ARFF file from the CSV, or use WEKA's CSVLoader).
3. Confirm `Performance` is the class attribute (last attribute in the supplied ARFF).
4. Go to Classify → Choose → trees → J48.
5. Select 10-fold cross-validation and click Start.
6. Save screenshots of the selected J48 model, generated tree, accuracy/statistics and confusion matrix.
7. Do not copy the example Python evaluation values into the WEKA section.

### Optional WEKA prediction inside the website
The checkbox on Home & Predict can invoke the WEKA Java CLI if:
- Java is installed and `java -version` works in the terminal.
- You enter the full path to `weka.jar`.
- You enter the full path to a valid ARFF file with the same five numeric features and nominal Performance as the last attribute.
The app trains J48 from that ARFF file for the request, then predicts the entered row. This route is optional and can be sensitive to WEKA version / ARFF formatting. If it fails, use WEKA Explorer for the required practical screenshots and use the website's Python demo mode for the live website.

## 8. Suggested tests
- Try low values and high values; check that the page responds.
- Check Attendance is within 0–100.
- Check the charts load on Analytics.
- Check Model Evaluation displays a report and confusion matrix.
- Test the WEKA option only after following its setup.
- Never enter real personal information into a public demo.

## 9. Deploying with Streamlit Community Cloud
1. Create a GitHub account and a new repository.
2. Upload `app.py`, `requirements.txt`, and `student_performance.csv`.
3. Visit https://share.streamlit.io/ and connect GitHub.
4. Choose your repository and set the main file to `app.py`.
5. Deploy.
6. Test the public link in a private/incognito window.
Do not upload real student records or secrets to a public repository.
