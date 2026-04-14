from flask import Flask, render_template, request, redirect, session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from lime.lime_tabular import LimeTabularExplainer

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = "secure_key_2026"

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset/student_data.csv")
df.columns = df.columns.str.strip()

# ---------- MODEL ----------
X = df[['math score', 'reading score']]
y = df['writing score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

accuracy = round(r2_score(y_test, model.predict(X_test)) * 100, 2)

# ---------- LIME ----------
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression'
)

# ---------- ROUTES ----------
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == "mamatha" and request.form['password'] == "636464":
            session['user'] = "admin"
            return redirect('/dashboard')
        else:
            return render_template("login.html", error="Invalid Login ❌")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template("dashboard.html", accuracy=accuracy)

# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    # Inputs
    math = float(request.form['math'])
    reading = float(request.form['reading'])
    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    sleep = float(request.form['sleep'])
    prep = request.form['prep']

    # Base ML prediction
    input_df = pd.DataFrame([[math, reading]], columns=['math score','reading score'])
    prediction = model.predict(input_df)[0]

    # Adjust using extra inputs
    bonus = (study*1.5 + sleep*1 + attendance*0.1 + (5 if prep=="yes" else 0))
    prediction = round(min(prediction + bonus, 100), 2)

    status = "PASS ✅" if prediction >= 35 else "FAIL ❌"

    # Recommendation
    if prediction >= 80:
        rec = "🌟 Excellent performance!"
    elif prediction >= 60:
        rec = "👍 Good job!"
    elif prediction >= 35:
        rec = "⚠️ Needs improvement."
    else:
        rec = "❗ Work harder."

    # LIME
    exp = explainer.explain_instance(input_df.values[0], model.predict)
    lime_exp = exp.as_list()

    return render_template(
        "dashboard.html",
        result=f"{prediction}% ({status})",
        math=math,
        reading=reading,
        prediction=prediction,
        recommendation=rec,
        lime_exp=lime_exp,
        accuracy=accuracy
    )

# ---------- GRAPH ----------
@app.route('/graph')
def graph():
    plt.figure()
    sns.histplot(df['writing score'], kde=True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    graph = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render_template("dashboard.html", overall_graph=graph, accuracy=accuracy)

# ---------- ANALYSIS ----------
@app.route('/analysis')
def analysis():
    graphs = []

    # Histogram
    plt.figure()
    sns.histplot(df['writing score'], kde=True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs.append(base64.b64encode(buf.getvalue()).decode())
    plt.close()

    # Scatter Math vs Writing
    plt.figure()
    sns.scatterplot(x=df['math score'], y=df['writing score'])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs.append(base64.b64encode(buf.getvalue()).decode())
    plt.close()

    # Scatter Reading vs Writing
    plt.figure()
    sns.scatterplot(x=df['reading score'], y=df['writing score'])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs.append(base64.b64encode(buf.getvalue()).decode())
    plt.close()

    # Heatmap
    plt.figure()
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs.append(base64.b64encode(buf.getvalue()).decode())
    plt.close()

    return render_template("dashboard.html", graphs=graphs, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)