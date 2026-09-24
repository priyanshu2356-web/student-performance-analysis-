import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="NEXUS | Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; }
.stApp { background: radial-gradient(circle at 15% 5%, #141c52 0%, #060916 45%, #02030a 100%); color: #dff6ff; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; color: #dff6ff; }
.hero { font-family: 'Orbitron', sans-serif; font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 900;
        background: linear-gradient(90deg, #00f5ff, #8a5cff, #ff2bd6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #7fa8c9; letter-spacing: 2px; margin-bottom: 1.2rem; }
.card { border: 1px solid rgba(0,245,255,.35); border-radius: 16px; padding: 18px 22px; margin-bottom: 14px;
        background: rgba(10,16,40,.65); box-shadow: 0 0 22px rgba(0,245,255,.12), inset 0 0 18px rgba(138,92,255,.08); }
.label { color: #7fa8c9; font-size: .9rem; letter-spacing: 1px; }
.big { font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; }
.low { color: #00ffa3; text-shadow: 0 0 14px #00ffa3; }
.mid { color: #ffd23f; text-shadow: 0 0 14px #ffd23f; }
.high { color: #ff3b6b; text-shadow: 0 0 14px #ff3b6b; }
section[data-testid="stSidebar"] { background: #050818; border-right: 1px solid rgba(0,245,255,.25); }
button[data-baseweb="tab"] { font-family: 'Orbitron', sans-serif; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

NUM = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
       "freetime", "goout", "Dalc", "Walc", "health", "absences"]
BIN = ["schoolsup", "famsup", "paid", "activities", "higher", "internet", "romantic"]


@st.cache_resource
def train_models():
    df = pd.read_csv("student-mat.csv", sep=";")
    for c in BIN:
        df[c] = (df[c] == "yes").astype(int)
    y, passed = df["G3"], (df["G3"] >= 10).astype(int)
    out = {}
    for mode, feats in {"early": NUM + BIN, "grades": NUM + BIN + ["G1", "G2"]}.items():
        Xtr, Xte, ytr, yte, ptr, _ = train_test_split(df[feats], y, passed, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced").fit(Xtr, ptr)
        imp = pd.Series(reg.feature_importances_, index=feats).sort_values().tail(10)
        out[mode] = {"reg": reg, "clf": clf, "r2": r2_score(yte, reg.predict(Xte)), "feats": feats, "imp": imp}
    return out


models = train_models()

# ---------- Sidebar inputs ----------
st.sidebar.markdown("### STUDENT PROFILE")
mode_label = st.sidebar.radio("Prediction mode", ["Early warning (no grades)", "With previous grades"])
mode = "early" if mode_label.startswith("Early") else "grades"

v = {}
v["studytime"] = st.sidebar.select_slider("Weekly study time", [1, 2, 3, 4], 2,
                                          format_func=lambda x: {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[x])
v["failures"] = st.sidebar.slider("Past class failures", 0, 3, 0)
v["absences"] = st.sidebar.slider("Absences", 0, 40, 4)
v["age"] = st.sidebar.slider("Age", 15, 22, 17)
v["Medu"] = st.sidebar.slider("Mother's education (0-4)", 0, 4, 2)
v["Fedu"] = st.sidebar.slider("Father's education (0-4)", 0, 4, 2)
v["traveltime"] = st.sidebar.slider("Travel time (1-4)", 1, 4, 1)
v["famrel"] = st.sidebar.slider("Family relationship (1-5)", 1, 5, 4)
v["freetime"] = st.sidebar.slider("Free time (1-5)", 1, 5, 3)
v["goout"] = st.sidebar.slider("Going out (1-5)", 1, 5, 3)
v["Dalc"] = st.sidebar.slider("Weekday alcohol (1-5)", 1, 5, 1)
v["Walc"] = st.sidebar.slider("Weekend alcohol (1-5)", 1, 5, 2)
v["health"] = st.sidebar.slider("Health (1-5)", 1, 5, 3)
for c, label in [("schoolsup", "School support"), ("famsup", "Family support"), ("paid", "Paid classes"),
                 ("activities", "Extra activities"), ("higher", "Wants higher education"),
                 ("internet", "Internet at home"), ("romantic", "In a relationship")]:
    v[c] = int(st.sidebar.checkbox(label, value=(c in ["higher", "internet"])))
if mode == "grades":
    v["G1"] = st.sidebar.slider("Period 1 grade (G1)", 0, 20, 10)
    v["G2"] = st.sidebar.slider("Period 2 grade (G2)", 0, 20, 10)

# ---------- Main ----------
st.markdown('<div class="hero">NEXUS // STUDENT PERFORMANCE PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Machine learning on real school data. Adjust the profile on the left and watch the forecast update.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "What drives results", "About the model"])

m = models[mode]
x = pd.DataFrame([v])[m["feats"]]
score = float(m["reg"].predict(x)[0])
score = max(0.0, min(20.0, score))
p_pass = float(m["clf"].predict_proba(x)[0][1])

if p_pass >= 0.75:
    risk, cls = "Low risk", "low"
elif p_pass >= 0.5:
    risk, cls = "Moderate risk", "mid"
else:
    risk, cls = "High risk", "high"

with tab1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": " / 20", "font": {"family": "Orbitron", "color": "#00f5ff"}},
            gauge={"axis": {"range": [0, 20], "tickcolor": "#7fa8c9"}, "bar": {"color": "#00f5ff"},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                   "steps": [{"range": [0, 10], "color": "rgba(255,59,107,.35)"},
                             {"range": [10, 14], "color": "rgba(255,210,63,.3)"},
                             {"range": [14, 20], "color": "rgba(0,255,163,.3)"}]}))
        fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dff6ff"},
                          margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="card"><div class="label">PREDICTED FINAL SCORE</div>'
                    f'<div class="big">{score:.1f}<span class="label"> / 20</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="label">CHANCE OF PASSING (10+)</div>'
                    f'<div class="big {cls}">{p_pass*100:.0f}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="label">STATUS</div><div class="big {cls}">{risk}</div></div>',
                    unsafe_allow_html=True)
    tips = []
    if v["failures"] > 0:
        tips.append("Past failures are the strongest non-grade risk factor. Targeted revision of earlier topics can help.")
    if v["studytime"] <= 1:
        tips.append("Study time is very low. Even a small weekly increase is linked with better results.")
    if v["goout"] >= 4 or v["Walc"] >= 4:
        tips.append("High social time and weekend alcohol use tend to go with lower scores in this data.")
    if not v["higher"]:
        tips.append("Students who plan higher education score better on average.")
    if tips:
        st.markdown("#### Suggestions")
        for t in tips:
            st.write("- " + t)

with tab2:
    st.markdown("#### Top 10 features in this mode")
    imp = m["imp"]
    fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                            marker=dict(color=imp.values, colorscale=[[0, "#8a5cff"], [1, "#00f5ff"]])))
    fig2.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": "#dff6ff"}, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(gridcolor="rgba(0,245,255,.1)"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Switch the mode in the sidebar to see how much the ranking changes once G1 and G2 are included.")

with tab3:
    st.markdown(f"""
<div class="card">
<b>Data:</b> UCI Student Performance dataset (math course, 395 students).<br>
<b>Models:</b> Random Forest regressor (final score) and Random Forest classifier (pass/fail).<br>
<b>Hold-out R² in this mode:</b> {m['r2']:.2f}<br>
<b>Limits:</b> small dataset from two Portuguese schools. Without earlier grades the model explains only a small part of the variation,
so treat the early-warning result as a rough signal, not a verdict on any student.
</div>
""", unsafe_allow_html=True)
