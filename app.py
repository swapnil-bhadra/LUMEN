import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="LUMEN // Light Intelligence", page_icon="🔆", layout="wide")

# ====================================================================== THEME
CYAN, MAGENTA, AMBER, GREEN = "#00e5ff", "#ff2e97", "#ffb300", "#39ff88"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(0,229,255,0.12), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(255,46,151,0.10), transparent 55%),
        linear-gradient(180deg, #040814 0%, #060d1f 100%);
    background-attachment: fixed;
}
/* faint tech grid */
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.045) 1px, transparent 1px);
    background-size: 44px 44px;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,16,38,0.95), rgba(5,10,26,0.95));
    border-right: 1px solid rgba(0,229,255,0.18);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Orbitron', sans-serif; letter-spacing: 2px; font-size: 0.85rem;
    color: #00e5ff; text-transform: uppercase;
}

/* hero title */
.hero { display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.4rem; }
.hero h1 {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 2.1rem; margin: 0;
    letter-spacing: 6px;
    background: linear-gradient(90deg, #00e5ff, #7b61ff 50%, #ff2e97);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    filter: drop-shadow(0 0 12px rgba(0,229,255,0.35));
}
.hero .sub { color:#7f93b8; letter-spacing: 3px; font-size: 0.85rem; text-transform: uppercase; }
.live {
    font-family: 'Orbitron', sans-serif; font-size: 0.72rem; letter-spacing: 2px;
    padding: 6px 14px; border-radius: 999px; color:#39ff88;
    border: 1px solid rgba(57,255,136,0.5); background: rgba(57,255,136,0.08);
}
.live .dot {
    display:inline-block; width:8px; height:8px; border-radius:50%; background:#39ff88;
    margin-right:8px; box-shadow:0 0 10px #39ff88; animation: pulse 1.4s infinite;
}
.live.off { color:#ff5d5d; border-color: rgba(255,93,93,0.5); background: rgba(255,93,93,0.08); }
.live.off .dot { background:#ff5d5d; box-shadow:0 0 10px #ff5d5d; }
@keyframes pulse { 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:.35; transform:scale(.7);} }

/* glass cards */
.card {
    position: relative; padding: 16px 18px; border-radius: 14px;
    background: linear-gradient(145deg, rgba(14,28,62,0.72), rgba(8,16,38,0.72));
    border: 1px solid rgba(0,229,255,0.22);
    box-shadow: 0 0 22px rgba(0,229,255,0.07), inset 0 0 18px rgba(0,229,255,0.04);
    backdrop-filter: blur(8px);
    transition: transform .25s ease, box-shadow .25s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 0 28px rgba(0,229,255,0.22); }
.card::after {  /* corner accent */
    content:""; position:absolute; top:-1px; left:-1px; width:26px; height:26px;
    border-top: 2px solid var(--accent, #00e5ff); border-left: 2px solid var(--accent, #00e5ff);
    border-top-left-radius: 14px;
}
.card .lbl { font-family:'Orbitron',sans-serif; font-size:.66rem; letter-spacing:2.5px;
             color:#7f93b8; text-transform:uppercase; }
.card .val { font-family:'Orbitron',sans-serif; font-size:1.9rem; font-weight:700;
             color: var(--accent, #00e5ff); text-shadow: 0 0 14px var(--accent, #00e5ff); margin-top:6px; }
.card .sub { color:#9fb4d6; font-size:.95rem; margin-top:2px; }

.section {
    font-family:'Orbitron',sans-serif; font-size:.8rem; letter-spacing:4px; color:#00e5ff;
    text-transform:uppercase; margin: 22px 0 8px; display:flex; align-items:center; gap:12px;
}
.section::after { content:""; flex:1; height:1px;
    background: linear-gradient(90deg, rgba(0,229,255,.5), transparent); }

.status-chip {
    display:inline-block; padding:6px 16px; border-radius:8px; margin-top:8px;
    font-family:'Orbitron',sans-serif; letter-spacing:2px; font-size:.85rem;
    border:1px solid var(--accent); color:var(--accent); background: rgba(255,255,255,0.03);
    text-shadow: 0 0 10px var(--accent);
}
/* streamlit tabs & dataframes */
button[data-baseweb="tab"] { font-family:'Orbitron',sans-serif; letter-spacing:2px; font-size:.75rem; }
div[data-testid="stDataFrame"] { border:1px solid rgba(0,229,255,0.2); border-radius:10px; }
.footer { text-align:center; color:#4d6089; font-size:.8rem; letter-spacing:3px; margin-top:30px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label, value, sub="", accent=CYAN):
    st.markdown(
        f"""<div class="card" style="--accent:{accent}">
              <div class="lbl">{label}</div>
              <div class="val">{value}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


# ==================================================================== SIDEBAR
st.sidebar.markdown("### ⚙ Data link")
channel_id = st.sidebar.text_input("Channel ID", value="")
read_key = st.sidebar.text_input("Read API key (if private)", type="password")
field_no = st.sidebar.selectbox("LDR field", [1, 2, 3, 4, 5, 6, 7, 8], index=0)
n_results = st.sidebar.slider("Readings to fetch", 20, 8000, 300, step=20)

st.sidebar.markdown("### 🔧 Sensor")
adc_max = st.sidebar.selectbox("ADC range", [1023, 4095], index=0,
                               help="1023 = Arduino / ESP8266 (10-bit), 4095 = ESP32 (12-bit)")
bright_high = st.sidebar.checkbox("Higher value = brighter", value=True,
                                  help="Untick if your LDR divider gives lower values in bright light.")

st.sidebar.markdown("### 🧠 Prediction engine")
n_future = st.sidebar.slider("Points to forecast", 1, 100, 20)
train_last = st.sidebar.slider("Training window (last N)", 10, 1000, 100)

st.sidebar.markdown("### 🔄 Refresh")
auto = st.sidebar.checkbox("Auto-refresh", value=False)
every = st.sidebar.slider("Interval (s)", 15, 120, 30)
if st.sidebar.button("⟳  Sync now"):
    st.cache_data.clear()


# ======================================================================= DATA
@st.cache_data(ttl=15, show_spinner=False)
def fetch_data(channel, key, field, results):
    url = f"https://api.thingspeak.com/channels/{channel}/fields/{field}.json"
    params = {"results": results}
    if key:
        params["api_key"] = key
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("feeds", []))
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["value"] = pd.to_numeric(df[f"field{field}"], errors="coerce")
    return df.dropna(subset=["value"])[["created_at", "value"]].reset_index(drop=True)


def forecast(df, window, steps):
    d = df.tail(window).copy()
    t0 = d["created_at"].iloc[0]
    X = (d["created_at"] - t0).dt.total_seconds().to_numpy().reshape(-1, 1)
    y = d["value"].to_numpy()
    model = LinearRegression().fit(X, y)
    fitted = model.predict(X)
    gaps = np.diff(X.ravel())
    step = float(np.median(gaps)) if len(gaps) else 15.0
    step = step if step > 0 else 15.0
    fut_X = (X.ravel()[-1] + step * np.arange(1, steps + 1)).reshape(-1, 1)
    return {
        "train": d.assign(fitted=fitted),
        "future": pd.DataFrame({"created_at": t0 + pd.to_timedelta(fut_X.ravel(), unit="s"),
                                "predicted": model.predict(fut_X)}),
        "slope_min": float(model.coef_[0]) * 60,
        "r2": float(r2_score(y, fitted)) if len(y) > 2 else float("nan"),
        "std": float(np.std(y - fitted)),
        "step": step,
    }


def light_state(pct):
    if pct < 10:  return "DARK", "🌑", "#7b61ff"
    if pct < 30:  return "DIM", "🌘", "#4d8dff"
    if pct < 60:  return "MODERATE", "🌤", GREEN
    if pct < 85:  return "BRIGHT", "☀️", AMBER
    return "INTENSE", "🔆", "#ff5d3b"


def to_pct(v):
    p = v / adc_max * 100
    return p if bright_high else 100 - p


def plot_layout(fig, height):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,16,38,0.45)",
        font=dict(family="Rajdhani", color="#9fb4d6", size=14),
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor="#0b1630", bordercolor=CYAN, font_color="#e6f6ff"),
    )
    fig.update_xaxes(gridcolor="rgba(0,229,255,0.07)", zeroline=False, showspikes=True,
                     spikecolor=CYAN, spikethickness=1)
    fig.update_yaxes(gridcolor="rgba(0,229,255,0.07)", zeroline=False)
    return fig


# ===================================================================== HEADER
def header(live: bool):
    cls, txt = ("live", "LIVE LINK") if live else ("live off", "NO SIGNAL")
    st.markdown(
        f"""<div class="hero">
              <div><h1>LUMEN</h1><div class="sub">Light Intelligence Console · LDR → ThingSpeak</div></div>
              <div class="{cls}"><span class="dot"></span>{txt}</div>
            </div>""",
        unsafe_allow_html=True,
    )


if not channel_id:
    header(False)
    st.info("Enter your ThingSpeak **Channel ID** in the sidebar to establish the data link.")
    st.stop()

try:
    df = fetch_data(channel_id, read_key, field_no, n_results)
except requests.HTTPError as e:
    header(False)
    st.error(f"ThingSpeak error: {e}. Check Channel ID / API key.")
    st.stop()
except requests.RequestException as e:
    header(False)
    st.error(f"Network problem: {e}")
    st.stop()

if df.empty or len(df) < 3:
    header(False)
    st.warning("Not enough telemetry yet (need at least 3 readings).")
    st.stop()

header(True)

# ================================================================== HERO ROW
latest, prev = df["value"].iloc[-1], df["value"].iloc[-2]
pct = float(np.clip(to_pct(latest), 0, 100))
state, emoji, state_color = light_state(pct)
res = forecast(df, train_last, n_future)

g_col, k_col = st.columns([1.15, 1.85])

with g_col:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(family="Orbitron", size=44, color=state_color)),
        title=dict(text="AMBIENT LIGHT LEVEL",
                   font=dict(family="Orbitron", size=13, color="#7f93b8")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#4d6089", tickfont=dict(color="#7f93b8")),
            bar=dict(color=state_color, thickness=0.28),
            bgcolor="rgba(8,16,38,0.6)", borderwidth=1, bordercolor="rgba(0,229,255,0.3)",
            steps=[
                dict(range=[0, 10], color="rgba(123,97,255,0.30)"),
                dict(range=[10, 30], color="rgba(77,141,255,0.25)"),
                dict(range=[30, 60], color="rgba(57,255,136,0.20)"),
                dict(range=[60, 85], color="rgba(255,179,0,0.22)"),
                dict(range=[85, 100], color="rgba(255,93,59,0.28)"),
            ],
            threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.8, value=pct),
        ),
    ))
    gauge.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=0),
                        font=dict(family="Rajdhani"))
    st.plotly_chart(gauge)
    st.markdown(
        f'<div style="text-align:center"><span class="status-chip" style="--accent:{state_color}">'
        f'{emoji}&nbsp; {state}</span></div>', unsafe_allow_html=True)

with k_col:
    r1 = st.columns(3)
    with r1[0]:
        card("Latest raw", f"{latest:.0f}", f"{latest - prev:+.0f} vs previous", CYAN)
    with r1[1]:
        card("Average", f"{df['value'].mean():.0f}", "over fetched window", "#7b61ff")
    with r1[2]:
        card("Samples", f"{len(df)}", f"every ~{res['step']:.0f}s", GREEN)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r2 = st.columns(3)
    with r2[0]:
        card("Peak", f"{df['value'].max():.0f}", df.loc[df['value'].idxmax(), 'created_at'].strftime('%H:%M:%S'), AMBER)
    with r2[1]:
        card("Floor", f"{df['value'].min():.0f}", df.loc[df['value'].idxmin(), 'created_at'].strftime('%H:%M:%S'), "#4d8dff")
    with r2[2]:
        trend = res["slope_min"] * (1 if bright_high else -1)
        arrow = "▲" if trend > 0.01 else "▼" if trend < -0.01 else "■"
        tcolor = AMBER if trend > 0.01 else "#4d8dff" if trend < -0.01 else GREEN
        card("Trend / min", f"{arrow} {abs(res['slope_min']):.2f}",
             "brightening" if trend > 0.01 else "dimming" if trend < -0.01 else "stable", tcolor)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r3 = st.columns(2)
    with r3[0]:
        r2v = res["r2"]
        card("Model fit (R²)", f"{r2v:.3f}",
             "reliable" if r2v > 0.6 else "moderate" if r2v > 0.3 else "weak · short-term only",
             GREEN if r2v > 0.6 else AMBER if r2v > 0.3 else MAGENTA)
    with r3[1]:
        nxt = res["future"]["predicted"].iloc[-1]
        card(f"Forecast +{n_future} pts", f"{nxt:.0f}",
             res["future"]["created_at"].iloc[-1].strftime("at %H:%M:%S"), MAGENTA)

# ============================================================== MAIN CHART
section("Telemetry & forecast")

fig = go.Figure()
# glow underlay + main line + gradient fill
fig.add_trace(go.Scatter(x=df["created_at"], y=df["value"], mode="lines", hoverinfo="skip",
                         line=dict(color="rgba(0,229,255,0.18)", width=10), showlegend=False))
fig.add_trace(go.Scatter(x=df["created_at"], y=df["value"], mode="lines", name="Actual",
                         line=dict(color=CYAN, width=2.4, shape="spline", smoothing=0.5),
                         fill="tozeroy", fillcolor="rgba(0,229,255,0.07)"))
fig.add_trace(go.Scatter(x=res["train"]["created_at"], y=res["train"]["fitted"], mode="lines",
                         name="Regression fit", line=dict(color=AMBER, width=1.8, dash="dot")))
# confidence band
fx = res["future"]["created_at"]
up, lo = res["future"]["predicted"] + res["std"], res["future"]["predicted"] - res["std"]
fig.add_trace(go.Scatter(x=pd.concat([fx, fx[::-1]]), y=pd.concat([up, lo[::-1]]),
                         fill="toself", fillcolor="rgba(255,46,151,0.13)", line=dict(width=0),
                         hoverinfo="skip", name="±1σ band"))
fig.add_trace(go.Scatter(x=fx, y=res["future"]["predicted"], mode="lines+markers", name="Predicted",
                         line=dict(color=MAGENTA, width=2.4, dash="dash"),
                         marker=dict(size=6, color=MAGENTA, line=dict(color="#fff", width=1))))
# "now" marker
now_x = df["created_at"].iloc[-1]
fig.add_shape(type="line", x0=now_x, x1=now_x, y0=0, y1=1, yref="paper",
              line=dict(color="rgba(255,255,255,0.35)", dash="dot"))
fig.add_annotation(x=now_x, y=1, yref="paper", text="NOW", showarrow=False,
                   yshift=10, font=dict(color="#fff", family="Orbitron", size=10))
fig.update_yaxes(title="LDR reading")
st.plotly_chart(plot_layout(fig, 470))

# ====================================================== SECONDARY ANALYTICS
c_left, c_right = st.columns(2)
with c_left:
    section("Light-level distribution")
    pcts = df["value"].apply(lambda v: float(np.clip(to_pct(v), 0, 100)))
    bins = pd.cut(pcts, [-1, 10, 30, 60, 85, 101],
                  labels=["Dark", "Dim", "Moderate", "Bright", "Intense"])
    counts = bins.value_counts().reindex(["Dark", "Dim", "Moderate", "Bright", "Intense"]).fillna(0)
    bar = go.Figure(go.Bar(x=counts.index, y=counts.values,
                           marker=dict(color=["#7b61ff", "#4d8dff", GREEN, AMBER, "#ff5d3b"],
                                       line=dict(color="rgba(255,255,255,0.5)", width=1))))
    bar.update_yaxes(title="readings")
    st.plotly_chart(plot_layout(bar, 280))
with c_right:
    section("Residuals (model error)")
    resid = res["train"]["value"] - res["train"]["fitted"]
    rf = go.Figure(go.Scatter(x=res["train"]["created_at"], y=resid, mode="markers",
                              marker=dict(color=resid, colorscale=[[0, "#4d8dff"], [0.5, CYAN], [1, MAGENTA]],
                                          size=7, opacity=0.85)))
    rf.add_hline(y=0, line=dict(color="rgba(255,255,255,0.4)", dash="dot"))
    st.plotly_chart(plot_layout(rf, 280))

# ============================================================== DATA TABLES
section("Data bank")
tab1, tab2 = st.tabs(["PREDICTIONS", "RAW TELEMETRY"])
with tab1:
    out = res["future"].copy()
    out["created_at"] = out["created_at"].dt.strftime("%H:%M:%S")
    out["predicted"] = out["predicted"].round(1)
    out["light %"] = out["predicted"].apply(lambda v: round(float(np.clip(to_pct(v), 0, 100)), 1))
    st.dataframe(out.rename(columns={"created_at": "Time", "predicted": "Predicted LDR"}),
                 hide_index=True)
with tab2:
    st.dataframe(df.sort_values("created_at", ascending=False), hide_index=True)
    st.download_button("⬇ Export CSV", df.to_csv(index=False), "ldr_telemetry.csv", "text/csv")

st.markdown(
    f'<div class="footer">LAST SYNC · {df["created_at"].iloc[-1].strftime("%d %b %Y %H:%M:%S")} IST'
    f' · CHANNEL {channel_id}</div>', unsafe_allow_html=True)

if auto:
    time.sleep(every)
    st.cache_data.clear()
    st.rerun()