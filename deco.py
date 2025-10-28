# -*- coding: utf-8 -*-
import io
import warnings
from typing import List, Dict, Any, Optional
import base64
import uuid
from datetime import datetime
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.signal import find_peaks, savgol_filter
from scipy.special import wofz
from scipy.interpolate import interp1d

# -------------------------------------------
# Page Config
# -------------------------------------------
st.set_page_config(
    page_title="Deconvolução Espectral Avançada Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# ======================================
# FUNÇÃO MELHORADA: Reconhecimento automático de separadores numéricos
# ======================================
def parse_numeric_value(value_str):
    if isinstance(value_str, (int, float, np.number)):
        return float(value_str)
    if pd.isna(value_str) or value_str == '':
        return np.nan
    value_str = str(value_str).strip().replace(' ', '')
    if 'e' in value_str.lower() or 'E' in value_str:
        try:
            return float(value_str)
        except:
            pass
    num_dots = value_str.count('.')
    num_commas = value_str.count(',')
    if num_dots > 1 or (num_dots >= 1 and num_commas == 1 and value_str.rfind(',') > value_str.rfind('.')):
        value_str = value_str.replace('.', '').replace(',', '.')
    elif num_commas > 1 or (num_commas >= 1 and num_dots == 1 and value_str.rfind('.') > value_str.rfind(',')):
        value_str = value_str.replace(',', '')
    elif num_commas == 1 and num_dots == 0:
        value_str = value_str.replace(',', '.')
    try:
        return float(value_str)
    except ValueError:
        return np.nan

def coerce_numeric_series(s: pd.Series):
    if s.dtype == object:
        return s.apply(parse_numeric_value)
    else:
        return pd.to_numeric(s, errors="coerce")

def coerce_numeric_df(df: pd.DataFrame):
    out = df.copy()
    for c in out.columns:
        out[c] = coerce_numeric_series(out[c])
    return out

def plotly_download_button(fig, filename="grafico.png", fmt="png", width=1600, height=900, scale=2):
    fig_json = fig.to_json()
    fig_b64 = base64.b64encode(fig_json.encode("utf-8")).decode("ascii")
    uid = "pldl_" + uuid.uuid4().hex
    html = f"""
    <div id="{uid}" style="position:absolute; left:-10000px; top:0; width:{width}px; height:{height}px;"></div>
    <button id="{uid}_btn" style="width: 100%; padding:0.5rem 0.75rem; border-radius:8px; border:1px solid #ccc; cursor:pointer; background-color: #007bff; color: white;">
      ⬇️ Baixar como {fmt.upper()}
    </button>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
    (function(){{
        const uid = "{uid}";
        const container = document.getElementById(uid);
        const btn = document.getElementById(uid + "_btn");
        const fig = JSON.parse(atob("{fig_b64}"));
        if ("{fmt}" === "png" && fig.layout.paper_bgcolor === 'rgba(0,0,0,0)') {{
            fig.layout.plot_bgcolor = 'rgba(0,0,0,0)';
        }}
        Plotly.newPlot(container, fig.data, fig.layout, {{displayModeBar:false, responsive:false}}).then(() => {{
            btn.onclick = function(){{
                btn.innerText = "Processando...";
                setTimeout(function() {{
                    Plotly.downloadImage(container, {{
                        format: "{fmt}",
                        filename: "{filename.rsplit('.', 1)[0]}",
                        width: {width},
                        height: {height},
                        scale: {scale}
                    }}).then(() => {{
                         btn.innerText = "⬇️ Baixar como {fmt.upper()}";
                    }});
                }}, 50);
            }};
        }});
    }})();
    </script>
    """
    components.html(html, height=60)

def excel_sheet_names(file):
    try:
        xls = pd.ExcelFile(file)
        return xls.sheet_names
    except Exception:
        return None

def get_excel_writer(buffer):
    try:
        import xlsxwriter
        return pd.ExcelWriter(buffer, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl
            return pd.ExcelWriter(buffer, engine="openpyxl")
        except Exception:
            return None

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
    }
    div[data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Peak models
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, amplitude, center, gamma):
    return amplitude * (gamma**2) / ((x - center) ** 2 + gamma**2)

def voigt_exact(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def pseudo_voigt(x, amplitude, center, width, eta):
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    gamma = width / 2.0
    g = gaussian(x, amplitude, center, sigma)
    l = lorentzian(x, amplitude, center, gamma)
    return eta * l + (1 - eta) * g

def asymmetric_gaussian(x, amplitude, center, sigma_left, sigma_right):
    left_mask = x < center
    right_mask = ~left_mask
    out = np.zeros_like(x, dtype=float)
    if np.any(left_mask):
        out[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / sigma_left) ** 2)
    if np.any(right_mask):
        out[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / sigma_right) ** 2)
    return out

def pearson_vii(x, amplitude, center, width, m):
    return amplitude / (1.0 + ((x - center) / width) ** 2) ** m

def exponential_gaussian(x, amplitude, center, sigma, tau):
    from scipy.special import erfc
    y = (sigma / tau) * np.sqrt(np.pi / 2) * np.exp(0.5 * (sigma / tau)**2 - (x - center) / tau)
    y *= erfc((sigma / tau - (x - center) / sigma) / np.sqrt(2))
    return amplitude * y / np.max(y) if np.max(y) > 0 else np.zeros_like(x)

def doniach_sunjic(x, amplitude, center, gamma, alpha):
    E = x - center
    numerator = np.cos(np.pi * alpha / 2 + (1 - alpha) * np.arctan(E / gamma))
    denominator = (gamma**2 + E**2)**((1 - alpha) / 2)
    return amplitude * numerator / denominator

def synthetic_example(n=1200, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 1800, n)
    y = (
        gaussian(x, 1.2, 800, 25)
        + lorentzian(x, 0.9, 1100, 30)
        + pseudo_voigt(x, 1.1, 1450, 80, 0.4)
    )
    y += rng.normal(0, noise, size=x.size)
    return pd.DataFrame({"x": x, "y": y})

def area_under_peak(x: np.ndarray, y_peak: np.ndarray, model_type: str, params: List[float]) -> float:
    if model_type == "Gaussiana":
        amp, _, sigma = params[:3]
        return float(amp * sigma * np.sqrt(2 * np.pi))
    if model_type == "Lorentziana":
        amp, _, gamma = params[:3]
        return float(np.pi * amp * gamma)
    return float(np.trapz(y_peak, x))

def fwhm_of_peak(model_type: str, params: List[float]) -> Optional[float]:
    if model_type == "Gaussiana":
        sigma = params[2]
        return float(2 * np.sqrt(2 * np.log(2)) * sigma)
    if model_type == "Lorentziana":
        gamma = params[2]
        return float(2 * gamma)
    if model_type == "Pseudo-Voigt":
        return float(params[2])
    if model_type == "Voigt (exato)":
        sigma, gamma = params[2], params[3]
        return float(0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2.355 * sigma) ** 2))
    if model_type == "Gaussiana Assimétrica":
        sigma_l, sigma_r = params[2], params[3]
        sigma_eq = 0.5 * (sigma_l + sigma_r)
        return float(2 * np.sqrt(2 * np.log(2)) * sigma_eq)
    if model_type == "Pearson VII":
        width, m = params[2], params[3]
        return float(2 * width * np.sqrt(2 ** (1.0 / m) - 1.0))
    return None

# Preprocessing Functions
def baseline_correction(x, y, method="linear", **kwargs):
    if method == "linear":
        coeffs = np.polyfit(x, y, 1)
        baseline = np.polyval(coeffs, x)
    elif method == "polynomial":
        degree = kwargs.get("degree", 3)
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
    elif method == "moving_average":
        window = kwargs.get("window", 50)
        baseline = np.convolve(y, np.ones(window)/window, mode='same')
    else:
        baseline = np.zeros_like(y)
    return y - baseline, baseline

def normalize_spectrum(y, method="max"):
    if method == "max":
        return y / np.max(y) if np.max(y) > 0 else y
    elif method == "area":
        area = np.trapz(y)
        return y / area if area > 0 else y
    elif method == "minmax":
        ymin, ymax = np.min(y), np.max(y)
        if ymax > ymin:
            return (y - ymin) / (ymax - ymin)
    return y

def smooth_spectrum(x, y, method="savgol", **kwargs):
    if method == "savgol":
        window = kwargs.get("window", 11)
        poly = kwargs.get("poly", 3)
        if window % 2 == 0:
            window += 1
        if window > len(y):
            window = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window < 3:
            return y
        if poly >= window:
            poly = window - 1
        return savgol_filter(y, window, poly)
    elif method == "moving_average":
        window = kwargs.get("window", 5)
        return np.convolve(y, np.ones(window)/window, mode='same')
    return y

# Deconvolution Engine
class SpectralDeconvolution:
    def __init__(self):
        self.peak_models = {
            "Gaussiana": ("gaussian", ["Amplitude", "Centro", "Sigma"]),
            "Lorentziana": ("lorentzian", ["Amplitude", "Centro", "Gamma"]),
            "Voigt (exato)": ("voigt_exact", ["Amplitude", "Centro", "Sigma (G)", "Gamma (L)"]),
            "Pseudo-Voigt": ("pseudo_voigt", ["Amplitude", "Centro", "Largura (FWHM~)", "Fração Lorentz (η)"]),
            "Gaussiana Assimétrica": ("asymmetric_gaussian", ["Amplitude", "Centro", "Sigma Esq", "Sigma Dir"]),
            "Pearson VII": ("pearson_vii", ["Amplitude", "Centro", "Largura", "Forma (m)"]),
            "Gaussiana Exponencial": ("exponential_gaussian", ["Amplitude", "Centro", "Sigma", "Tau"]),
            "Doniach-Sunjic": ("doniach_sunjic", ["Amplitude", "Centro", "Gamma", "Alpha"]),
        }

    def _eval_single(self, x: np.ndarray, peak_type: str, params: List[float]) -> np.ndarray:
        if peak_type == "Gaussiana": return gaussian(x, *params)
        if peak_type == "Lorentziana": return lorentzian(x, *params)
        if peak_type == "Voigt (exato)": return voigt_exact(x, *params)
        if peak_type == "Pseudo-Voigt": return pseudo_voigt(x, *params)
        if peak_type == "Gaussiana Assimétrica": return asymmetric_gaussian(x, *params)
        if peak_type == "Pearson VII": return pearson_vii(x, *params)
        if peak_type == "Gaussiana Exponencial": return exponential_gaussian(x, *params)
        if peak_type == "Doniach-Sunjic": return doniach_sunjic(x, *params)
        return np.zeros_like(x)

    def create_composite(self, peak_list: List[Dict[str, Any]]):
        def comp(x, *flat_params):
            y = np.zeros_like(x, dtype=float)
            idx = 0
            for peak in peak_list:
                n = len(self.peak_models[peak["type"]][1])
                vals = flat_params[idx:idx+n]
                y += self._eval_single(x, peak["type"], vals)
                idx += n
            return y
        return comp

    def fit(self, x: np.ndarray, y: np.ndarray, peak_list: List[Dict[str, Any]], method: str, **kwargs):
        comp = self.create_composite(peak_list)
        p0 = []
        bounds_lower, bounds_upper = [], []
        for pk in peak_list:
            p0.extend(pk["params"])
            bl, bu = zip(*pk["bounds"])
            bounds_lower.extend(bl)
            bounds_upper.extend(bu)
        bounds = (np.array(bounds_lower, dtype=float), np.array(bounds_upper, dtype=float))

        try:
            if method == "curve_fit":
                popt, pcov = curve_fit(comp, x, y, p0=p0, bounds=bounds,
                                     maxfev=kwargs.get("maxfev", 20000),
                                     method=kwargs.get("algorithm", "trf"))
                return popt, pcov
            elif method == "differential_evolution":
                def objective(vec): return np.sum((y - comp(x, *vec)) ** 2)
                res = differential_evolution(objective, list(zip(bounds[0], bounds[1])),
                                           maxiter=kwargs.get("maxiter", 1000))
                return res.x, None
            elif method == "minimize":
                def objective(vec): return np.sum((y - comp(x, *vec)) ** 2)
                res = minimize(objective, p0, method=kwargs.get("algorithm", "L-BFGS-B"),
                             bounds=list(zip(bounds[0], bounds[1])))
                return res.x, None
        except Exception as exc:
            return None, None

def plot_figure(x, y, peaks, dec, settings: Dict[str, Any], y_fit_total_ext=None):
    vs = settings
    color_schemes = {
        "default": {"data": "#4DA3FF", "sum": "#FF6EC7", "residuals": "#FF4D4D", "highlight": "#FFD166"},
        "scientific": {"data": "#1f77b4", "sum": "#ff7f0e", "residuals": "#2ca02c", "highlight": "#d62728"},
        "dark": {"data": "#00D9FF", "sum": "#FF00FF", "residuals": "#00FF00", "highlight": "#FFFF00"},
        "publication": {"data": "#000000", "sum": "#E74C3C", "residuals": "#3498DB", "highlight": "#F39C12"}
    }
    colors = color_schemes.get(vs.get("color_scheme"), color_schemes["default"])

    fig = go.Figure()
    trace_mode = vs.get("plot_style", "lines")
    fig.add_trace(go.Scatter(x=x, y=y, mode=trace_mode, name="Dados",
                            line=dict(width=vs.get("line_width", 2), color=colors["data"]),
                            marker=dict(size=vs.get("marker_size", 4), color=colors["data"])))

    y_fit_total = np.zeros_like(x, dtype=float) if y_fit_total_ext is None else y_fit_total_ext
    shapes = []

    if vs.get("show_fit", True) and len(peaks) > 0:
        okabe_ito_palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
        palettes = {p: getattr(px.colors.sequential, p, None) for p in ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "IceFire", "Sunset", "Jet"]}
        palettes["Plotly"] = px.colors.qualitative.Plotly
        palettes["Okabe-Ito"] = okabe_ito_palette
        comp_colors = palettes.get(vs.get("component_palette"), px.colors.qualitative.Plotly)
        
        is_fitting_run = y_fit_total_ext is None
        
        for i, pk in enumerate(peaks):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            if is_fitting_run: y_fit_total += y_comp

            if vs.get("show_components"):
                is_h = (vs.get("highlight_idx") is not None and i == vs.get("highlight_idx"))
                comp_color = comp_colors[i % len(comp_colors)]
                line_style = dict(width=vs.get("line_width", 2) + (1 if is_h else -1), dash="solid" if is_h else "dot", color=colors["highlight"] if is_h else comp_color)
                name = f"{pk['type']} #{i+1}" + (" (★)" if is_h else "")

                fill_color_rgba = 'rgba(0,0,0,0)'
                if vs.get("fill_areas"):
                    from plotly.colors import hex_to_rgb
                    rgb = hex_to_rgb(colors["highlight"] if is_h else comp_color)
                    fill_opacity = 0.6 if is_h else vs.get("comp_opacity", 0.35)
                    fill_color_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{fill_opacity})"
                fig.add_trace(go.Scatter(x=x, y=y_comp, mode="lines", name=name, line=line_style, fill="tozeroy", fillcolor=fill_color_rgba))

                if vs.get("show_centers"):
                    cx = float(pk["params"][1])
                    y0, y1 = (vs.get("y_range")[0], vs.get("y_range")[1]) if vs.get("y_range") else (y.min(), y.max())
                    shapes.append(dict(type="line", x0=cx, x1=cx, y0=y0, y1=y1, line=dict(color=colors["highlight"] if is_h else "#666", width=1, dash="dash")))

        fig.add_trace(go.Scatter(x=x, y=y_fit_total, mode="lines", name="Soma Ajuste", line=dict(width=vs.get("line_width", 2) + 1, color=colors["sum"])))

        if vs.get("show_residuals"):
            res = y - y_fit_total
            fig.add_trace(go.Scatter(x=x, y=res, mode="lines", name="Resíduos", line=dict(width=vs.get("line_width", 2) - 1, color=colors["residuals"]), yaxis="y2"))

    legend_positions = { "topright": dict(y=0.98, x=0.98), "topleft": dict(y=0.98, x=0.02), "bottomright": dict(y=0.02, x=0.98), "bottomleft": dict(y=0.02, x=0.02), "outside": dict(y=0.5, x=1.05) }
    tick_formats = {"auto": None, "científico": ".2e", "SI": "~s"}

    layout = dict(
        title=dict(text=vs.get('title', 'Deconvolução Espectral'), font=dict(size=vs.get("title_size", 20))),
        height=650, hovermode="x unified", showlegend=vs.get("show_legend"),
        legend=legend_positions.get(vs.get("legend_position")),
        shapes=shapes,
        plot_bgcolor='rgba(0,0,0,0)' if vs.get("transparent_bg") else ('#1a1a1a' if vs.get("color_scheme") == "dark" else 'white'),
        paper_bgcolor='rgba(0,0,0,0)' if vs.get("transparent_bg") else ('#2a2a2a' if vs.get("color_scheme") == "dark" else 'white'),
        font=dict(color='white' if vs.get("color_scheme") == "dark" else 'black')
    )
    
    grid_color = '#444' if vs.get("color_scheme") == "dark" else '#E0E0E0'
    layout["xaxis"] = dict(title=dict(text=vs.get('x_label', 'X'), font=dict(size=vs.get("label_size", 14))), showgrid=vs.get("show_grid"), gridcolor=grid_color, tickformat=tick_formats.get(vs.get("x_tick_format")), tickfont=dict(size=vs.get("tick_size", 12)))
    layout["yaxis"] = dict(title=dict(text=vs.get('y_label', 'Intensidade'), font=dict(size=vs.get("label_size", 14))), showgrid=vs.get("show_grid"), gridcolor=grid_color, tickformat=tick_formats.get(vs.get("y_tick_format")), tickfont=dict(size=vs.get("tick_size", 12)))
    if vs.get("show_residuals"): layout["yaxis2"] = dict(overlaying="y", side="right", title="Resíduos", showgrid=False, zeroline=True, tickfont=dict(color=colors["residuals"]))
    if vs.get("y_range") is not None: layout["yaxis"]["range"] = vs.get("y_range")

    fig.update_layout(**layout)
    return fig, y_fit_total

# Session initialization
if "df" not in st.session_state: st.session_state.df = None
if "x" not in st.session_state: st.session_state.x = None
if "y" not in st.session_state: st.session_state.y = None
if "x_original" not in st.session_state: st.session_state.x_original = None
if "y_original" not in st.session_state: st.session_state.y_original = None
if "peaks" not in st.session_state: st.session_state.peaks = []
if "y_range" not in st.session_state: st.session_state.y_range = None
if "visual_settings" not in st.session_state: st.session_state.visual_settings = {}

dec = SpectralDeconvolution()

st.title("📊 Deconvolução Espectral Avançada Pro")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Painel de Controle")
    tab_data, tab_preproc, tab_peaks, tab_fit, tab_visual = st.tabs(["📁 Dados", "🔧 Pré-proc.", "🔍 Picos", "🎯 Ajuste", "🎨 Visual"])

    with tab_data:
        st.subheader("📁 Carregar Dados")
        up = st.file_uploader("CSV/TXT/Excel", type=["csv", "txt", "xlsx", "xls"])
        if up is None and st.session_state.df is None:
            st.info("Carregando dados de exemplo...")
            st.session_state.df = synthetic_example()
        if up is not None:
            try:
                if up.name.lower().endswith((".csv", ".txt")):
                    sample = up.read(2000).decode('utf-8', errors='ignore')
                    up.seek(0)
                    separators = {'\t': sample.count('\t'), ',': sample.count(','), ';': sample.count(';'), ' ': sample.count(' ')}
                    detected_sep = max(separators, key=separators.get)
                    dots = sample.count('.')
                    commas = sample.count(',')
                    if detected_sep == ',':
                        detected_decimal = '.'
                    elif detected_sep == ';':
                        detected_decimal = ','
                    else:
                        if commas > dots * 2:
                            detected_decimal = ','
                        else:
                            detected_decimal = '.'
                    sep_names = {'\t': 'Tabulação (TAB)', ',': 'Vírgula (,)', ';': 'Ponto-e-vírgula (;)', ' ': 'Espaço'}
                    st.info(f"🔍 Detectado: {sep_names.get(detected_sep, 'TAB')} | Decimal = {detected_decimal}")
                    col1, col2 = st.columns(2)
                    with col1:
                        sep = st.selectbox("Separador", ["\t", ",", ";", " "], 
                                         index=["\t", ",", ";", " "].index(detected_sep),
                                         format_func=lambda x: sep_names.get(x, x))
                    with col2:
                        decimal = st.selectbox("Decimal", [".", ","], 
                                             index=[".", ","].index(detected_decimal))
                    df = pd.read_csv(up, decimal=decimal, sep=sep, engine="python", header=None)
                else:
                    names = excel_sheet_names(up)
                    sheet = st.selectbox("Planilha", names) if names else 0
                    df = pd.read_excel(up, sheet_name=sheet, header=st.number_input("Linha cabeçalho", 0, 100, 0))
                st.session_state.df = df
                st.session_state.peaks = []
            except Exception as exc: 
                st.error(f"Erro ao ler: {exc}")
        
        if st.session_state.df is not None:
            df = coerce_numeric_df(st.session_state.df)
            st.dataframe(df.head(10), height=200)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                colx = st.selectbox("Coluna X", numeric_cols, 0)
                coly = st.selectbox("Coluna Y", numeric_cols, min(1, len(numeric_cols)-1))
                x,y = df[colx].dropna().to_numpy(dtype=float), df[coly].dropna().to_numpy(dtype=float)
                idx = np.argsort(x); x, y = x[idx], y[idx]
                st.session_state.x_original, st.session_state.y_original = x.copy(), y.copy()
                st.session_state.x, st.session_state.y = x.copy(), y.copy()
                
        if st.session_state.y is not None:
            y_min_auto, y_max_auto = float(np.nanmin(st.session_state.y)), float(np.nanmax(st.session_state.y))
            st.session_state.y_range = st.slider("Intervalo Eixo Y", y_min_auto, y_max_auto, (y_min_auto, y_max_auto))

    with tab_preproc:
        st.subheader("🔧 Pré-processamento")
        
        # Mostra status
        if st.session_state.y is None:
            st.error("⚠️ Nenhum dado carregado. Vá para a aba 'Dados'")
        else:
            st.success(f"✅ Dados carregados: {len(st.session_state.y)} pontos")
            
            baseline_method = st.selectbox("Linha Base", ["none", "linear", "polynomial", "moving_average"])
            poly_degree = st.slider("Grau (polinomial)", 1, 10, 3) if baseline_method == 'polynomial' else 3
            ma_window_base = st.slider("Janela (média móvel)", 10, 200, 50, 10) if baseline_method == 'moving_average' else 50
            
            st.markdown("---")
            smooth_method = st.selectbox("Suavização", ["none", "savgol", "moving_average"])
            sg_window = st.slider("Janela (Savgol)", 5, 51, 11, 2) if smooth_method == 'savgol' else 11
            sg_poly = st.slider("Grau Polinômio", 1, 5, 3) if smooth_method == 'savgol' else 3
            ma_window_smooth = st.slider("Janela (média)", 3, 51, 5, 2) if smooth_method == 'moving_average' else 5
            
            st.markdown("---")
            norm_method = st.selectbox("Normalização", ["none", "max", "area", "minmax"])
            
            st.markdown("---")
            if st.button("✅ APLICAR PRÉ-PROCESSAMENTO", type="primary", use_container_width=True, key="btn_preproc"):
                try:
                    x = st.session_state.x_original.copy()
                    y = st.session_state.y_original.copy()
                    
                    if baseline_method != "none":
                        if baseline_method == "polynomial":
                            y, _ = baseline_correction(x, y, baseline_method, degree=poly_degree)
                        elif baseline_method == "moving_average":
                            y, _ = baseline_correction(x, y, baseline_method, window=ma_window_base)
                        else:
                            y, _ = baseline_correction(x, y, baseline_method)
                        st.success(f"✅ Linha base: {baseline_method}")
                    
                    if smooth_method != "none":
                        if smooth_method == "savgol":
                            y = smooth_spectrum(x, y, smooth_method, window=sg_window, poly=sg_poly)
                        elif smooth_method == "moving_average":
                            y = smooth_spectrum(x, y, smooth_method, window=ma_window_smooth)
                        st.success(f"✅ Suavização: {smooth_method}")
                    
                    if norm_method != "none":
                        y = normalize_spectrum(y, norm_method)
                        st.success(f"✅ Normalização: {norm_method}")
                    
                    st.session_state.x = x
                    st.session_state.y = y
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erro: {e}")

    with tab_peaks:
        st.subheader("🔍 Gerenciamento de Picos")
        
        num_picos = len(st.session_state.peaks)
        if num_picos > 0:
            st.success(f"📊 {num_picos} pico(s)")
        
        # DETECÇÃO AUTOMÁTICA
        st.markdown("### 🔎 Detecção Automática")
        col1, col2 = st.columns(2)
        with col1:
            prom = st.number_input("Proeminência", 0.0, 1.0, 0.05, 0.01, format="%.3f", key="prom_detect")
        with col2:
            dist = st.number_input("Distância", 5, 200, 30, 5, key="dist_detect")
        
        if st.button("🔍 DETECTAR PICOS", type="primary", use_container_width=True, key="btn_detect"):
            if st.session_state.y is None:
                st.error("❌ Carregue dados primeiro!")
            else:
                try:
                    y_norm = st.session_state.y / np.max(st.session_state.y)
                    pks, _ = find_peaks(y_norm, prominence=prom, distance=dist)
                    
                    if len(pks) == 0:
                        st.warning("⚠️ Nenhum pico detectado")
                    else:
                        y_max = float(np.max(st.session_state.y))
                        x_min, x_max = float(st.session_state.x.min()), float(st.session_state.x.max())
                        x_range = x_max - x_min
                        default_width = x_range / 30.0
                        
                        st.session_state.peaks = []
                        
                        for idx in pks:
                            amplitude = float(st.session_state.y[idx])
                            center = float(st.session_state.x[idx])
                            
                            try:
                                half_max = amplitude / 2.0
                                left_idx = idx
                                while left_idx > 0 and st.session_state.y[left_idx] > half_max:
                                    left_idx -= 1
                                right_idx = idx
                                while right_idx < len(st.session_state.y) - 1 and st.session_state.y[right_idx] > half_max:
                                    right_idx += 1
                                
                                fwhm = abs(st.session_state.x[right_idx] - st.session_state.x[left_idx])
                                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                                
                                if sigma < 1e-6 or sigma > x_range:
                                    sigma = default_width
                            except:
                                sigma = default_width
                            
                            st.session_state.peaks.append({
                                "type": "Gaussiana",
                                "params": [amplitude, center, sigma],
                                "bounds": [
                                    (amplitude * 0.1, amplitude * 3.0),
                                    (center - x_range * 0.1, center + x_range * 0.1),
                                    (sigma * 0.1, sigma * 10.0)
                                ]
                            })
                        
                        st.success(f"✅ {len(pks)} pico(s) detectado(s)!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
        
        # ADICIONAR MANUAL
        st.markdown("---")
        st.markdown("### ➕ Adicionar Manual")
        pk_type = st.selectbox("Tipo de Pico", list(dec.peak_models.keys()), key="peak_type_select")
        
        if st.button("➕ ADICIONAR PICO MANUAL", type="secondary", use_container_width=True, key="btn_add_manual"):
            if st.session_state.y is None:
                st.error("❌ Carregue dados primeiro!")
            else:
                try:
                    y_max = float(st.session_state.y.max())
                    x_min, x_max = float(st.session_state.x.min()), float(st.session_state.x.max())
                    x_center = float(np.mean(st.session_state.x))
                    x_range = x_max - x_min
                    default_width = x_range / 20.0
                    
                    if pk_type == "Gaussiana":
                        params = [y_max/3, x_center, default_width]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range)]
                    elif pk_type == "Lorentziana":
                        params = [y_max/3, x_center, default_width]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range)]
                    elif pk_type == "Voigt (exato)":
                        params = [y_max/3, x_center, default_width, default_width]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (1e-6, x_range)]
                    elif pk_type == "Pseudo-Voigt":
                        params = [y_max/3, x_center, default_width, 0.5]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (0, 1)]
                    elif pk_type == "Gaussiana Assimétrica":
                        params = [y_max/3, x_center, default_width, default_width]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (1e-6, x_range)]
                    elif pk_type == "Pearson VII":
                        params = [y_max/3, x_center, default_width, 2.0]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (0.5, 10)]
                    elif pk_type == "Gaussiana Exponencial":
                        params = [y_max/3, x_center, default_width, default_width]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (1e-6, x_range)]
                    else:  # Doniach-Sunjic
                        params = [y_max/3, x_center, default_width, 0.1]
                        bounds = [(0, y_max*2), (x_min, x_max), (1e-6, x_range), (0, 1)]
                    
                    st.session_state.peaks.append({
                        "type": pk_type, 
                        "params": params, 
                        "bounds": bounds
                    })
                    st.success(f"✅ Pico {pk_type} adicionado!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
        
        # LISTA DE PICOS
        if len(st.session_state.peaks) > 0:
            st.markdown("---")
            st.markdown("### 📝 Picos Configurados")
            for i in range(len(st.session_state.peaks)):
                pk = st.session_state.peaks[i]
                with st.expander(f"Pico {i+1}: {pk['type']}", expanded=False):
                    param_names = dec.peak_models[pk["type"]][1]
                    for j, p_name in enumerate(param_names):
                        new_val = st.number_input(
                            p_name, 
                            value=float(pk["params"][j]), 
                            format="%.6f", 
                            key=f"p_{i}_{j}"
                        )
                        st.session_state.peaks[i]["params"][j] = new_val
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Remover", key=f"rem_{i}", use_container_width=True):
                            st.session_state.peaks.pop(i)
                            st.rerun()
                    with col2:
                        if st.button(f"📋 Duplicar", key=f"dup_{i}", use_container_width=True):
                            st.session_state.peaks.append({
                                "type": pk["type"],
                                "params": pk["params"].copy(),
                                "bounds": pk["bounds"].copy()
                            })
                            st.rerun()
            
            st.markdown("---")
            if st.button("🗑️ LIMPAR TODOS", type="secondary", use_container_width=True, key="btn_clear_all"):
                st.session_state.peaks = []
                st.rerun()

    with tab_fit:
        st.subheader("🎯 Ajuste dos Parâmetros")
        
        if len(st.session_state.peaks) == 0:
            st.error("⚠️ Adicione picos primeiro!")
        else:
            st.info(f"📊 {len(st.session_state.peaks)} pico(s) configurado(s)")
            
            fit_method = st.selectbox("Método", ["curve_fit", "differential_evolution", "minimize"], key="fit_method_select")
            
            if fit_method == "curve_fit":
                maxfev = st.number_input("Máx avaliações", 1000, 100000, 20000, 1000, key="maxfev_input")
                algorithm = st.selectbox("Algoritmo", ["trf", "dogbox", "lm"], key="algo_cf")
            elif fit_method == "differential_evolution":
                maxiter = st.number_input("Máx iterações", 100, 5000, 1000, 100, key="maxiter_de")
                algorithm = None
                maxfev = None
            else:
                algorithm = st.selectbox("Algoritmo", ["L-BFGS-B", "TNC", "SLSQP"], key="algo_min")
                maxfev = None
                maxiter = None
            
            if st.button("🚀 EXECUTAR AJUSTE", type="primary", use_container_width=True, key="btn_fit"):
                with st.spinner("⏳ Otimizando..."):
                    try:
                        kwargs = {}
                        if fit_method == "curve_fit":
                            kwargs["maxfev"] = maxfev
                            kwargs["algorithm"] = algorithm
                        elif fit_method == "differential_evolution":
                            kwargs["maxiter"] = maxiter if 'maxiter' in locals() else 1000
                        elif fit_method == "minimize":
                            kwargs["algorithm"] = algorithm
                        
                        flat_params, pcov = dec.fit(
                            st.session_state.x, 
                            st.session_state.y, 
                            st.session_state.peaks, 
                            fit_method,
                            **kwargs
                        )
                        
                        if flat_params is None:
                            st.error("❌ Ajuste falhou")
                        else:
                            pos = 0
                            for i, pk in enumerate(st.session_state.peaks):
                                n = len(dec.peak_models[pk["type"]][1])
                                new_params = [float(v) for v in flat_params[pos:pos+n]]
                                st.session_state.peaks[i]["params"] = new_params
                                pos += n
                            
                            y_fit = dec.create_composite(st.session_state.peaks)(st.session_state.x, *flat_params)
                            residuals = st.session_state.y - y_fit
                            ss_res = np.sum(residuals**2)
                            ss_tot = np.sum((st.session_state.y - np.mean(st.session_state.y))**2)
                            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            rmse = np.sqrt(np.mean(residuals**2))
                            
                            st.success("✅ Ajuste concluído!")
                            st.metric("R²", f"{r2:.4f}")
                            st.metric("RMSE", f"{rmse:.4f}")
                            
                            if r2 < 0.8:
                                st.warning("⚠️ R² < 0.8. Ajuste pode melhorar.")
                            
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")

    with tab_visual:
        st.subheader("🎨 Customização Visual")
        vs = st.session_state.visual_settings
        
        vs["color_scheme"] = st.selectbox("Tema", ["default", "scientific", "dark", "publication"], key="color_scheme_sel")
        vs["component_palette"] = st.selectbox("Paleta", 
            ["Plotly", "Okabe-Ito", "Viridis", "Plasma", "Inferno"], key="palette_sel")
        vs["fill_areas"] = st.checkbox("Preencher áreas", value=True, key="fill_check")
        vs["comp_opacity"] = st.slider("Opacidade", 0.1, 1.0, 0.4, key="opacity_slider")
        vs["transparent_bg"] = st.checkbox("Fundo Transparente", False, key="transp_check")
        
        st.markdown("---")
        vs["title"] = st.text_input("Título", "Deconvolução Espectral", key="title_input")
        vs["x_label"] = st.text_input("Eixo X", "X", key="xlabel_input")
        vs["y_label"] = st.text_input("Eixo Y", "Intensidade", key="ylabel_input")
        
        st.markdown("---")
        vs["show_fit"] = st.checkbox("Mostrar ajuste", True, key="show_fit_check")
        vs["show_components"] = st.checkbox("Mostrar componentes", True, key="show_comp_check")
        vs["show_centers"] = st.checkbox("Mostrar centros", False, key="show_center_check")
        vs["show_residuals"] = st.checkbox("Mostrar resíduos", False, key="show_res_check")
        
        vs["show_grid"] = st.checkbox("Grade", True, key="grid_check")
        vs["show_legend"] = st.checkbox("Legenda", True, key="legend_check")

# Main Content & Plot
visual_settings = st.session_state.get("visual_settings", {})
visual_settings.setdefault("show_fit", True)
visual_settings.setdefault("show_components", True)
visual_settings.setdefault("show_centers", False)
visual_settings.setdefault("line_width", 2)
visual_settings.setdefault("marker_size", 4)
visual_settings.setdefault("plot_style", "lines")
visual_settings.setdefault("title_size", 20)
visual_settings.setdefault("label_size", 14)
visual_settings.setdefault("tick_size", 12)

col_main, col_stats = st.columns([3, 1])

with col_main:
    if len(st.session_state.peaks) > 0:
        opts = ["Nenhum"] + [f"{i+1}. {p['type']}" for i,p in enumerate(st.session_state.peaks)]
        sel = st.selectbox("🔦 Pico em destaque", opts, 0)
        visual_settings["highlight_idx"] = None if sel == "Nenhum" else int(sel.split(".")[0]) - 1
    
    visual_settings["y_range"] = st.session_state.get("y_range")
    if st.session_state.x is not None:
        fig, y_fit_total = plot_figure(st.session_state.x, st.session_state.y, st.session_state.peaks, dec, settings=visual_settings)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Carregue dados para começar.")

with col_stats:
    st.markdown("### 📊 Estatísticas")
    if len(st.session_state.peaks) > 0 and 'y_fit_total' in locals():
        res = st.session_state.y - y_fit_total
        ss_res, ss_tot = np.sum(res**2), np.sum((st.session_state.y - np.mean(st.session_state.y))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        st.metric("R²", f"{r2:.4f}")
        st.metric("RMSE", f"{np.sqrt(np.mean(res**2)):.4f}")
        st.metric("Nº Picos", len(st.session_state.peaks))

# Results and Export
st.markdown("---")
tab_results, tab_export = st.tabs(["📊 Resultados", "💾 Exportação"])

with tab_results:
    if not st.session_state.peaks:
        st.info("Adicione picos e execute o ajuste")
    else:
        rows = []
        x = st.session_state.x
        y_total = np.sum([dec._eval_single(x, pk["type"], pk["params"]) for pk in st.session_state.peaks], axis=0)
        total_area = np.trapz(y_total, x) if np.any(y_total) else 1.0
        for i, pk in enumerate(st.session_state.peaks, 1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            area = area_under_peak(x, y_comp, pk["type"], pk["params"])
            fwhm = fwhm_of_peak(pk['type'], pk['params'])
            rows.append({
                "Pico": i, 
                "Tipo": pk["type"], 
                "Amplitude": f"{pk['params'][0]:.4f}", 
                "Centro": f"{pk['params'][1]:.4f}", 
                "FWHM": f"{fwhm:.4f}" if fwhm else "N/A", 
                "Área": f"{area:.4f}", 
                "Área (%)": f"{100*area/total_area:.2f}"
            })
        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

with tab_export:
    if not st.session_state.peaks:
        st.info("Execute o ajuste para exportar")
    else:
        st.markdown("### 📤 Exportar Figura")
        col1, col2 = st.columns(2)
        fmt = col1.selectbox("Formato", ["PNG", "JPEG", "SVG"], 0, key="export_fmt")
        preset = col2.selectbox("Resolução", ["1080p (1920x1080)","2K (2560x1440)","4K (3840x2160)"], 1, key="export_res")
        export_w, export_h = {"1080p (1920x1080)": (1920,1080), "2K (2560x1440)": (2560,1440), "4K (3840x2160)": (3840,2160)}[preset]
        
        export_settings = visual_settings.copy()
        fig_exp, _ = plot_figure(st.session_state.x, st.session_state.y, st.session_state.peaks, dec, settings=export_settings)
        plotly_download_button(fig_exp, f"deconv.{fmt.lower()}", fmt.lower(), export_w, export_h, 2.0)
