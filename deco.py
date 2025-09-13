# -*- coding: utf-8 -*-
import io
import warnings
from typing import List, Dict, Any, Optional
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.signal import find_peaks, savgol_filter
from scipy.special import wofz
from scipy.interpolate import interp1d
import json

# -------------------------------------------
# Data Loading Helpers (robust I/O)
# -------------------------------------------
def coerce_numeric_series(s: pd.Series):
    if s.dtype == object:
        s2 = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s2, errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")

def coerce_numeric_df(df: pd.DataFrame):
    out = df.copy()
    for c in out.columns:
        out[c] = coerce_numeric_series(out[c])
    return out

def excel_sheet_names(file):
    try:
        xls = pd.ExcelFile(file)
        return xls.sheet_names
    except Exception:
        return None

warnings.filterwarnings("ignore")

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

# -------------------------------------------
# Page Config
# -------------------------------------------
st.set_page_config(
    page_title="Deconvolução Espectral Avançada Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better layout
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    div[data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 450px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# Helpers (compatibility & math)
# -------------------------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

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
    """Exponentially modified Gaussian"""
    from scipy.special import erfc
    y = (sigma / tau) * np.sqrt(np.pi / 2) * np.exp(0.5 * (sigma / tau)**2 - (x - center) / tau)
    y *= erfc((sigma / tau - (x - center) / sigma) / np.sqrt(2))
    return amplitude * y / np.max(y) if np.max(y) > 0 else np.zeros_like(x)

def doniach_sunjic(x, amplitude, center, gamma, alpha):
    """Doniach-Sunjic line shape for XPS"""
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

# -------------------------------------------
# Preprocessing Functions
# -------------------------------------------
def baseline_correction(x, y, method="linear", **kwargs):
    """Correção de linha de base"""
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
    """Normalização do espectro"""
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
    """Suavização do espectro"""
    if method == "savgol":
        window = kwargs.get("window", 11)
        poly = kwargs.get("poly", 3)
        if window % 2 == 0:
            window += 1
        return savgol_filter(y, window, poly)
    elif method == "moving_average":
        window = kwargs.get("window", 5)
        return np.convolve(y, np.ones(window)/window, mode='same')
    return y

# -------------------------------------------
# Deconvolution Engine
# -------------------------------------------
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
        if peak_type == "Gaussiana":
            return gaussian(x, *params)
        if peak_type == "Lorentziana":
            return lorentzian(x, *params)
        if peak_type == "Voigt (exato)":
            return voigt_exact(x, *params)
        if peak_type == "Pseudo-Voigt":
            return pseudo_voigt(x, *params)
        if peak_type == "Gaussiana Assimétrica":
            return asymmetric_gaussian(x, *params)
        if peak_type == "Pearson VII":
            return pearson_vii(x, *params)
        if peak_type == "Gaussiana Exponencial":
            return exponential_gaussian(x, *params)
        if peak_type == "Doniach-Sunjic":
            return doniach_sunjic(x, *params)
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
                def objective(vec):
                    return np.sum((y - comp(x, *vec)) ** 2)
                de_bounds = list(zip(bounds[0], bounds[1]))
                res = differential_evolution(objective, de_bounds, 
                                           seed=kwargs.get("seed", 42),
                                           maxiter=kwargs.get("maxiter", 1000),
                                           popsize=kwargs.get("popsize", 15))
                return res.x, None
            elif method == "minimize":
                def objective(vec):
                    return np.sum((y - comp(x, *vec)) ** 2)
                res = minimize(objective, p0, 
                             method=kwargs.get("algorithm", "L-BFGS-B"),
                             bounds=list(zip(bounds[0], bounds[1])))
                return res.x, None
        except Exception as exc:
            st.error(f"Erro no ajuste: {exc}")
            return np.array(p0, dtype=float), None

def plot_figure(x, y, peaks, dec, show_fit=True, show_components=True, 
                show_residuals=True, y_range=None, highlight_idx=None, 
                comp_opacity=0.35, fill_areas=False, show_centers=True,
                plot_style="default", color_scheme="default", 
                line_width=2, marker_size=4, show_grid=True,
                show_legend=True, legend_position="topright",
                title="Deconvolução Espectral", x_label="X", y_label="Intensidade"):
    """Build figure with advanced customization options"""
    
    # Color schemes
    color_schemes = {
        "default": {
            "data": "#4DA3FF",
            "sum": "#FF6EC7",
            "residuals": "#FF4D4D",
            "highlight": "#FFD166",
            "components": "#A0AEC0"
        },
        "scientific": {
            "data": "#1f77b4",
            "sum": "#ff7f0e",
            "residuals": "#2ca02c",
            "highlight": "#d62728",
            "components": "#9467bd"
        },
        "dark": {
            "data": "#00D9FF",
            "sum": "#FF00FF",
            "residuals": "#00FF00",
            "highlight": "#FFFF00",
            "components": "#808080"
        },
        "publication": {
            "data": "#000000",
            "sum": "#E74C3C",
            "residuals": "#3498DB",
            "highlight": "#F39C12",
            "components": "#95A5A6"
        }
    }
    
    colors = color_schemes.get(color_scheme, color_schemes["default"])
    
    fig = go.Figure()
    
    # Data trace
    if plot_style == "lines":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Dados", 
                                line=dict(width=line_width, color=colors["data"])))
    elif plot_style == "markers":
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Dados",
                                marker=dict(size=marker_size, color=colors["data"])))
    else:  # lines+markers
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Dados",
                                line=dict(width=line_width, color=colors["data"]),
                                marker=dict(size=marker_size, color=colors["data"])))
    
    y_fit_total = None
    shapes = []

    if show_fit and len(peaks) > 0:
        y_fit_total = np.zeros_like(x, dtype=float)
        for i, pk in enumerate(peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            y_fit_total += y_comp

            if show_components:
                is_h = (highlight_idx is not None and (i-1) == highlight_idx)
                line_style = dict(
                    width=line_width+1 if is_h else line_width-1, 
                    dash="solid" if is_h else "dot",
                    color=colors["highlight"] if is_h else colors["components"]
                )
                name = f"{pk['type']} #{i}" + (" (★)" if is_h else "")
                
                if fill_areas and not is_h:
                    fig.add_trace(go.Scatter(x=x, y=y_comp, mode="lines", name=name,
                                           line=line_style, opacity=comp_opacity,
                                           fill="tozeroy", fillcolor=f"rgba(160,174,192,{comp_opacity*0.3})"))
                else:
                    fig.add_trace(go.Scatter(x=x, y=y_comp, mode="lines", name=name,
                                           line=line_style, opacity=1.0 if is_h else comp_opacity))

                if show_centers:
                    cx = float(pk["params"][1])
                    y0 = float(y.min() if y_range is None else y_range[0])
                    y1 = float(y.max() if y_range is None else y_range[1])
                    shapes.append(dict(type="line", x0=cx, x1=cx, y0=y0, y1=y1,
                                     line=dict(color=colors["highlight"] if is_h else "#666", 
                                             width=1, dash="dash")))

        fig.add_trace(go.Scatter(x=x, y=y_fit_total, mode="lines", name="Soma Ajuste",
                               line=dict(width=line_width+1, color=colors["sum"])))

        if show_residuals:
            res = y - y_fit_total
            fig.add_trace(go.Scatter(x=x, y=res, mode="lines", name="Resíduos",
                                   line=dict(width=line_width-1, color=colors["residuals"]), 
                                   yaxis="y2"))

    # Legend position mapping
    legend_positions = {
        "topright": dict(y=1, x=1.02, yanchor="top"),
        "topleft": dict(y=1, x=0, yanchor="top"),
        "bottomright": dict(y=0, x=1, yanchor="bottom"),
        "bottomleft": dict(y=0, x=0, yanchor="bottom"),
        "outside": dict(y=0.5, x=1.1, yanchor="middle")
    }
    
    layout = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=650,
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="v", **legend_positions.get(legend_position, legend_positions["topright"])),
        shapes=shapes,
        plot_bgcolor='white' if color_scheme != "dark" else '#1a1a1a',
        paper_bgcolor='white' if color_scheme != "dark" else '#2a2a2a'
    )
    
    if show_grid:
        layout["xaxis"] = dict(title=x_label, showgrid=True, gridcolor='#E0E0E0')
        layout["yaxis"] = dict(title=y_label, showgrid=True, gridcolor='#E0E0E0')
    else:
        layout["xaxis"] = dict(title=x_label, showgrid=False)
        layout["yaxis"] = dict(title=y_label, showgrid=False)
    
    if show_residuals:
        layout["yaxis2"] = dict(overlaying="y", side="right", title="Resíduos", 
                              showgrid=False, zeroline=True)
    
    if y_range is not None:
        layout["yaxis"]["range"] = y_range

    fig.update_layout(**layout)
    return fig, y_fit_total

# -------------------------------------------
# Session initialization
# -------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "x" not in st.session_state:
    st.session_state.x = None
if "y" not in st.session_state:
    st.session_state.y = None
if "x_original" not in st.session_state:
    st.session_state.x_original = None
if "y_original" not in st.session_state:
    st.session_state.y_original = None
if "peaks" not in st.session_state:
    st.session_state.peaks = []
if "fit_params" not in st.session_state:
    st.session_state.fit_params = None
if "y_range" not in st.session_state:
    st.session_state.y_range = None
if "preprocessing" not in st.session_state:
    st.session_state.preprocessing = {"baseline": None, "normalized": False, "smoothed": False}
if "fit_history" not in st.session_state:
    st.session_state.fit_history = []

dec = SpectralDeconvolution()

# -------------------------------------------
# Main Title
# -------------------------------------------
st.title("📊 Deconvolução Espectral Avançada Pro")
st.markdown("---")

# -------------------------------------------
# Layout: Sidebar (left) and Main area (right)
# -------------------------------------------
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    # Tabs in sidebar
    tab_data, tab_preproc, tab_peaks, tab_fit, tab_visual = st.tabs(
        ["📁 Dados", "🔧 Pré-proc.", "📍 Picos", "🎯 Ajuste", "🎨 Visual"]
    )
    
    # ===== TAB: DADOS =====
    with tab_data:
        st.subheader("📁 Carregar Dados")
        up = st.file_uploader("CSV/TXT/Excel", type=["csv", "txt", "xlsx", "xls"])
        
        sheet = None
        header_row = 0
        decimal_csv = st.selectbox("Separador decimal (CSV/TXT)", 
                                  options=[",", "."], index=1)
        transpose = st.checkbox("Dados transpostos", value=False)
        force_numeric = st.checkbox("Forçar numérico", value=True)
        sort_x = st.checkbox("Ordenar por X", value=True)
        
        if up is not None and (up.name.lower().endswith(".xlsx") or up.name.lower().endswith(".xls")):
            names = excel_sheet_names(up)
            if names:
                sheet = st.selectbox("Planilha", names, index=0)
            header_row = st.number_input("Linha do cabeçalho", min_value=1, value=1, step=1) - 1
        
        # Load data
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    df = pd.read_csv(up, decimal=decimal_csv, sep=None, engine="python")
                elif up.name.lower().endswith(".txt"):
                    try:
                        df = pd.read_csv(up, sep="\t", decimal=decimal_csv)
                    except:
                        up.seek(0)
                        df = pd.read_csv(up, sep=r"\s+", engine="python", decimal=decimal_csv)
                else:
                    df = pd.read_excel(up, sheet_name=sheet if sheet else 0, header=header_row)
                
                if transpose:
                    df = df.T
                    df.reset_index(drop=False, inplace=True)
                if force_numeric:
                    df = coerce_numeric_df(df)
                st.session_state.df = df
            except Exception as exc:
                st.error(f"Erro ao ler arquivo: {exc}")
        
        if st.session_state.df is None:
            st.info("Carregando dados de exemplo...")
            st.session_state.df = synthetic_example()
        
        df = st.session_state.df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        if len(numeric_cols) < 2:
            df_num = coerce_numeric_df(df)
            numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
            if len(numeric_cols) >= 2:
                df = df_num
                st.session_state.df = df
        
        st.write("**Prévia dos dados:**")
        st.dataframe(df.head(10), use_container_width=True, height=200)
        
        if len(numeric_cols) < 2:
            st.error("Pelo menos 2 colunas numéricas são necessárias.")
            numeric_cols = list(df.columns)
        
        colx = st.selectbox("Coluna X", numeric_cols, index=0)
        coly = st.selectbox("Coluna Y", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        x = df[colx].to_numpy(dtype=float)
        y = df[coly].to_numpy(dtype=float)
        
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if sort_x and x.size > 1:
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
        
        st.session_state.x_original = x.copy()
        st.session_state.y_original = y.copy()
        st.session_state.x = x
        st.session_state.y = y
        
        # Y-axis range controls
        st.markdown("### 📏 Eixo Y")
        col1, col2 = st.columns(2)
        with col1:
            y_min = st.number_input("Y min", value=float(np.nanmin(y)) if y.size else 0.0, 
                                   step=0.1, format="%.4f")
        with col2:
            y_max = st.number_input("Y max", value=float(np.nanmax(y)) if y.size else 1.0, 
                                   step=0.1, format="%.4f")
        
        if st.button("🔄 Auto Range", use_container_width=True):
            y_min = float(np.nanmin(y)) if y.size else 0.0
            y_max = float(np.nanmax(y)) if y.size else 1.0
        
        st.session_state.y_range = [y_min, y_max]
    
    # ===== TAB: PRÉ-PROCESSAMENTO =====
    with tab_preproc:
        st.subheader("🔧 Pré-processamento")
        
        # Baseline correction
        st.markdown("**Correção de Linha Base**")
        baseline_method = st.selectbox("Método", 
                                      ["none", "linear", "polynomial", "moving_average"])
        
        if baseline_method == "polynomial":
            poly_degree = st.slider("Grau do polinômio", 1, 10, 3)
        elif baseline_method == "moving_average":
            ma_window = st.slider("Janela", 10, 200, 50)
        
        # Smoothing
        st.markdown("**Suavização**")
        smooth_method = st.selectbox("Método", 
                                    ["none", "savgol", "moving_average"])
        
        if smooth_method == "savgol":
            sg_window = st.slider("Janela (ímpar)", 5, 51, 11, step=2)
            sg_poly = st.slider("Ordem polinomial", 1, 5, 3)
        elif smooth_method == "moving_average":
            sma_window = st.slider("Janela", 3, 21, 5)
        
        # Normalization
        st.markdown("**Normalização**")
        norm_method = st.selectbox("Método", 
                                  ["none", "max", "area", "minmax"])
        
        # Apply preprocessing
        if st.button("Aplicar Pré-processamento", type="primary", use_container_width=True):
            x = st.session_state.x_original.copy()
            y = st.session_state.y_original.copy()
            
            # Apply baseline correction
            if baseline_method != "none":
                if baseline_method == "polynomial":
                    y, _ = baseline_correction(x, y, "polynomial", degree=poly_degree)
                elif baseline_method == "moving_average":
                    y, _ = baseline_correction(x, y, "moving_average", window=ma_window)
                else:
                    y, _ = baseline_correction(x, y, baseline_method)
            
            # Apply smoothing
            if smooth_method != "none":
                if smooth_method == "savgol":
                    y = smooth_spectrum(x, y, "savgol", window=sg_window, poly=sg_poly)
                elif smooth_method == "moving_average":
                    y = smooth_spectrum(x, y, "moving_average", window=sma_window)
            
            # Apply normalization
            if norm_method != "none":
                y = normalize_spectrum(y, norm_method)
            
            st.session_state.x = x
            st.session_state.y = y
            st.session_state.preprocessing = {
                "baseline": baseline_method,
                "smooth": smooth_method,
                "normalize": norm_method
            }
            st.success("Pré-processamento aplicado!")
            safe_rerun()
        
        # Reset preprocessing
        if st.button("🔄 Resetar", use_container_width=True):
            st.session_state.x = st.session_state.x_original.copy()
            st.session_state.y = st.session_state.y_original.copy()
            st.session_state.preprocessing = {"baseline": None, "normalized": False, "smoothed": False}
            safe_rerun()
    
    # ===== TAB: PICOS =====
    with tab_peaks:
        st.subheader("📍 Gerenciamento de Picos")
        
        # Auto-detection
        st.markdown("**🔍 Detecção Automática**")
        col1, col2 = st.columns(2)
        with col1:
            prom = st.number_input("Proeminência", value=0.05, step=0.01, format="%.3f")
        with col2:
            dist = st.number_input("Distância mín.", value=30, step=1, min_value=1)
        
        if st.button("Detectar Picos", use_container_width=True):
            y = st.session_state.y
            x = st.session_state.x
            pks, props = find_peaks(y, prominence=prom, distance=int(dist))
            st.session_state.peaks = []
            
            if len(pks) == 0:
                st.warning("Nenhum pico detectado.")
            else:
                x_min, x_max = float(np.min(x)), float(np.max(x))
                xr = x_max - x_min
                
                for idx in pks:
                    amp = float(y[idx])
                    xc = float(x[idx])
                    
                    # Estimate width at half maximum
                    half = amp / 2.0
                    li, ri = idx, idx
                    while li > 0 and y[li] > half:
                        li -= 1
                    while ri < len(y) - 1 and y[ri] > half:
                        ri += 1
                    width = max(1e-6, float(x[min(ri, len(x)-1)] - x[max(li, 0)]))
                    sigma_guess = max(width / 2.355, xr / 200.0)
                    
                    st.session_state.peaks.append({
                        "type": "Gaussiana",
                        "params": [amp, xc, sigma_guess],
                        "bounds": [(0.0, amp * 2.0), (x_min, x_max), (1e-6, xr)]
                    })
                st.success(f"✅ {len(pks)} picos detectados")
        
        st.markdown("---")
        
        # Manual peak addition
        st.markdown("**➕ Adicionar Pico Manual**")
        pk_type = st.selectbox("Tipo de pico", list(dec.peak_models.keys()))
        
        if st.button("Adicionar Pico", use_container_width=True):
            x_min = float(np.min(st.session_state.x))
            x_max = float(np.max(st.session_state.x))
            xr = x_max - x_min
            y_max = float(np.max(st.session_state.y))
            
            # Initialize parameters based on peak type
            if pk_type == "Gaussiana":
                params = [y_max/3, (x_min+x_max)/2, xr/20]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr)]
            elif pk_type == "Lorentziana":
                params = [y_max/3, (x_min+x_max)/2, xr/20]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr)]
            elif pk_type == "Voigt (exato)":
                params = [y_max/3, (x_min+x_max)/2, xr/30, xr/30]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (1e-6, xr)]
            elif pk_type == "Pseudo-Voigt":
                params = [y_max/3, (x_min+x_max)/2, xr/15, 0.5]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.0, 1.0)]
            elif pk_type == "Gaussiana Assimétrica":
                params = [y_max/3, (x_min+x_max)/2, xr/30, xr/30]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (1e-6, xr)]
            elif pk_type == "Pearson VII":
                params = [y_max/3, (x_min+x_max)/2, xr/20, 1.5]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.5, 10.0)]
            elif pk_type == "Gaussiana Exponencial":
                params = [y_max/3, (x_min+x_max)/2, xr/20, xr/10]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (1e-6, xr)]
            else:  # Doniach-Sunjic
                params = [y_max/3, (x_min+x_max)/2, xr/20, 0.2]
                bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.0, 0.5)]
            
            st.session_state.peaks.append({"type": pk_type, "params": params, "bounds": bounds})
            st.success(f"✅ Pico {pk_type} adicionado")
            safe_rerun()
        
        st.markdown("---")
        
        # Peak list management
        st.markdown("**📋 Lista de Picos**")
        if len(st.session_state.peaks) > 0:
            for i, pk in enumerate(st.session_state.peaks):
                with st.expander(f"Pico {i+1}: {pk['type']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Centro: {pk['params'][1]:.3f}")
                        st.write(f"Amplitude: {pk['params'][0]:.3f}")
                    with col2:
                        if st.button(f"🗑️ Remover", key=f"del_{i}"):
                            st.session_state.peaks.pop(i)
                            safe_rerun()
            
            # Clear all peaks
            if st.button("🗑️ Limpar Todos os Picos", use_container_width=True):
                st.session_state.peaks = []
                st.session_state.fit_params = None
                safe_rerun()
        else:
            st.info("Nenhum pico adicionado ainda.")
    
    # ===== TAB: AJUSTE =====
    with tab_fit:
        st.subheader("🎯 Configurações de Ajuste")
        
        # Optimization method
        st.markdown("**Método de Otimização**")
        fit_method = st.selectbox("Algoritmo", 
                                ["curve_fit", "differential_evolution", "minimize"])
        
        # Advanced parameters based on method
        if fit_method == "curve_fit":
            algorithm = st.selectbox("Algoritmo interno", ["trf", "dogbox", "lm"])
            maxfev = st.number_input("Max. avaliações", value=20000, min_value=1000, step=1000)
            fit_kwargs = {"algorithm": algorithm, "maxfev": maxfev}
        
        elif fit_method == "differential_evolution":
            col1, col2 = st.columns(2)
            with col1:
                maxiter = st.number_input("Max. iterações", value=1000, min_value=100, step=100)
                seed = st.number_input("Seed", value=42, min_value=0)
            with col2:
                popsize = st.number_input("Tamanho população", value=15, min_value=5)
            fit_kwargs = {"maxiter": maxiter, "seed": seed, "popsize": popsize}
        
        else:  # minimize
            algorithm = st.selectbox("Algoritmo", 
                                    ["L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead"])
            fit_kwargs = {"algorithm": algorithm}
        
        st.markdown("---")
        
        # Constraints
        st.markdown("**Restrições Globais**")
        constrain_centers = st.checkbox("Restringir centros ao intervalo X", value=True)
        constrain_amplitudes = st.checkbox("Restringir amplitudes positivas", value=True)
        
        # Run optimization
        if st.button("🚀 Executar Ajuste", type="primary", use_container_width=True,
                    disabled=len(st.session_state.peaks) == 0):
            with st.spinner("Otimizando..."):
                try:
                    flat, pcov = dec.fit(st.session_state.x, st.session_state.y, 
                                        st.session_state.peaks, fit_method, **fit_kwargs)
                    
                    # Update peak parameters
                    pos = 0
                    for i, pk in enumerate(st.session_state.peaks):
                        n = len(dec.peak_models[pk["type"]][1])
                        st.session_state.peaks[i]["params"] = list(map(float, flat[pos:pos+n]))
                        pos += n
                    
                    # Save to history
                    st.session_state.fit_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "method": fit_method,
                        "peaks": st.session_state.peaks.copy(),
                        "params": flat.tolist()
                    })
                    
                    st.success("✅ Ajuste concluído com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro no ajuste: {str(e)}")
            safe_rerun()
        
        # Fit history
        if len(st.session_state.fit_history) > 0:
            st.markdown("---")
            st.markdown("**📜 Histórico de Ajustes**")
            for i, fit in enumerate(reversed(st.session_state.fit_history[-5:])):
                if st.button(f"Restaurar: {fit['timestamp']}", key=f"hist_{i}"):
                    st.session_state.peaks = fit['peaks']
                    safe_rerun()
    
    # ===== TAB: VISUALIZAÇÃO =====
    with tab_visual:
        st.subheader("🎨 Customização Visual")
        
        # Plot style
        st.markdown("**Estilo do Gráfico**")
        plot_style = st.selectbox("Tipo de plot", 
                                 ["lines", "markers", "lines+markers"])
        color_scheme = st.selectbox("Esquema de cores",
                                   ["default", "scientific", "dark", "publication"])
        
        # Line and marker settings
        col1, col2 = st.columns(2)
        with col1:
            line_width = st.slider("Espessura linha", 1, 5, 2)
            marker_size = st.slider("Tamanho marcador", 2, 10, 4)
        with col2:
            comp_opacity = st.slider("Opacidade componentes", 0.1, 1.0, 0.35)
        
        # Display options
        st.markdown("**Opções de Exibição**")
        col1, col2 = st.columns(2)
        with col1:
            show_components = st.checkbox("Mostrar componentes", value=True)
            show_residuals = st.checkbox("Mostrar resíduos", value=True)
            show_grid = st.checkbox("Mostrar grade", value=True)
            fill_areas = st.checkbox("Preencher áreas", value=False)
        with col2:
            show_centers = st.checkbox("Linhas de centro", value=True)
            show_legend = st.checkbox("Mostrar legenda", value=True)
        
        # Legend position
        if show_legend:
            legend_position = st.selectbox("Posição da legenda",
                                         ["topright", "topleft", "bottomright", 
                                          "bottomleft", "outside"])
        else:
            legend_position = "topright"
        
        # Labels
        st.markdown("**Rótulos**")
        title = st.text_input("Título", value="Deconvolução Espectral")
        x_label = st.text_input("Rótulo eixo X", value="X")
        y_label = st.text_input("Rótulo eixo Y", value="Intensidade")
        
        # Save visual settings
        st.session_state.visual_settings = {
            "plot_style": plot_style,
            "color_scheme": color_scheme,
            "line_width": line_width,
            "marker_size": marker_size,
            "comp_opacity": comp_opacity,
            "show_components": show_components,
            "show_residuals": show_residuals,
            "show_grid": show_grid,
            "fill_areas": fill_areas,
            "show_centers": show_centers,
            "show_legend": show_legend,
            "legend_position": legend_position,
            "title": title,
            "x_label": x_label,
            "y_label": y_label
        }

# -------------------------------------------
# Main Content Area (Right side)
# -------------------------------------------
# Get visual settings
vs = st.session_state.get("visual_settings", {
    "plot_style": "lines",
    "color_scheme": "default",
    "line_width": 2,
    "marker_size": 4,
    "comp_opacity": 0.35,
    "show_components": True,
    "show_residuals": True,
    "show_grid": True,
    "fill_areas": False,
    "show_centers": True,
    "show_legend": True,
    "legend_position": "topright",
    "title": "Deconvolução Espectral",
    "x_label": "X",
    "y_label": "Intensidade"
})

# Create columns for main area
col_main, col_stats = st.columns([3, 1])

with col_main:
    # Highlight selection
    if len(st.session_state.peaks) > 0:
        highlight_opt = ["Nenhum"] + [f"{i+1}. {p['type']}" for i, p in enumerate(st.session_state.peaks)]
        sel = st.selectbox("🔍 Pico em destaque", options=highlight_opt, index=0)
        highlight_idx = None if sel == "Nenhum" else int(sel.split(".")[0]) - 1
    else:
        highlight_idx = None
    
    # Generate plot
    fig, y_fit_total = plot_figure(
        st.session_state.x, st.session_state.y, st.session_state.peaks, dec,
        show_fit=True, 
        show_components=vs["show_components"],
        show_residuals=vs["show_residuals"],
        y_range=st.session_state.y_range,
        highlight_idx=highlight_idx,
        comp_opacity=vs["comp_opacity"],
        fill_areas=vs["fill_areas"],
        show_centers=vs["show_centers"],
        plot_style=vs["plot_style"],
        color_scheme=vs["color_scheme"],
        line_width=vs["line_width"],
        marker_size=vs["marker_size"],
        show_grid=vs["show_grid"],
        show_legend=vs["show_legend"],
        legend_position=vs["legend_position"],
        title=vs["title"],
        x_label=vs["x_label"],
        y_label=vs["y_label"]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown("### 📊 Estatísticas")
    
    if len(st.session_state.peaks) > 0 and y_fit_total is not None:
        # Calculate statistics
        res = st.session_state.y - y_fit_total
        ss_res = float(np.sum(res**2))
        ss_tot = float(np.sum((st.session_state.y - np.mean(st.session_state.y))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean(res**2)))
        chi2 = float(np.sum(res**2 / np.maximum(y_fit_total, 1e-12)))
        
        # Display metrics
        st.metric("R²", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("χ²", f"{chi2:.2f}")
        st.metric("Nº Picos", len(st.session_state.peaks))
        
        # Additional statistics
        st.markdown("---")
        st.markdown("**Resíduos**")
        st.metric("Média", f"{np.mean(res):.4e}")
        st.metric("Desvio Padrão", f"{np.std(res):.4f}")
        st.metric("Min/Max", f"{np.min(res):.3f} / {np.max(res):.3f}")
    else:
        st.info("Execute o ajuste para ver as estatísticas")

# -------------------------------------------
# Results and Export Section
# -------------------------------------------
st.markdown("---")
tab_results, tab_export, tab_advanced = st.tabs(
    ["📊 Resultados", "💾 Exportação", "⚡ Avançado"]
)

with tab_results:
    if len(st.session_state.peaks) == 0:
        st.info("Nenhum pico para exibir. Adicione picos e execute o ajuste.")
    else:
        # Calculate results
        rows = []
        x = st.session_state.x
        y = st.session_state.y
        y_total = np.zeros_like(x, dtype=float)
        
        for pk in st.session_state.peaks:
            y_total += dec._eval_single(x, pk["type"], pk["params"])
        
        total_area = float(np.trapz(y_total, x)) if np.any(y_total) else 1.0
        
        for i, pk in enumerate(st.session_state.peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            area = area_under_peak(x, y_comp, pk["type"], pk["params"])
            perc = 100.0 * area / total_area if total_area > 0 else np.nan
            fwhm = fwhm_of_peak(pk["type"], pk["params"])
            
            rows.append({
                "Pico": i,
                "Tipo": pk["type"],
                "Amplitude": f"{pk['params'][0]:.4f}",
                "Centro": f"{pk['params'][1]:.4f}",
                "FWHM": f"{fwhm:.4f}" if fwhm else "N/A",
                "Área": f"{area:.4f}",
                "Área (%)": f"{perc:.2f}",
            })
        
        res_df = pd.DataFrame(rows)
        
        # Display results table
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Picos", len(rows))
        with col2:
            st.metric("Área Total", f"{total_area:.4f}")
        with col3:
            if len(rows) > 0:
                centers = [pk["params"][1] for pk in st.session_state.peaks]
                st.metric("Centro Médio", f"{np.mean(centers):.4f}")

with tab_export:
    if len(st.session_state.peaks) == 0:
        st.info("Nenhum dado para exportar. Execute o ajuste primeiro.")
    else:
        st.markdown("### 📥 Opções de Exportação")
        
        # Prepare data for export
        x = st.session_state.x
        y = st.session_state.y
        
        # Recalculate results
        rows = []
        y_total = np.zeros_like(x, dtype=float)
        comp_arrays = []
        comp_names = []
        
        for pk in st.session_state.peaks:
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            y_total += y_comp
            comp_arrays.append(y_comp)
            comp_names.append(f"comp_{pk['type'].replace(' ', '_')}")
        
        total_area = float(np.trapz(y_total, x)) if np.any(y_total) else 1.0
        
        for i, pk in enumerate(st.session_state.peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            area = area_under_peak(x, y_comp, pk["type"], pk["params"])
            perc = 100.0 * area / total_area if total_area > 0 else np.nan
            fwhm = fwhm_of_peak(pk["type"], pk["params"])
            
            rows.append({
                "Pico": i,
                "Tipo": pk["type"],
                "Amplitude": pk["params"][0],
                "Centro": pk["params"][1],
                "FWHM": fwhm,
                "Área": area,
                "Área (%)": perc,
                "Parâmetros": str(pk["params"]),
            })
        
        res_df = pd.DataFrame(rows)
        residual = y - y_total
        
        # Export formats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV - Results
            res_csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Resultados (CSV)",
                data=res_csv,
                file_name=f"deconv_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON - Parameters
            payload = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "n_peaks": len(st.session_state.peaks),
                    "r_squared": float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)) if np.sum((y - np.mean(y))**2) > 0 else None,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "preprocessing": st.session_state.preprocessing
                },
                "peaks": [
                    {
                        "type": pk["type"],
                        "params": list(map(float, pk["params"])),
                        "bounds": pk["bounds"]
                    }
                    for pk in st.session_state.peaks
                ]
            }
            json_bytes = json.dumps(payload, indent=2).encode("utf-8")
            st.download_button(
                "🔧 Parâmetros (JSON)",
                data=json_bytes,
                file_name=f"deconv_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # HTML - Interactive plot
            html_buf = io.StringIO()
            fig.write_html(html_buf, include_plotlyjs="cdn", full_html=True)
            html_bytes = html_buf.getvalue().encode("utf-8")
            st.download_button(
                "📊 Gráfico (HTML)",
                data=html_bytes,
                file_name=f"deconv_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        
        with col4:
            # Excel - Complete data
            xlsx_buf = io.BytesIO()
            writer_obj = get_excel_writer(xlsx_buf)
            
            if writer_obj:
                with writer_obj as writer:
                    # Sheet 1: Fitted curves
                    curves_df = pd.DataFrame({
                        "x": x,
                        "y_data": y,
                        "y_fit": y_total,
                        "residual": residual
                    })
                    for name, arr in zip(comp_names, comp_arrays):
                        curves_df[name] = arr
                    curves_df.to_excel(writer, index=False, sheet_name="Curvas")
                    
                    # Sheet 2: Results
                    res_df.to_excel(writer, index=False, sheet_name="Resultados")
                    
                    # Sheet 3: Parameters
                    params_df = pd.DataFrame([
                        {
                            "Pico": i,
                            "Tipo": pk["type"],
                            **{f"Param_{j}": v for j, v in enumerate(pk["params"])}
                        }
                        for i, pk in enumerate(st.session_state.peaks, start=1)
                    ])
                    params_df.to_excel(writer, index=False, sheet_name="Parâmetros")
                    
                    # Sheet 4: Raw data
                    raw_df = pd.DataFrame({
                        "x_original": st.session_state.x_original,
                        "y_original": st.session_state.y_original,
                        "x_processed": x,
                        "y_processed": y
                    })
                    raw_df.to_excel(writer, index=False, sheet_name="Dados")
                
                st.download_button(
                    "📑 Completo (Excel)",
                    data=xlsx_buf.getvalue(),
                    file_name=f"deconv_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Excel writer não disponível")
        
        # Advanced export options
        st.markdown("---")
        st.markdown("### 🔬 Exportação Avançada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export high-resolution figure
            if st.button("🖼️ Gerar Figura Alta Resolução"):
                # Configure for publication
                fig_pub = plot_figure(
                    x, y, st.session_state.peaks, dec,
                    show_fit=True,
                    show_components=True,
                    show_residuals=False,
                    y_range=st.session_state.y_range,
                    plot_style="lines",
                    color_scheme="publication",
                    line_width=2,
                    show_grid=True,
                    show_legend=True,
                    title="",
                    x_label=vs["x_label"],
                    y_label=vs["y_label"]
                )
                
                # Export as static image bytes
                img_bytes = pio.to_image(fig_pub, format="png", width=1200, height=800, scale=2)
                
                st.download_button(
                    "Download PNG (2400x1600)",
                    data=img_bytes,
                    file_name=f"deconv_highres_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        with col2:
            # LaTeX table export
            if st.button("📝 Gerar Tabela LaTeX"):
                latex_table = res_df.to_latex(index=False, float_format
