# -*- coding: utf-8 -*-
import io
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
from scipy.special import wofz  # for exact Voigt
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

# Excel writer helper: prefer XlsxWriter, fallback to openpyxl
def get_excel_writer(buffer):
    try:
        import xlsxwriter  # noqa: F401
        return pd.ExcelWriter(buffer, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return pd.ExcelWriter(buffer, engine="openpyxl")
        except Exception:
            return None


# -------------------------------------------
# Page Config
# -------------------------------------------
st.set_page_config(
    page_title="DeconvoluÃ§Ã£o Espectral AvanÃ§ada",
    page_icon="ðŸ"Š",
    layout="wide"
)

# -------------------------------------------
# Helpers (compatibility & math)
# -------------------------------------------
def safe_rerun():
    """Streamlit compatibility: prefer st.rerun (new) and fallback to experimental_rerun (old)."""
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
    # width ~ FWHM; eta in [0,1]
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
        amp, _, sigma = params
        return float(amp * sigma * np.sqrt(2 * np.pi))
    if model_type == "Lorentziana":
        amp, _, gamma = params
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
    if model_type == "Gaussiana AssimÃ©trica":
        sigma_l, sigma_r = params[2], params[3]
        sigma_eq = 0.5 * (sigma_l + sigma_r)
        return float(2 * np.sqrt(2 * np.log(2)) * sigma_eq)
    if model_type == "Pearson VII":
        width, m = params[2], params[3]
        return float(2 * width * np.sqrt(2 ** (1.0 / m) - 1.0))
    return None

# -------------------------------------------
# Enhanced Deconvolution Engine with Export Options
# -------------------------------------------
class SpectralDeconvolution:
    def __init__(self):
        self.peak_models = {
            "Gaussiana": ("gaussian", ["Amplitude", "Centro", "Sigma"]),
            "Lorentziana": ("lorentzian", ["Amplitude", "Centro", "Gamma"]),
            "Voigt (exato)": ("voigt_exact", ["Amplitude", "Centro", "Sigma (G)", "Gamma (L)"]),
            "Pseudo-Voigt": ("pseudo_voigt", ["Amplitude", "Centro", "Largura (FWHM~)", "FraÃ§Ã£o Lorentz (Î·)"]),
            "Gaussiana AssimÃ©trica": ("asymmetric_gaussian", ["Amplitude", "Centro", "Sigma Esq", "Sigma Dir"]),
            "Pearson VII": ("pearson_vii", ["Amplitude", "Centro", "Largura", "Forma (m)"]),
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
        if peak_type == "Gaussiana AssimÃ©trica":
            return asymmetric_gaussian(x, *params)
        if peak_type == "Pearson VII":
            return pearson_vii(x, *params)
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

    def fit(self, x: np.ndarray, y: np.ndarray, peak_list: List[Dict[str, Any]], method: str):
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
                popt, pcov = curve_fit(comp, x, y, p0=p0, bounds=bounds, maxfev=20000)
                return popt, pcov
            else:
                def objective(vec):
                    return np.sum((y - comp(x, *vec)) ** 2)
                de_bounds = list(zip(bounds[0], bounds[1]))
                res = differential_evolution(objective, de_bounds, seed=42, maxiter=500)
                return res.x, None
        except Exception as exc:
            st.error(f"Erro no ajuste: {exc}")
            return np.array(p0, dtype=float), None

def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba with specified alpha."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color

def generate_peak_colors(n_peaks, base_colors=None):
    """Generate a list of colors for peaks."""
    if base_colors is None:
        base_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D7BDE2"
        ]
    colors = []
    for i in range(n_peaks):
        colors.append(base_colors[i % len(base_colors)])
    return colors

def plot_figure_export(x, y, peaks, dec, export_config=None, show_fit=True, show_components=True, 
                      show_residuals=True, y_range=None, highlight_idx=None):
    """Enhanced plot function with export customization options."""
    
    # Default export configuration
    default_config = {
        'data_color': '#4DA3FF',
        'fit_color': '#FF6EC7',
        'residual_color': '#FF4D4D',
        'component_colors': None,  # Will be auto-generated
        'fill_components': False,
        'fill_opacity': 0.3,
        'component_opacity': 0.7,
        'line_width_data': 2,
        'line_width_fit': 3,
        'line_width_components': 2,
        'show_centers': True,
        'center_line_style': 'dash',
        'center_line_color': '#666666',
        'center_line_width': 1,
        'background_color': 'white',
        'plot_title': 'Deconvolução Espectral',
        'x_title': 'X',
        'y_title': 'Intensidade',
        'font_family': 'Arial',
        'font_size': 12,
        'title_font_size': 16,
        'show_legend': True,
        'legend_position': 'right'
    }
    
    # Merge with user configuration
    if export_config:
        config = {**default_config, **export_config}
    else:
        config = default_config
    
    # Generate component colors if not provided
    if config['component_colors'] is None:
        config['component_colors'] = generate_peak_colors(len(peaks))
    
    fig = go.Figure()
    
    # Add data trace
    fig.add_trace(go.Scatter(
        x=x, y=y, 
        mode="lines", 
        name="Dados",
        line=dict(width=config['line_width_data'], color=config['data_color'])
    ))
    
    y_fit_total = None
    shapes = []

    if show_fit and len(peaks) > 0:
        y_fit_total = np.zeros_like(x, dtype=float)
        
        # Add component traces
        for i, pk in enumerate(peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            y_fit_total += y_comp

            if show_components:
                is_highlighted = (highlight_idx is not None and (i-1) == highlight_idx)
                component_color = config['component_colors'][(i-1) % len(config['component_colors'])]
                
                line_style = dict(
                    width=config['line_width_components'] + (1 if is_highlighted else 0),
                    color=component_color
                )
                
                name = f"{pk['type']} #{i}" + (" (★)" if is_highlighted else "")
                
                if config['fill_components']:
                    fill_color = hex_to_rgba(component_color, config['fill_opacity'])
                    fig.add_trace(go.Scatter(
                        x=x, y=y_comp, 
                        mode="lines", 
                        name=name,
                        line=line_style, 
                        opacity=config['component_opacity'],
                        fill="tozeroy", 
                        fillcolor=fill_color
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x, y=y_comp, 
                        mode="lines", 
                        name=name,
                        line=line_style, 
                        opacity=config['component_opacity']
                    ))

                # Add center lines
                if config['show_centers']:
                    cx = float(pk["params"][1])
                    y0 = float(y.min() if y_range is None else y_range[0])
                    y1 = float(y.max() if y_range is None else y_range[1])
                    shapes.append(dict(
                        type="line", 
                        x0=cx, x1=cx, y0=y0, y1=y1,
                        line=dict(
                            color=config['center_line_color'], 
                            width=config['center_line_width'], 
                            dash=config['center_line_style']
                        )
                    ))

        # Add fit trace
        fig.add_trace(go.Scatter(
            x=x, y=y_fit_total, 
            mode="lines", 
            name="Ajuste Total",
            line=dict(width=config['line_width_fit'], color=config['fit_color'])
        ))

        # Add residuals
        if show_residuals:
            res = y - y_fit_total
            fig.add_trace(go.Scatter(
                x=x, y=res, 
                mode="lines", 
                name="Resíduos",
                line=dict(width=1, color=config['residual_color']), 
                yaxis="y2"
            ))

    # Configure layout
    layout = dict(
        title=dict(
            text=config['plot_title'],
            font=dict(family=config['font_family'], size=config['title_font_size'])
        ),
        xaxis_title=config['x_title'],
        yaxis_title=config['y_title'],
        font=dict(family=config['font_family'], size=config['font_size']),
        plot_bgcolor=config['background_color'],
        paper_bgcolor=config['background_color'],
        height=650,
        hovermode="x unified",
        shapes=shapes,
        showlegend=config['show_legend']
    )
    
    # Legend position
    if config['show_legend']:
        if config['legend_position'] == 'right':
            layout['legend'] = dict(orientation="v", y=1, x=1.02, yanchor="top")
        elif config['legend_position'] == 'top':
            layout['legend'] = dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        elif config['legend_position'] == 'bottom':
            layout['legend'] = dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    
    # Residuals axis
    if show_residuals:
        layout.update(
            yaxis2=dict(
                overlaying="y", 
                side="right", 
                title="Resíduos", 
                showgrid=False, 
                zeroline=True
            )
        )
    
    # Y-axis range
    if y_range is not None:
        layout["yaxis"] = dict(title=config['y_title'], range=y_range)

    fig.update_layout(**layout)
    return fig, y_fit_total

def plot_figure(x, y, peaks, dec, show_fit=True, show_components=True, show_residuals=True, y_range=None,
                 highlight_idx=None, comp_opacity=0.35, fill_areas=False, show_centers=True):
    """Original plot function for compatibility."""
    COLOR_DATA = "#4DA3FF"
    COLOR_SUM  = "#FF6EC7"
    COLOR_RES  = "#FF4D4D"
    COLOR_HILITE = "#FFD166"
    COLOR_COMP = "#A0AEC0"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Dados", line=dict(width=2, color=COLOR_DATA)))
    y_fit_total = None
    shapes = []

    if show_fit and len(peaks) > 0:
        y_fit_total = np.zeros_like(x, dtype=float)
        for i, pk in enumerate(peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            y_fit_total += y_comp

            if show_components:
                is_h = (highlight_idx is not None and (i-1) == highlight_idx)
                line_style = dict(width=3 if is_h else 1, dash="solid" if is_h else "dot",
                                  color=COLOR_HILITE if is_h else COLOR_COMP)
                name = f"{pk['type']} #{i}" + (" (☆)" if is_h else "")
                if fill_areas and not is_h:
                    fig.add_trace(go.Scatter(x=x, y=y_comp, mode="lines", name=name,
                                             line=line_style, opacity=comp_opacity,
                                             fill="tozeroy", fillcolor="rgba(160,174,192,0.15)"))
                else:
                    fig.add_trace(go.Scatter(x=x, y=y_comp, mode="lines", name=name,
                                             line=line_style, opacity=1.0 if is_h else comp_opacity))

                if show_centers:
                    cx = float(pk["params"][1])
                    y0 = float(y.min() if y_range is None else y_range[0])
                    y1 = float(y.max() if y_range is None else y_range[1])
                    shapes.append(dict(type="line", x0=cx, x1=cx, y0=y0, y1=y1,
                                       line=dict(color=COLOR_HILITE if is_h else "#666", width=1, dash="dash")))

        fig.add_trace(go.Scatter(x=x, y=y_fit_total, mode="lines", name="Soma Ajuste",
                                 line=dict(width=3, color=COLOR_SUM)))

        if show_residuals:
            res = y - y_fit_total
            fig.add_trace(go.Scatter(x=x, y=res, mode="lines", name="Resíduos",
                                     line=dict(width=1, color=COLOR_RES), yaxis="y2"))

    layout = dict(
        title="Deconvolução Espectral",
        xaxis_title="X",
        yaxis_title="Intensidade",
        height=650,
        hovermode="x unified",
        legend=dict(orientation="v", y=1, x=1.02, yanchor="top"),
        shapes=shapes
    )
    if show_residuals:
        layout.update(
            yaxis2=dict(overlaying="y", side="right", title="Resíduos", showgrid=False, zeroline=True)
        )
    if y_range is not None:
        layout["yaxis"] = dict(title="Intensidade", range=y_range)

    fig.update_layout(**layout)
    return fig, y_fit_total

# -------------------------------------------
# Session init
# -------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "x" not in st.session_state:
    st.session_state.x = None
if "y" not in st.session_state:
    st.session_state.y = None
if "peaks" not in st.session_state:
    st.session_state.peaks = []
if "fit_params" not in st.session_state:
    st.session_state.fit_params = None
if "y_range" not in st.session_state:
    st.session_state.y_range = None

dec = SpectralDeconvolution()

# -------------------------------------------
# Sidebar – Data & Settings
# -------------------------------------------

with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("📁 Carregar Dados")
    up = st.file_uploader("CSV/TXT/Excel", type=["csv", "txt", "xlsx", "xls"])

    sheet = None
    header_row = 0
    decimal_csv = st.selectbox("Separador decimal (CSV/TXT)", options=[",", "."], index=1, help="Apenas para CSV/TXT; Excel detecta sozinho.")
    transpose = st.checkbox("Dados transpostos (linhas=variáveis)", value=False)
    force_numeric = st.checkbox("Forçar numérico (coagir strings)", value=True)
    sort_x = st.checkbox("Ordenar por X", value=True)

    if up is not None and (up.name.lower().endswith(".xlsx") or up.name.lower().endswith(".xls")):
        names = excel_sheet_names(up)
        if names:
            sheet = st.selectbox("Planilha", names, index=0)
        header_row = st.number_input("Linha do cabeçalho (1 = primeira)", min_value=1, value=1, step=1) - 1

    # Leitura
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up, decimal=decimal_csv, sep=None, engine="python")
            elif up.name.lower().endswith(".txt"):
                try:
                    df = pd.read_csv(up, sep="\t", decimal=decimal_csv)
                except Exception:
                    up.seek(0)
                    df = pd.read_csv(up, sep=r"\s+", engine="python", decimal=decimal_csv)
            else:
                df = pd.read_excel(up, sheet_name=sheet if sheet is not None else 0, header=header_row)
            if transpose:
                df = df.T
                df.reset_index(drop=False, inplace=True)
            if force_numeric:
                df = coerce_numeric_df(df)
            st.session_state.df = df
        except Exception as exc:
            st.error(f"Falha ao ler arquivo: {exc}")

    if st.session_state.df is None:
        st.info("Sem arquivo? Carregando dados de exemplo.")
        st.session_state.df = synthetic_example()

    df = st.session_state.df

    # Sugerir colunas numéricas
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        df_num = coerce_numeric_df(df)
        numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
        if len(numeric_cols) >= 2:
            df = df_num
            st.session_state.df = df

    st.write("Prévia dos dados (topo):")
    st.dataframe(df.head(15), use_container_width=True)

    if len(numeric_cols) < 2:
        st.error("Não encontrei ao menos duas colunas numéricas. Ajuste as opções acima (planilha, transpose, forçar numérico).")
        numeric_cols = list(df.columns)

    colx = st.selectbox("Coluna X", numeric_cols, index=0 if len(numeric_cols)>0 else 0)
    coly = st.selectbox("Coluna Y", numeric_cols, index=1 if len(numeric_cols)>1 else 0)

    x = df[colx].to_numpy(dtype=float)
    y = df[coly].to_numpy(dtype=float)

    # Limpeza básica
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if sort_x and x.size > 1:
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

    st.session_state.x, st.session_state.y = x, y

    st.divider()
    st.subheader("📊 Eixo Y")
    y_min = st.number_input("Y min", value=float(np.nanmin(y)) if y.size else 0.0, step=0.1, format="%.4f")
    y_max = st.number_input("Y max", value=float(np.nanmax(y)) if y.size else 1.0, step=0.1, format="%.4f")
    if st.button("Adicionar", use_container_width=True):
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
        else:  # Pearson VII
            params = [y_max/3, (x_min+x_max)/2, xr/20, 1.5]
            bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.5, 10.0)]
        st.session_state.peaks.append({"type": pk_type, "params": params, "bounds": bounds})
        safe_rerun()

    st.divider()
    
    st.subheader("🎚️ Modo Live: posição, largura & amplitude")
    if len(st.session_state.peaks) == 0:
        st.info("Nenhum pico. Use 'Detectar picos' ou 'Adicionar'.")
    else:
        for i, pk in enumerate(st.session_state.peaks, start=1):
            with st.expander(f"Pico {i}: {pk['type']}", expanded=False):
                # Amplitude (nº + slider)
                a_lo, a_hi = pk["bounds"][0]
                a_num = st.number_input(f"Amplitude (pico {i})", min_value=float(a_lo), max_value=float(a_hi),
                                        value=float(pk["params"][0]), key=f"amp_num_{i}")
                a_sl = st.slider(f"Amplitude (pico {i}) [slider]", float(a_lo), float(a_hi), float(a_num), key=f"amp_sl_{i}")
                pk["params"][0] = float(a_sl)

                # Centro (nº + slider)
                c_lo, c_hi = pk["bounds"][1]
                c_num = st.number_input(f"Centro (pico {i})", min_value=float(c_lo), max_value=float(c_hi),
                                        value=float(pk["params"][1]), key=f"center_num_{i}")
                c_sl = st.slider(f"Centro (pico {i}) [slider]", float(c_lo), float(c_hi), float(c_num), key=f"center_sl_{i}")
                pk["params"][1] = float(c_sl)

                # Larguras por tipo (nº + slider)
                t = pk["type"]
                xr = float(np.max(st.session_state.x) - np.min(st.session_state.x))
                if t == "Gaussiana":
                    lo, hi = pk["bounds"][2]
                    s_num = st.number_input(f"Sigma (pico {i})", min_value=float(lo), max_value=float(hi),
                                            value=float(pk['params'][2]), key=f"sigma_num_{i}")
                    s_sl = st.slider(f"Sigma (pico {i}) [slider]", float(lo), float(hi), float(s_num), key=f"sigma_sl_{i}")
                    pk['params'][2] = float(s_sl)
                    st.caption(f"FWHM ≈ {2*np.sqrt(2*np.log(2))*pk['params'][2]:.3f}")
                elif t == "Lorentziana":
                    lo, hi = pk["bounds"][2]
                    g_num = st.number_input(f"Gamma (pico {i})", min_value=float(lo), max_value=float(hi),
                                            value=float(pk['params'][2]), key=f"gamma_num_{i}")
                    g_sl = st.slider(f"Gamma (pico {i}) [slider]", float(lo), float(hi), float(g_num), key=f"gamma_sl_{i}")
                    pk['params'][2] = float(g_sl)
                    st.caption(f"FWHM ≈ {2*pk['params'][2]:.3f}")
                elif t == "Pseudo-Voigt":
                    lo, hi = pk["bounds"][2]
                    w_num = st.number_input(f"Largura FWHM~ (pico {i})", min_value=float(lo), max_value=float(hi),
                                            value=float(pk['params'][2]), key=f"width_num_{i}")
                    w_sl = st.slider(f"Largura FWHM~ (pico {i}) [slider]", float(lo), float(hi), float(w_num), key=f"width_sl_{i}")
                    pk['params'][2] = float(w_sl)
                    e_lo, e_hi = pk["bounds"][3]
                    e_num = st.number_input(f"η Lorentziano (pico {i})", min_value=float(e_lo), max_value=float(e_hi),
                                            value=float(pk['params'][3]), key=f"eta_num_{i}")
                    e_sl = st.slider(f"η Lorentziano (pico {i}) [slider]", float(e_lo), float(e_hi), float(e_num), key=f"eta_sl_{i}")
                    pk['params'][3] = float(e_sl)
                elif t == "Voigt (exato)":
                    lo_s, hi_s = pk["bounds"][2]
                    s_num = st.number_input(f"Sigma G (pico {i})", min_value=float(lo_s), max_value=float(hi_s),
                                            value=float(pk['params'][2]), key=f"vsigma_num_{i}")
                    s_sl = st.slider(f"Sigma G (pico {i}) [slider]", float(lo_s), float(hi_s), float(s_num), key=f"vsigma_sl_{i}")
                    pk['params'][2] = float(s_sl)

                    lo_g, hi_g = pk["bounds"][3]
                    g_num = st.number_input(f"Gamma L (pico {i})", min_value=float(lo_g), max_value=float(hi_g),
                                            value=float(pk['params'][3]), key=f"vgamma_num_{i}")
                    g_sl = st.slider(f"Gamma L (pico {i}) [slider]", float(lo_g), float(hi_g), float(g_num), key=f"vgamma_sl_{i}")
                    pk['params'][3] = float(g_sl)
                    from math import sqrt, log
                    st.caption(f"FWHM (aprox) ≈ {0.5346*2*pk['params'][3] + np.sqrt(0.2166*(2*pk['params'][3])**2 + (2.355*pk['params'][2])**2):.3f}")
                elif t == "Gaussiana Assimétrica":
                    lo_l, hi_l = pk["bounds"][2]
                    sl_num = st.number_input(f"Sigma Esq (pico {i})", min_value=float(lo_l), max_value=float(hi_l),
                                             value=float(pk['params'][2]), key=f"sigl_num_{i}")
                    sl_sl = st.slider(f"Sigma Esq (pico {i}) [slider]", float(lo_l), float(hi_l), float(sl_num), key=f"sigl_sl_{i}")
                    pk['params'][2] = float(sl_sl)

                    lo_r, hi_r = pk["bounds"][3]
                    sr_num = st.number_input(f"Sigma Dir (pico {i})", min_value=float(lo_r), max_value=float(hi_r),
                                             value=float(pk['params'][3]), key=f"sigr_num_{i}")
                    sr_sl = st.slider(f"Sigma Dir (pico {i}) [slider]", float(lo_r), float(hi_r), float(sr_num), key=f"sigr_sl_{i}")
                    pk['params'][3] = float(sr_sl)
                    st.caption(f"FWHM (aprox) ≈ {2*np.sqrt(2*np.log(2))*(0.5*(pk['params'][2]+pk['params'][3])):.3f}")
                elif t == "Pearson VII":
                    lo_w, hi_w = pk["bounds"][2]
                    w_num = st.number_input(f"Largura (pico {i})", min_value=float(lo_w), max_value=float(hi_w),
                                            value=float(pk['params'][2]), key=f"pvw_num_{i}")
                    w_sl = st.slider(f"Largura (pico {i}) [slider]", float(lo_w), float(hi_w), float(w_num), key=f"pvw_sl_{i}")
                    pk['params'][2] = float(w_sl)

                    lo_m, hi_m = pk["bounds"][3]
                    m_num = st.number_input(f"Forma m (pico {i})", min_value=float(lo_m), max_value=float(hi_m),
                                            value=float(pk['params'][3]), key=f"pvm_num_{i}")
                    m_sl = st.slider(f"Forma m (pico {i}) [slider]", float(lo_m), float(hi_m), float(m_num), key=f"pvm_sl_{i}")
                    pk['params'][3] = float(m_sl)
                    st.caption(f"FWHM (aprox) ≈ {2*pk['params'][2]*np.sqrt(2**(1.0/pk['params'][3]) - 1.0):.3f}")
    st.divider()
    
    
    st.subheader("🗑️ Excluir picos")
    if len(st.session_state.peaks) > 0:
        labels = [f"{i+1}. {p['type']} — centro: {p['params'][1]:.3f}" for i, p in enumerate(st.session_state.peaks)]
        to_delete = st.multiselect(
            "Selecione os picos para remover",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="multi_del_peaks"
        )
        if st.button("Excluir selecionados", use_container_width=True, disabled=(len(to_delete) == 0)):
            for idx in sorted(to_delete, reverse=True):
                st.session_state.peaks.pop(idx)
            safe_rerun()
    else:
        st.info("Nenhum pico para excluir.")
    st.divider()
    st.subheader("🎛️ Ajuste Manual (tipo & gestão)")
    if len(st.session_state.peaks) == 0:
        st.info("Nenhum pico. Use 'Detectar picos' ou 'Adicionar'.")
    else:
        names = [f"{i+1}. {p['type']}" for i, p in enumerate(st.session_state.peaks)]
        idx = st.selectbox("Selecione o pico", list(range(len(names))), format_func=lambda i: names[i])
        pk = st.session_state.peaks[idx]

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Remover pico", use_container_width=True):
                st.session_state.peaks.pop(idx)
                safe_rerun()
        with c2:
            new_type = st.selectbox("Alterar tipo", list(dec.peak_models.keys()),
                                    index=list(dec.peak_models.keys()).index(pk["type"]))
            if new_type != pk["type"]:
                amp0 = pk["params"][0]
                cen0 = pk["params"][1]
                x_min, x_max = float(np.min(st.session_state.x)), float(np.max(st.session_state.x))
                xr = x_max - x_min
                y_max = float(np.max(st.session_state.y))
                if new_type == "Gaussiana":
                    params = [amp0, cen0, xr/20]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr)]
                elif new_type == "Lorentziana":
                    params = [amp0, cen0, xr/20]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr)]
                elif new_type == "Voigt (exato)":
                    params = [amp0, cen0, xr/30, xr/30]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (1e-6, xr)]
                elif new_type == "Pseudo-Voigt":
                    params = [amp0, cen0, xr/15, 0.5]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.0, 1.0)]
                elif new_type == "Gaussiana Assimétrica":
                    params = [amp0, cen0, xr/30, xr/30]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (1e-6, xr)]
                else:  # Pearson VII
                    params = [amp0, cen0, xr/20, 1.5]
                    bounds = [(0, y_max*2), (x_min, x_max), (1e-6, xr), (0.5, 10.0)]
                pk.update({"type": new_type, "params": params, "bounds": bounds})
                safe_rerun()
    st.divider()
    if st.button("🚀 Otimizar Ajuste", type="primary", use_container_width=True, disabled=len(st.session_state.peaks) == 0):
        with st.spinner("Otimizando..."):
            flat, pcov = dec.fit(st.session_state.x, st.session_state.y, st.session_state.peaks, fit_method)
            pos = 0
            for i, pk in enumerate(st.session_state.peaks):
                n = len(dec.peak_models[pk["type"]][1])
                st.session_state.peaks[i]["params"] = list(map(float, flat[pos:pos+n]))
                pos += n
        st.success("Ajuste concluído.")
        safe_rerun()

    if st.button("🔄 Resetar picos", use_container_width=True):
        st.session_state.peaks = []
        st.session_state.fit_params = None
        safe_rerun()

with col_plot:
    st.subheader("🖼️ Exibição")
    highlight_opt = ["Nenhum"] + [f"{i+1}. {p['type']}" for i, p in enumerate(st.session_state.peaks)]
    sel = st.selectbox("Pico em destaque", options=highlight_opt, index=0)
    highlight_idx = None if sel == "Nenhum" else int(sel.split(".")[0]) - 1

    comp_opacity = st.slider("Opacidade dos componentes", 0.1, 1.0, 0.35)
    colA, colB, colC = st.columns(3)
    with colA:
        fill_areas = st.checkbox("Preencher área", value=False)
    with colB:
        show_centers = st.checkbox("Linhas de centro", value=True)
    with colC:
        show_resid = st.checkbox("Mostrar resíduos", value=True)

    fig, y_fit_total = plot_figure(
        st.session_state.x, st.session_state.y, st.session_state.peaks, dec,
        show_fit=True, show_components=True, show_residuals=show_resid, y_range=st.session_state.y_range,
        highlight_idx=highlight_idx, comp_opacity=comp_opacity, fill_areas=fill_areas, show_centers=show_centers
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(st.session_state.peaks) > 0:
        if y_fit_total is None:
            y_fit_total = np.zeros_like(st.session_state.x)
            for pk in st.session_state.peaks:
                y_fit_total += dec._eval_single(st.session_state.x, pk["type"], pk["params"])
        res = st.session_state.y - y_fit_total
        ss_res = float(np.sum(res**2))
        ss_tot = float(np.sum((st.session_state.y - np.mean(st.session_state.y))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean(res**2)))
        chi2 = float(np.sum(res**2 / np.maximum(y_fit_total, 1e-12)))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{r2:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("χ²", f"{chi2:.2f}")
        c4.metric("Nº de picos", len(st.session_state.peaks))

# -------------------------------------------
# Results & Export
# -------------------------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(["📝 Resultados dos picos", "💾 Exportar", "🎨 Exportação Personalizada"])

with tab1:
    if len(st.session_state.peaks) == 0:
        st.info("Sem picos para listar.")
    else:
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
                "Amplitude": pk["params"][0],
                "Centro": pk["params"][1],
                "FWHM (aprox)": fwhm,
                "Área": area,
                "Área (%)": perc,
                "Parâmetros": str(pk["params"]),
            })
        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, use_container_width=True)


with tab2:
    if len(st.session_state.peaks) == 0:
        st.info("Nada para exportar.")
    else:
        # --- Recompute results table (res_df) to avoid cross-tab scope issues ---
        x = st.session_state.x
        y = st.session_state.y
        rows = []
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
                "Amplitude": pk["params"][0],
                "Centro": pk["params"][1],
                "FWHM (aprox)": fwhm,
                "Área": area,
                "Área (%)": perc,
                "Parâmetros": str(pk["params"]),
            })
        res_df = pd.DataFrame(rows)

        # --- Buttons with explicit keys to avoid UI confusion on reruns ---
        res_csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar resultados (CSV)", data=res_csv, file_name="deconv_resultados.csv", mime="text/csv", key="dl_results_csv")

        payload = {
            "peaks": [
                {"type": pk["type"], "params": list(map(float, pk["params"])), "bounds": pk["bounds"]}
                for pk in st.session_state.peaks
            ]
        }
        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button("⬇️ Baixar parâmetros (JSON)", data=json_bytes, file_name="deconv_parametros.json", mime="application/json", key="dl_params_json")

        # --- HTML interactive figure ---
        fig, _ = plot_figure(st.session_state.x, st.session_state.y, st.session_state.peaks, dec,
                             show_fit=True, show_components=True, show_residuals=True, y_range=st.session_state.y_range)
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn", full_html=True)
        html_bytes = html_buf.getvalue().encode("utf-8")
        st.download_button("⬇️ Baixar gráfico (HTML interativo)", data=html_bytes, file_name="deconv_grafico.html", mime="text/html", key="dl_fig_html")

        # --- Curves export (CSV) ---
        comp_names, comp_arrays = [], []
        for i, pk in enumerate(st.session_state.peaks, start=1):
            y_comp = dec._eval_single(x, pk["type"], pk["params"])
            comp_names.append(f"comp{i}_{pk['type'].split()[0]}")
            comp_arrays.append(y_comp)
        if comp_arrays:
            y_fit_total = np.sum(np.vstack(comp_arrays), axis=0)
        else:
            y_fit_total = np.zeros_like(x)
        residual = y - y_fit_total
        curves_df = pd.DataFrame({"x": x, "y_data": y, "y_fit_total": y_fit_total, "residual": residual})
        for name, arr in zip(comp_names, comp_arrays):
            curves_df[name] = arr
        cols = ["x", "y_data", "y_fit_total", "residual"] + comp_names
        curves_df = curves_df[cols]
        curves_csv = curves_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar curvas ajustadas (CSV)", data=curves_csv, file_name="deconv_curvas.csv", mime="text/csv", key="dl_curves_csv")

        
        # --- Curves export (XLSX) ---
        import io
        xlsx_buf = io.BytesIO()
        writer_obj = get_excel_writer(xlsx_buf)
        if writer_obj is None:
            st.error("Para exportar XLSX, instale **XlsxWriter** (recomendado) ou **openpyxl**. Ex.: `pip install XlsxWriter openpyxl`.")
        else:
            with writer_obj as writer:
                # 1) Curvas
                curves_df.to_excel(writer, index=False, sheet_name="curvas")
                # 2) Resultados dos picos
                try:
                    res_df.to_excel(writer, index=False, sheet_name="resultados_picos")
                except Exception:
                    pass
                # 3) Parâmetros atuais
                try:
                    params_rows = []
                    for i, pk in enumerate(st.session_state.peaks, start=1):
                        params_rows.append({"Pico": i, "Tipo": pk["type"], "Parâmetros": str([float(v) for v in pk["params"]])})
                    if params_rows:
                        pd.DataFrame(params_rows).to_excel(writer, index=False, sheet_name="parametros")
                except Exception:
                    pass
                # 4) Dados brutos
                try:
                    pd.DataFrame({"x": st.session_state.x, "y": st.session_state.y}).to_excel(writer, index=False, sheet_name="dados_brutos")
                except Exception:
                    pass
            st.download_button("⬇️ Baixar curvas ajustadas (XLSX)", data=xlsx_buf.getvalue(), file_name="deconv_curvas.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_curves_xlsx")
            st.caption("Arquivo XLSX gerado com engine automática (XlsxWriter ou openpyxl).")

with tab3:
    st.subheader("🎨 Configurações de Exportação Personalizada")
    
    if len(st.session_state.peaks) == 0:
        st.info("Adicione picos para usar a exportação personalizada.")
    else:
        # Configurações do layout em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Configurações Gerais**")
            
            # Configurações básicas
            plot_title = st.text_input("Título do gráfico", value="Deconvolução Espectral")
            x_title = st.text_input("Título do eixo X", value="X")
            y_title = st.text_input("Título do eixo Y", value="Intensidade")
            
            # Configurações de fonte
            font_family = st.selectbox("Família da fonte", 
                                     ["Arial", "Times New Roman", "Helvetica", "Courier", "Georgia"], 
                                     index=0)
            font_size = st.slider("Tamanho da fonte", 8, 20, 12)
            title_font_size = st.slider("Tamanho da fonte do título", 12, 24, 16)
            
            # Configurações de cor de fundo
            background_color = st.selectbox("Cor de fundo", 
                                          ["white", "lightgray", "#f0f0f0", "#fafafa"], 
                                          index=0)
            
            # Configurações da legenda
            show_legend = st.checkbox("Mostrar legenda", value=True)
            if show_legend:
                legend_position = st.selectbox("Posição da legenda", 
                                             ["right", "top", "bottom"], 
                                             index=0)
        
        with col2:
            st.markdown("**🎨 Cores e Estilos**")
            
            # Cores principais
            data_color = st.color_picker("Cor dos dados", value="#4DA3FF")
            fit_color = st.color_picker("Cor do ajuste total", value="#FF6EC7")
            residual_color = st.color_picker("Cor dos resíduos", value="#FF4D4D")
            
            # Configurações de linha
            line_width_data = st.slider("Espessura da linha dos dados", 1, 5, 2)
            line_width_fit = st.slider("Espessura da linha do ajuste", 1, 5, 3)
            line_width_components = st.slider("Espessura das linhas dos componentes", 1, 4, 2)
            
            # Configurações dos componentes
            fill_components = st.checkbox("Preencher área dos componentes", value=False)
            if fill_components:
                fill_opacity = st.slider("Opacidade do preenchimento", 0.1, 1.0, 0.3)
            else:
                fill_opacity = 0.3
            
            component_opacity = st.slider("Opacidade dos componentes", 0.1, 1.0, 0.7)
            
            # Configurações das linhas de centro
            show_centers_export = st.checkbox("Mostrar linhas de centro", value=True)
            if show_centers_export:
                center_line_color = st.color_picker("Cor das linhas de centro", value="#666666")
                center_line_style = st.selectbox("Estilo das linhas de centro", 
                                                ["dash", "dot", "solid"], index=0)
                center_line_width = st.slider("Espessura das linhas de centro", 1, 3, 1)
            else:
                center_line_color = "#666666"
                center_line_style = "dash"
                center_line_width = 1
        
        st.markdown("**🌈 Cores dos Componentes Individuais**")
        
        # Cores personalizadas para cada pico
        component_colors = []
        default_colors = generate_peak_colors(len(st.session_state.peaks))
        
        cols_per_row = 4
        num_rows = (len(st.session_state.peaks) + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                peak_idx = row * cols_per_row + col_idx
                if peak_idx < len(st.session_state.peaks):
                    pk = st.session_state.peaks[peak_idx]
                    with cols[col_idx]:
                        color = st.color_picker(
                            f"Pico {peak_idx + 1}: {pk['type'][:8]}...", 
                            value=default_colors[peak_idx],
                            key=f"color_peak_{peak_idx}"
                        )
                        component_colors.append(color)
        
        st.divider()
        
        # Opções de exibição para exportação
        st.markdown("**👁️ Opções de Exibição**")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            export_show_fit = st.checkbox("Mostrar ajuste", value=True, key="export_fit")
        with col_exp2:
            export_show_components = st.checkbox("Mostrar componentes", value=True, key="export_comp")
        with col_exp3:
            export_show_residuals = st.checkbox("Mostrar resíduos", value=True, key="export_resid")
        
        # Configurar o dicionário de exportação
        export_config = {
            'data_color': data_color,
            'fit_color': fit_color,
            'residual_color': residual_color,
            'component_colors': component_colors,
            'fill_components': fill_components,
            'fill_opacity': fill_opacity,
            'component_opacity': component_opacity,
            'line_width_data': line_width_data,
            'line_width_fit': line_width_fit,
            'line_width_components': line_width_components,
            'show_centers': show_centers_export,
            'center_line_style': center_line_style,
            'center_line_color': center_line_color,
            'center_line_width': center_line_width,
            'background_color': background_color,
            'plot_title': plot_title,
            'x_title': x_title,
            'y_title': y_title,
            'font_family': font_family,
            'font_size': font_size,
            'title_font_size': title_font_size,
            'show_legend': show_legend,
            'legend_position': legend_position if show_legend else 'right'
        }
        
        st.divider()
        
        # Preview do gráfico personalizado
        st.markdown("**🔍 Preview do Gráfico Personalizado**")
        
        try:
            fig_export, _ = plot_figure_export(
                st.session_state.x, st.session_state.y, st.session_state.peaks, dec,
                export_config=export_config,
                show_fit=export_show_fit,
                show_components=export_show_components,
                show_residuals=export_show_residuals,
                y_range=st.session_state.y_range
            )
            
            st.plotly_chart(fig_export, use_container_width=True, key="preview_chart")
            
        except Exception as e:
            st.error(f"Erro na geração do preview: {e}")
            fig_export = None
        
        st.divider()
        
        # Botões de exportação personalizada
        st.markdown("**💾 Exportar Gráfico Personalizado**")
        
        col_download1, col_download2, col_download3 = st.columns(3)
        
        if fig_export is not None:
            with col_download1:
                # HTML interativo
                html_buf_custom = io.StringIO()
                fig_export.write_html(html_buf_custom, include_plotlyjs="cdn", full_html=True)
                html_bytes_custom = html_buf_custom.getvalue().encode("utf-8")
                st.download_button(
                    "⬇️ HTML Interativo",
                    data=html_bytes_custom,
                    file_name="deconv_personalizado.html",
                    mime="text/html",
                    key="dl_custom_html"
                )
            
            with col_download2:
                # Tentar exportar como PNG (requer kaleido)
                try:
                    png_bytes = fig_export.to_image(format="png", width=1200, height=800)
                    st.download_button(
                        "⬇️ PNG (Alta Res)",
                        data=png_bytes,
                        file_name="deconv_personalizado.png",
                        mime="image/png",
                        key="dl_custom_png"
                    )
                except Exception:
                    st.button("PNG (instale kaleido)", disabled=True, 
                             help="Execute: pip install kaleido")
            
            with col_download3:
                # Tentar exportar como PDF (requer kaleido)
                try:
                    pdf_bytes = fig_export.to_image(format="pdf", width=1200, height=800)
                    st.download_button(
                        "⬇️ PDF Vetorial",
                        data=pdf_bytes,
                        file_name="deconv_personalizado.pdf",
                        mime="application/pdf",
                        key="dl_custom_pdf"
                    )
                except Exception:
                    st.button("PDF (instale kaleido)", disabled=True, 
                             help="Execute: pip install kaleido")
        
        # Exportar configurações
        st.divider()
        st.markdown("**⚙️ Salvar/Carregar Configurações**")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            # Salvar configurações
            config_json = json.dumps(export_config, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Salvar Configurações",
                data=config_json,
                file_name="config_exportacao.json",
                mime="application/json",
                key="dl_config_json"
            )
        
        with col_config2:
            # Carregar configurações
            uploaded_config = st.file_uploader(
                "Carregar Configurações",
                type="json",
                key="upload_config",
                help="Carregue um arquivo JSON de configurações salvo anteriormente"
            )
            
            if uploaded_config is not None:
                try:
                    config_data = json.load(uploaded_config)
                    st.success("Configurações carregadas! Atualize a página para aplicar.")
                    # Aqui você poderia implementar a aplicação automática das configurações
                except Exception as e:
                    st.error(f"Erro ao carregar configurações: {e}")
        
        # Instruções adicionais
        st.info("""
        **💡 Dicas para Exportação:**
        - Para PNG/PDF de alta qualidade, instale: `pip install kaleido`
        - Use HTML interativo para apresentações online
        - Salve suas configurações para reutilizar em outros projetos
        - O preview mostra exatamente como será o gráfico exportado
        """)
    
st.caption("Dica: para exportar PNG em servidores sem Chrome, prefira baixar HTML e usar captura em alta resolução localmente.")("🔄 Auto Y"):
        y_min, y_max = (float(np.nanmin(y)) if y.size else 0.0, float(np.nanmax(y)) if y.size else 1.0)
    st.session_state.y_range = [y_min, y_max]

    st.divider()
    st.subheader("🔍 Detecção Automática de Picos")
    prom = st.number_input("Proeminência mínima", value=0.05, step=0.01, format="%.3f")
    dist = st.number_input("Distância mínima entre picos (pontos)", value=30, step=1, min_value=1)
    if st.button("Detectar picos"):
        pks, _ = find_peaks(y, prominence=prom, distance=int(dist))
        st.session_state.peaks = []
        if len(pks) == 0:
            st.warning("Nenhum pico detectado com esses parâmetros.")
        else:
            x_min, x_max = float(np.min(x)), float(np.max(x))
            xr = x_max - x_min
            for idx in pks:
                amp = float(y[idx])
                xc = float(x[idx])
                half = amp / 2.0
                li = idx
                ri = idx
                while li > 0 and y[li] > half:
                    li -= 1
                while ri < len(y) - 1 and y[ri] > half:
                    ri += 1
                width = max(1e-6, float(x[min(ri, len(x)-1)] - x[max(li, 0)]))
                sigma_guess = max(width / 2.355, xr / 200.0)
                st.session_state.peaks.append({
                    "type": "Gaussiana",
                    "params": [amp, xc, sigma_guess],
                    "bounds": [
                        (0.0, amp * 2.0),
                        (x_min, x_max),
                        (1e-6, xr)
                    ]
                })
            st.success(f"{len(pks)} picos adicionados.")

    st.divider()
    st.subheader("🎯 Método de Ajuste")
    fit_method = st.selectbox("Otimizador", ["curve_fit", "differential_evolution"], index=0)
    st.caption("Se o ajuste travar, tente diminuir o número de picos ou trocar o método.")

st.title("📊 Deconvolução Espectral Avançada")
col_ctrl, col_plot = st.columns([1, 2], gap="large")

with col_ctrl:
    st.subheader("➕ Adicionar Pico")
    pk_type = st.selectbox("Tipo", list(dec.peak_models.keys()), index=0, key="add_type")
    x_min, x_max = float(np.min(st.session_state.x)), float(np.max(st.session_state.x))
    xr = x_max - x_min
    y_max = float(np.max(st.session_state.y))

    if st.button
