
# deconv_app.py
# Streamlit app for 1D signal deconvolution (Gaussian/Lorentzian) with advanced customization & exports
# Author: ChatGPT (GPT-5 Thinking)
# Run locally: streamlit run deconv_app.py

import io
import json
import time
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# SciPy pieces
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import grey_opening
from scipy.optimize import curve_fit

# ------------- Utils: Styling & Presets -------------

OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

VIRIDIS_10 = [
    "#440154", "#472c7a", "#3b518b", "#2c718e", "#21918c",
    "#27ad81", "#5cc863", "#aadc32", "#fde725", "#fefcc6"
]

HIGH_CONTRAST = [
    "#000000", "#ffffff", "#ff0000", "#00ffff", "#ffff00",
    "#00ff00", "#ff00ff", "#0000ff", "#ff8800", "#8800ff"
]

DEFAULT_PRESET = {
    "theme": "scientific-dark",
    "font": {"family": "Inter, Arial, Helvetica, sans-serif", "title": 20, "axes": 14, "ticks": 12, "legend": 12},
    "axes": {"x": {"scale": "linear", "tickformat": "~s"}, "y": {"scale": "linear", "minorticks": False}},
    "grid": {"show": True, "width": 0.6, "alpha": 0.25},
    "legend": {"position": "outside right", "bg": "transparent"},
    "line": {"width": 2.0, "style": "solid", "opacity": 1.0, "marker": "circle", "markersize": 6},
    "colors": {"palette": "okabe-ito", "lockByLabel": False},
    "error": {"show": False, "mode": "band"},
    "export": {"dpi": 300, "transparent": True, "formats": ["png", "svg", "pdf"]},
    "preprocess": {
        "smoothing": {"method": "savgol", "window": 11, "order": 3},
        "baseline": {"method": "asls", "lam": 1e5, "p": 0.01, "iterations": 10, "peaks": "down"},
        "outliers": {"method": "iqr", "remove": False}
    },
    "layout_preset": "Científico"
}

THEMES_MAP = {
    "Claro": "plotly",
    "Escuro": "plotly_dark",
    "Científico (cinza)": "ggplot2"
}

PALETTE_MAP = {
    "okabe-ito": OKABE_ITO,
    "viridis-10": VIRIDIS_10,
    "high-contrast": HIGH_CONTRAST
}

LINE_STYLES = {
    "sólida": None,
    "tracejada": "dash",
    "ponto-traço": "dot",
}

MARKERS = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]

# ------------- Data Classes -------------

@dataclass
class Peak:
    center: float
    amplitude: float
    width: float

@dataclass
class FitConfig:
    model: str           # 'gaussian' | 'lorentzian'
    baseline_first: bool # apply baseline & smoothing before fitting

# ------------- Math: Profiles -------------

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-0.5 * ((x - cen) / np.clip(wid, 1e-12, None))**2)

def lorentzian(x, amp, cen, wid):
    return amp * (np.clip(wid, 1e-12, None)**2) / ((x - cen)**2 + np.clip(wid, 1e-12, None)**2)

def sum_of_profiles(x, params, model="gaussian"):
    y = np.zeros_like(x, dtype=float)
    for (amp, cen, wid) in params:
        if model == "gaussian":
            y += gaussian(x, amp, cen, wid)
        else:
            y += lorentzian(x, amp, cen, wid)
    return y

# ------------- Baseline Methods -------------

def baseline_asls(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L)).tocsc()
    w = np.ones(L)
    for _ in range(int(niter)):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_poly(x, y, order=3):
    coeffs = np.polyfit(x, y, deg=int(order))
    return np.polyval(coeffs, x)

def baseline_tophat(y, size=101):
    size = max(3, int(size) // 2 * 2 + 1)  # make odd
    opened = grey_opening(y, size=size)
    return opened

# ------------- Smoothing -------------

def smooth_signal(x, y, method="savgol", **kwargs):
    if method == "moving_average":
        N = int(kwargs.get("window", 5))
        if N < 2: 
            return y
        return np.convolve(y, np.ones(N)/N, mode="same")
    elif method == "savgol":
        w = int(kwargs.get("window", 11))
        o = int(kwargs.get("order", 3))
        if w % 2 == 0: w += 1
        w = max(3, w)
        o = max(1, o)
        if w <= o: w = o + 2 + (o % 2)  # ensure w > o and odd
        return savgol_filter(y, window_length=w, polyorder=o, mode="interp")
    elif method == "spline":
        s = float(kwargs.get("s", 0.001)) * len(x)
        spl = UnivariateSpline(x, y, s=s)
        return spl(x)
    else:
        return y

# ------------- Peak Detection -------------

def detect_peaks(y, orientation="up", prominence=0.01, width=3, distance=None):
    y_use = y if orientation == "up" else -y
    peaks, props = find_peaks(
        y_use, 
        prominence=prominence * (np.nanmax(y_use) - np.nanmin(y_use)),
        width=width,
        distance=distance
    )
    return peaks, props

# ------------- Fit -------------

def _model_for_curvefit(x, *params, model="gaussian"):
    params_triplet = [(params[i], params[i+1], params[i+2]) for i in range(0, len(params), 3)]
    return sum_of_profiles(x, params_triplet, model=model)

def fit_peaks(x, y, init_peaks: List[Peak], model="gaussian", bounds_factor=10.0):
    if len(init_peaks) == 0:
        raise ValueError("Nenhum pico informado para ajuste.")
    p0 = []
    lower = []
    upper = []
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    dy = np.nanmax(y) - np.nanmin(y)
    for pk in init_peaks:
        p0.extend([pk.amplitude, pk.center, max(pk.width, 1e-6)])
        lower.extend([0, x_min - (x_max - x_min)*0.1, 1e-6])
        upper.extend([dy*bounds_factor, x_max + (x_max - x_min)*0.1, (x_max - x_min)])
    try:
        popt, pcov = curve_fit(lambda xx, *pp: _model_for_curvefit(xx, *pp, model=model),
                               x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
    except Exception as e:
        raise RuntimeError(f"Falha no ajuste: {e}")
    fitted_params = [(popt[i], popt[i+1], popt[i+2]) for i in range(0, len(popt), 3)]
    y_fit = sum_of_profiles(x, fitted_params, model=model)
    return fitted_params, y_fit

# ------------- Export Helpers -------------

def fig_to_bytes(fig, fmt="png", scale=2, width=1200, height=700, transparent=True):
    import plotly.io as pio
    return pio.to_image(fig, format=fmt, scale=scale, width=width, height=height, engine="kaleido")

def bytes_download_link(data: bytes, filename: str, label: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">{label}</a>'
    return href

# ------------- Streamlit App -------------

st.set_page_config(page_title="Deconvolução 1D • Científico", page_icon="🧪", layout="wide")

# Sidebar: Presets & Theme
with st.sidebar:
    st.title("⚙️ Configurações")
    # Presets de layout
    layout_preset = st.selectbox("Preset de layout", ["Compacto", "Científico", "Apresentação"], index=1)
    theme_choice = st.selectbox("Tema", ["Escuro", "Claro", "Científico (cinza)"], index=0)
    palette_choice = st.selectbox("Paleta", ["okabe-ito", "viridis-10", "high-contrast"], index=0)
    alpha_bands = st.slider("Transparência das bandas", 0.05, 0.9, 0.35, 0.05)
    line_style = st.selectbox("Estilo da linha", list(LINE_STYLES.keys()), index=0)
    marker_style = st.selectbox("Marcador", MARKERS, index=0)
    line_width = st.slider("Espessura da linha", 1.0, 6.0, 2.2, 0.2)
    marker_size = st.slider("Tamanho do marcador", 0, 16, 0, 1)
    font_title = st.slider("Fonte - Título", 12, 28, 20, 1)
    font_axes = st.slider("Fonte - Eixos", 8, 22, 14, 1)
    font_ticks = st.slider("Fonte - Ticks", 8, 20, 12, 1)
    legend_pos = st.selectbox("Legenda", ["inside", "outside right", "outside bottom"], index=1)
    grid_show = st.checkbox("Mostrar grade", True)
    grid_alpha = st.slider("Transparência da grade", 0.0, 1.0, 0.25, 0.05)

    st.markdown("---")
    # Save/Load preset JSON
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Salvar preset JSON"):
            preset = DEFAULT_PRESET.copy()
            preset["layout_preset"] = layout_preset
            preset["theme"] = theme_choice
            preset["colors"]["palette"] = palette_choice
            preset["line"]["style"] = line_style
            preset["line"]["width"] = line_width
            preset["line"]["marker"] = marker_style
            preset["line"]["markersize"] = marker_size
            preset["font"]["title"] = font_title
            preset["font"]["axes"] = font_axes
            preset["font"]["ticks"] = font_ticks
            preset["grid"]["show"] = grid_show
            preset["grid"]["alpha"] = grid_alpha
            st.download_button("Baixar preset.json",
                               data=json.dumps(preset, indent=2).encode("utf-8"),
                               file_name="preset.json",
                               mime="application/json")
    with col_b:
        up_preset = st.file_uploader("Carregar preset JSON", type=["json"], label_visibility="collapsed")
        if up_preset is not None:
            try:
                loaded = json.load(up_preset)
                # apply a subset safely
                layout_preset = loaded.get("layout_preset", layout_preset)
                theme_choice = loaded.get("theme", theme_choice)
                palette_choice = loaded.get("colors", {}).get("palette", palette_choice)
                line_style = loaded.get("line", {}).get("style", line_style)
                line_width = float(loaded.get("line", {}).get("width", line_width))
                marker_style = loaded.get("line", {}).get("marker", marker_style)
                marker_size = int(loaded.get("line", {}).get("markersize", marker_size))
                ft = loaded.get("font", {})
                font_title = int(ft.get("title", font_title))
                font_axes = int(ft.get("axes", font_axes))
                font_ticks = int(ft.get("ticks", font_ticks))
                grid_show = bool(loaded.get("grid", {}).get("show", grid_show))
                grid_alpha = float(loaded.get("grid", {}).get("alpha", grid_alpha))
                st.success("Preset carregado e aplicado.")
            except Exception as e:
                st.error(f"Falha ao carregar preset: {e}")

# Tabs
tab_dados, tab_pre, tab_graf, tab_export = st.tabs(["📁 Dados", "🧪 Pré-processamento", "📈 Gráfico", "💾 Exportar"])

# ---------- Dados
with tab_dados:
    st.subheader("Importar dados")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        file = st.file_uploader("Arquivo (CSV, XLSX, TXT, DPT)", type=["csv","xlsx","txt","dpt"])
    with col2:
        decimal = st.selectbox("Decimal", [",", "."], index=1)
        sep_auto = st.checkbox("Separador automático", True)
        sep = None if sep_auto else st.text_input("Separador (ex.: ; , \\t)", value=",")
    with col3:
        header_rows = st.number_input("Pular linhas de header", min_value=0, value=0, step=1)
        x_col_name = st.text_input("Nome da coluna X", value="x")
        y_col_name = st.text_input("Nome da coluna Y", value="y")

    if file is not None:
        try:
            ext = Path(file.name).suffix.lower()
            if ext == ".xlsx":
                df_raw = pd.read_excel(file)
            else:
                df_raw = pd.read_csv(file, sep=sep, engine="python", skiprows=header_rows, decimal=decimal)
            # Try to guess columns
            if x_col_name not in df_raw.columns or y_col_name not in df_raw.columns:
                # fallback to first two columns
                df_raw.columns = [str(c).strip() for c in df_raw.columns]
                x_guess = df_raw.columns[0]
                y_guess = df_raw.columns[1] if len(df_raw.columns) > 1 else df_raw.columns[0]
                x_col_name = x_guess
                y_col_name = y_guess
            x = pd.to_numeric(df_raw[x_col_name], errors="coerce").to_numpy()
            y = pd.to_numeric(df_raw[y_col_name], errors="coerce").to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            st.success(f"Carregado: {file.name} — {len(x)} pontos.")
            st.dataframe(df_raw.head(8))
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            x = y = None
    else:
        x = y = None
        st.info("Aguardando arquivo...")

# ---------- Pré-processamento
with tab_pre:
    st.subheader("Pré-processamento científico")
    if x is None:
        st.info("Carregue um arquivo na aba **Dados**.")
    else:
        pre_col1, pre_col2, pre_col3 = st.columns(3)
        with pre_col1:
            st.markdown("**Suavização**")
            sm_method = st.selectbox("Método", ["savgol", "moving_average", "spline"], index=0)
            sm_window = st.slider("Janela", 3, 199, 11, 2)
            sm_order = st.slider("Ordem (se aplicável)", 1, 9, 3, 1)
        with pre_col2:
            st.markdown("**Baseline**")
            bl_method = st.selectbox("Método", ["asls", "polinômio", "tophat"], index=0)
            bl_peaks_orient = st.selectbox("Orientação dos picos", ["↑ (para cima)", "↓ (para baixo)"], index=1)
            orient = "up" if "para cima" in bl_peaks_orient else "down"
            if bl_method == "asls":
                lam = st.number_input("λ (ASLS)", min_value=1e2, max_value=1e9, value=1e5, step=1e3, format="%.0f")
                p = st.slider("p (assimetria)", 0.001, 0.2, 0.01, 0.001)
                bl_params = {"lam": float(lam), "p": float(p), "iterations": 10}
            elif bl_method == "polinômio":
                order = st.slider("Ordem", 1, 9, 3, 1)
                bl_params = {"order": order}
            else:
                size = st.slider("Tamanho TopHat (pontos)", 5, 1001, 101, 2)
                bl_params = {"size": size}
        with pre_col3:
            st.markdown("**Detecção de picos (auto)**")
            prom = st.slider("Proeminência (relativa)", 0.0, 0.5, 0.05, 0.01)
            width = st.slider("Largura mínima (pontos)", 1, 200, 5, 1)
            distance = st.slider("Distância mínima (pontos)", 0, 1000, 0, 1) or None
            roi = st.text_input("ROI (xmin,xmax) opcional", "")

        # Process
        t0 = time.time()
        y_sm = smooth_signal(x, y, method=sm_method, window=sm_window, order=sm_order, s=0.001)
        if bl_method == "asls":
            y_use = y_sm if orient == "up" else -y_sm
            baseline = baseline_asls(y_use, lam=bl_params["lam"], p=bl_params["p"], niter=bl_params["iterations"])
            baseline = baseline if orient == "up" else -baseline
        elif bl_method == "polinômio":
            baseline = baseline_poly(x, y_sm, order=bl_params["order"])
        else:
            baseline = baseline_tophat(y_sm, size=bl_params["size"])

        y_corr = y_sm - baseline
        if orient == "down":
            y_corr = -y_corr  # ensure peaks are "up" after correction if they were downwards

        # ROI clip
        if roi.strip():
            try:
                xmin, xmax = map(float, roi.split(","))
                sel = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_proc, y_proc = x[sel], y_corr[sel]
            except Exception:
                st.warning("ROI inválida. Usando todo o intervalo.")
                x_proc, y_proc = x, y_corr
        else:
            x_proc, y_proc = x, y_corr

        idx_peaks, props = detect_peaks(y_proc, orientation="up", prominence=prom, width=width, distance=distance)
        t1 = time.time()

        st.success(f"Pré-processamento concluído em {(t1-t0)*1e3:.1f} ms.")
        st.line_chart(pd.DataFrame({"x": x, "raw": y, "smoothed": y_sm, "baseline": baseline}).set_index("x"))

        st.markdown("**Picos detectados (auto)**")
        auto_peaks_df = pd.DataFrame({
            "index": idx_peaks,
            "center": x_proc[idx_peaks] if len(idx_peaks) else [],
            "amplitude_est": y_proc[idx_peaks] if len(idx_peaks) else [],
            "width_est": props["widths"] if len(idx_peaks) else []
        })
        st.dataframe(auto_peaks_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Picos (entrada **manual**: centro, amplitude, largura)")
        st.caption("Edite a tabela abaixo. Clique em **+** para adicionar linhas. Você pode iniciar a partir da detecção automática.")
        if "peaks_df" not in st.session_state:
            st.session_state.peaks_df = pd.DataFrame(columns=["center","amplitude","width"])
        colm1, colm2 = st.columns(2)
        with colm1:
            if st.button("⬇️ Usar picos detectados como chute inicial"):
                st.session_state.peaks_df = auto_peaks_df[["center","amplitude_est","width_est"]].rename(
                    columns={"amplitude_est":"amplitude","width_est":"width"}).copy()
        with colm2:
            if st.button("🗑 Limpar tabela"):
                st.session_state.peaks_df = pd.DataFrame(columns=["center","amplitude","width"])

        peaks_df = st.data_editor(
            st.session_state.peaks_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "center": st.column_config.NumberColumn("Centro", step=0.01),
                "amplitude": st.column_config.NumberColumn("Amplitude", step=0.01),
                "width": st.column_config.NumberColumn("Largura", step=0.01, help="Desvio (Gauss) ou gamma (Lorentz)")
            },
            key="peaks_editor"
        )
        st.session_state.peaks_df = peaks_df

        # Store processed arrays for next tab
        st.session_state.proc_data = {
            "x": x, "y_raw": y, "y_sm": y_sm, "baseline": baseline, "y_corr": y_corr,
            "x_proc": x_proc, "y_proc": y_proc, "idx_peaks": idx_peaks.tolist()
        }

# ---------- Gráfico & Ajuste
with tab_graf:
    st.subheader("Ajuste e plotagem")
    if "proc_data" not in st.session_state or st.session_state.get("peaks_df") is None or x is None:
        st.info("Finalize a etapa de **Pré-processamento** e defina picos.")
    else:
        proc = st.session_state.proc_data
        x, y, y_sm, baseline, y_corr = proc["x"], proc["y_raw"], proc["y_sm"], proc["baseline"], proc["y_corr"]
        x_proc, y_proc = proc["x_proc"], proc["y_proc"]

        # Ajuste config
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            model = st.selectbox("Modelo de pico", ["gaussian", "lorentzian"], index=0)
            residual_on_y2 = st.checkbox("Mostrar resíduo no eixo Y2", True)
        with colm2:
            si_ticks = st.selectbox("Formato de ticks", ["padrão", "SI (k, M, µ)", "notação científica"], index=1)
            rot_ticks = st.slider("Rotação ticks X (°)", 0, 90, 0, 5)
            percent_lims = st.slider("Limites por percentil (Ycorr)", 0, 10, (1, 99), 1)
        with colm3:
            dragmode = st.selectbox("Ferramenta de arrasto", ["zoom", "pan", "select", "lasso"], index=0)
            hovermode = st.selectbox("Hover", ["x unified", "closest"], index=0)
            crosshair = st.checkbox("Crosshair (spikes)", True)

        # Build init peaks list
        peaks_list = []
        peaks_df = st.session_state.peaks_df.dropna()
        for _, row in peaks_df.iterrows():
            try:
                peaks_list.append(Peak(center=float(row["center"]), amplitude=float(row["amplitude"]), width=float(row["width"])))
            except Exception:
                pass

        fitted_params = []
        yfit = np.zeros_like(x_proc)
        fit_ok = False
        fit_err = None

        if len(peaks_list) >= 1:
            try:
                fitted_params, yfit = fit_peaks(x_proc, y_proc, peaks_list, model=model, bounds_factor=10.0)
                fit_ok = True
            except Exception as e:
                fit_err = str(e)

        # Figure
        palette = PALETTE_MAP.get(palette_choice, OKABE_ITO)
        fig = go.Figure()
        # Raw & preprocessed lines
        fig.add_trace(go.Scatter(x=x, y=y, name="Original", mode="lines+markers" if marker_size>0 else "lines",
                                 line=dict(width=line_width, dash=LINE_STYLES[line_style]),
                                 marker=dict(size=marker_size, symbol=marker_style)))
        fig.add_trace(go.Scatter(x=x, y=y_sm, name="Suavizado", mode="lines",
                                 line=dict(width=max(1.0, line_width-0.6), dash="dot")))
        fig.add_trace(go.Scatter(x=x, y=baseline, name="Baseline", mode="lines",
                                 line=dict(width=1.5, dash="dash", color="#888888")))
        fig.add_trace(go.Scatter(x=x, y=y_corr, name="Corrigido (para cima)", mode="lines",
                                 line=dict(width=max(1.2, line_width-0.2))))

        # Components (bands) + filled areas
        if fit_ok:
            # Full fit on x_proc
            fig.add_trace(go.Scatter(x=x_proc, y=yfit, name=f"Soma ({model})", mode="lines",
                                     line=dict(width=line_width+0.6)))
            # Residual
            residual = y_proc - yfit
            if residual_on_y2:
                fig.add_trace(go.Scatter(x=x_proc, y=residual, name="Resíduo", mode="lines",
                                         yaxis="y2", line=dict(width=1.2, color="#444444")))
            else:
                fig.add_trace(go.Scatter(x=x_proc, y=residual, name="Resíduo", mode="lines",
                                         line=dict(width=1.2, color="#444444")))

            # Individual components with fill
            for i, (amp, cen, wid) in enumerate(fitted_params):
                if model == "gaussian":
                    comp = gaussian(x_proc, amp, cen, wid)
                else:
                    comp = lorentzian(x_proc, amp, cen, wid)
                color = palette[i % len(palette)]
                # Filled band
                fig.add_trace(go.Scatter(
                    x=x_proc, y=comp, mode="lines", name=f"Comp {i+1} (c={cen:.3g})",
                    line=dict(width=line_width, color=color),
                    fill="tozeroy", fillcolor=color.replace("#", "rgba(") if False else color, opacity=alpha_bands
                ))

        # Axes & theme
        template = THEMES_MAP.get(theme_choice, "plotly_dark")
        fig.update_layout(template=template)

        # Legend
        if legend_pos == "inside":
            fig.update_layout(legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0)" if theme_choice!="Escuro" else "rgba(0,0,0,0)"))
        elif legend_pos == "outside right":
            fig.update_layout(legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left"))
        else:
            fig.update_layout(legend=dict(orientation="h", y=-0.2))

        # Grid
        fig.update_xaxes(showgrid=grid_show, gridwidth=DEFAULT_PRESET["grid"]["width"], gridcolor=f"rgba(128,128,128,{grid_alpha})")
        fig.update_yaxes(showgrid=grid_show, gridwidth=DEFAULT_PRESET["grid"]["width"], gridcolor=f"rgba(128,128,128,{grid_alpha})")

        # Fonts
        fig.update_layout(
            title=dict(text="Deconvolução 1D", font=dict(size=font_title)),
            xaxis=dict(title="X", tickangle=rot_ticks, tickfont=dict(size=font_ticks)),
            yaxis=dict(title="Y", tickfont=dict(size=font_ticks)),
            font=dict(size=font_axes),
            dragmode=dragmode,
            hovermode=hovermode
        )
        if crosshair:
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")

        # Tick formatting
        if si_ticks == "SI (k, M, µ)":
            fig.update_xaxes(tickformat="~s")
            fig.update_yaxes(tickformat="~s")
        elif si_ticks == "notação científica":
            fig.update_xaxes(tickformat=".2e")
            fig.update_yaxes(tickformat=".2e")

        # Percentile Y limits
        ypc_lo, ypc_hi = np.percentile(y_corr[np.isfinite(y_corr)], [percent_lims[0], percent_lims[1]])
        fig.update_yaxes(range=[ypc_lo, ypc_hi])

        # Secondary y-axis for residual if chosen
        if residual_on_y2:
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Resíduo"))

        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displaylogo": False})

        # Alt-text
        st.caption("Alt-text: gráfico linha com dados originais, suavizados, baseline, sinal corrigido e componentes ajustados por picos, incluindo áreas preenchidas coloridas para cada banda.")

        # Save fig & data into session for export
        st.session_state.fig = fig
        st.session_state.export_bundle = {
            "x": x.tolist(), "y_raw": y.tolist(), "y_sm": y_sm.tolist(),
            "baseline": baseline.tolist(), "y_corr": y_corr.tolist(),
            "x_proc": x_proc.tolist(), "y_proc": y_proc.tolist(),
            "fit_ok": fit_ok,
            "model": model,
            "fitted_params": fitted_params,
            "residual_on_y2": residual_on_y2
        }

# ---------- Export
with tab_export:
    st.subheader("Exportação caprichada")
    if "fig" not in st.session_state:
        st.info("Gere um gráfico na aba **Gráfico**.")
    else:
        fig = st.session_state.fig
        bundle = st.session_state.export_bundle

        colx1, colx2, colx3 = st.columns(3)
        with colx1:
            bg_transparent = st.checkbox("Fundo transparente", True)
            dpi = st.slider("DPI/escala", 1, 6, 3, 1)
        with colx2:
            width = st.number_input("Largura px", 600, 4000, 1600, 50)
            height = st.number_input("Altura px", 400, 3000, 900, 50)
        with colx3:
            fname_base = st.text_input("Nome base do arquivo", value="deconv")
            fmt = st.multiselect("Formatos", ["png", "svg", "pdf"], default=["png","svg","pdf"])

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{fname_base}_{timestamp}"
        # Export buttons
        exp_cols = st.columns(len(fmt))
        for i, f in enumerate(fmt):
            with exp_cols[i]:
                if st.button(f"Exportar {f.upper()}"):
                    try:
                        # Transparent background
                        if bg_transparent:
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        img_bytes = fig_to_bytes(fig, fmt=f, scale=dpi, width=int(width), height=int(height), transparent=bg_transparent)
                        st.download_button(f"Baixar {f.upper()}", data=img_bytes, file_name=f"{fname}.{f}")
                        st.success(f"{f.upper()} gerado.")
                    except Exception as e:
                        st.error(f"Falha ao exportar {f}: {e}")

        st.markdown("---")
        st.markdown("### Exportar dados processados + metadados (JSON)")
        if st.button("Gerar CSV + JSON"):
            df_proc = pd.DataFrame({
                "x": bundle["x"],
                "y_raw": bundle["y_raw"],
                "y_sm": bundle["y_sm"],
                "baseline": bundle["baseline"],
                "y_corr": bundle["y_corr"]
            })
            # Per-component curves if fit_ok
            if bundle["fit_ok"] and len(bundle["fitted_params"]) > 0:
                x_proc = np.array(bundle["x_proc"])
                comps = []
                for j, (amp, cen, wid) in enumerate(bundle["fitted_params"]):
                    if bundle["model"] == "gaussian":
                        comp = gaussian(x_proc, amp, cen, wid)
                    else:
                        comp = lorentzian(x_proc, amp, cen, wid)
                    df_proc[f"comp_{j+1}"] = np.interp(df_proc["x"].values, x_proc, comp)

            csv_bytes = df_proc.to_csv(index=False).encode("utf-8")

            meta = {
                "app": "deconv_app.py",
                "timestamp": timestamp,
                "preset": {
                    "layout_preset": st.session_state.get("layout_preset", "Científico"),
                    "theme": st.session_state.get("theme", "Escuro"),
                    "palette": st.session_state.get("palette", "okabe-ito"),
                },
                "preprocess": {
                    # Note: Not all controls stored here; add as needed
                },
                "fit": {
                    "model": bundle["model"],
                    "params": [{"amplitude": float(a), "center": float(c), "width": float(w)} for (a,c,w) in bundle["fitted_params"]]
                }
            }
            json_bytes = json.dumps(meta, indent=2).encode("utf-8")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Baixar dados.csv", data=csv_bytes, file_name=f"{fname}.csv", mime="text/csv")
            with c2:
                st.download_button("Baixar metadata.json", data=json_bytes, file_name=f"{fname}.json", mime="application/json")

        st.markdown("---")
        st.markdown("### Relatório rápido (HTML)")
        if st.button("Gerar relatório HTML"):
            try:
                png_bytes = fig_to_bytes(fig, fmt="png", scale=2, width=int(width), height=int(height), transparent=bg_transparent)
                img_b64 = base64.b64encode(png_bytes).decode()
                html = f"""
                <html><head><meta charset="utf-8"><title>Relatório</title></head>
                <body style="font-family: Arial, sans-serif; padding: 1rem;">
                <h2>Relatório de Deconvolução</h2>
                <p><b>Gerado em:</b> {timestamp}</p>
                <img src="data:image/png;base64,{img_b64}" alt="Gráfico de deconvolução" style="max-width: 100%; border:1px solid #ccc;">
                <h3>Parâmetros</h3>
                <pre>{json.dumps(bundle['fitted_params'], indent=2)}</pre>
                </body></html>
                """.strip()
                st.download_button("Baixar relatório.html", data=html.encode("utf-8"),
                                   file_name=f"{fname}.html", mime="text/html")
                st.success("Relatório gerado.")
            except Exception as e:
                st.error(f"Falha no relatório: {e}")

st.markdown("---")
st.caption("Dicas: • Use ROI para focar nos picos de interesse • Ajuste a orientação (↑/↓) para picos negativos • Exporte em SVG para publicação • Presets JSON permitem reusar seu estilo entre apps.")
