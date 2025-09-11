# -*- coding: utf-8 -*-
import io, sys, os, time, json, importlib
import platform
import streamlit as st

st.set_page_config(page_title="Health Check • Streamlit Cloud", page_icon="🩺", layout="wide")

def pill(ok: bool) -> str:
    return "🟢 OK" if ok else "🔴 FALHOU"

st.title("🩺 Health Check — Streamlit Community Cloud")

# --- Versions ---
cols = st.columns(4)
with cols[0]:
    st.metric("Python", platform.python_version())
with cols[1]:
    st.metric("Streamlit", getattr(importlib.import_module("streamlit"), "__version__", "?"))
with cols[2]:
    try:
        import numpy as np
        st.metric("NumPy", np.__version__)
    except Exception as e:
        st.error(f"NumPy: {e}")
with cols[3]:
    try:
        import pandas as pd
        st.metric("Pandas", pd.__version__)
    except Exception as e:
        st.error(f"Pandas: {e}")

# --- Optional libs ---
opt = {
    "scipy": "scipy",
    "plotly": "plotly",
    "openpyxl": "openpyxl",
    "XlsxWriter": "xlsxwriter",
}
st.subheader("📦 Bibliotecas opcionais")
cc = st.columns(len(opt))
for (i,(name, mod)) in enumerate(opt.items()):
    with cc[i]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "OK")
            st.success(f"{name}: {ver}")
        except Exception as e:
            st.warning(f"{name}: não encontrada")

# --- Rerun check ---
st.subheader("🔁 Rerun")
has_rerun = hasattr(st, "rerun") or hasattr(st, "experimental_rerun")
st.write("Função disponível:", pill(has_rerun))
if st.button("Testar rerun agora"):
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# --- Write permissions (/tmp) ---
st.subheader("💾 Escrita em disco (tmp)")
try:
    p = "/tmp/healthcheck_write.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("ok")
    ok1 = os.path.exists(p)
except Exception as e:
    ok1 = False
st.write("Escrita em /tmp:", pill(ok1))

# --- Plotly to HTML ---
st.subheader("📈 Plotly")
try:
    import plotly.graph_objects as go
    import numpy as np
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="seno"))
    st.plotly_chart(fig, use_container_width=True)
    html = io.StringIO()
    fig.write_html(html, include_plotlyjs="cdn", full_html=True)
    ok_plotly = len(html.getvalue()) > 1000
except Exception as e:
    ok_plotly = False
st.write("Exportar HTML:", pill(ok_plotly))

# --- Excel export (engine auto) ---
st.subheader("📤 Exportação Excel (engine automática)")
excel_ok = False
try:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"x": np.arange(5), "y": np.arange(5)**2})
    buf = io.BytesIO()
    engine = None
    try:
        import xlsxwriter  # type: ignore
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # type: ignore
            engine = "openpyxl"
        except Exception:
            engine = None
    if engine is None:
        st.warning("Nem XlsxWriter nem openpyxl instalados.")
    else:
        with pd.ExcelWriter(buf, engine=engine) as writer:
            df.to_excel(writer, index=False, sheet_name="teste")
        st.download_button("Baixar XLSX de teste", data=buf.getvalue(), file_name="healthcheck.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx_test")
        excel_ok = True
except Exception as e:
    excel_ok = False
st.write("Gerar XLSX:", pill(excel_ok))

# --- Cache sanity ---
st.subheader("🗃️ Cache")
@st.cache_data
def _cached_sum(n: int):
    import numpy as np
    a = np.arange(n)
    return int(a.sum())
ok_cache = _cached_sum(1000) == 499500
st.write("st.cache_data:", pill(ok_cache))

# --- Summary ---
all_ok = has_rerun and ok1 and ok_plotly and excel_ok
st.markdown("---")
st.subheader("Resumo")
st.write("- Ambiente OK:", pill(all_ok))
st.caption("Se algo estiver em vermelho, verifique `requirements.txt` e os logs de build do app.")
