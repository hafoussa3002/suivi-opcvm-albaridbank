from __future__ import annotations

import base64
from datetime import date, datetime, timedelta
from io import BytesIO
import math
from pathlib import Path
import re
import unicodedata

import pandas as pd
import streamlit as st
from vba_finance import (
    DatePr_Cp,
    DateSerial,
    calcul_taux,
    mati,
    prix_amortissable,
)

from storage import (
    add_asfim_files,
    add_bam_files,
    get_asfim_records,
    get_bam_records,
    init_storage,
    list_asfim_dates,
    list_asfim_files,
    list_bam_dates,
    list_bam_files,
    summarize_asfim_history,
    summarize_bam_history,
)

st.set_page_config(page_title="Suivi des OPCVM", layout="wide")
init_storage()

APP_USER = "sara"
APP_PASSWORD = "albarid2026"


def _fix_ui_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    txt = value
    # Recover common mojibake (UTF-8 interpreted as latin1), max 2 passes.
    for _ in range(2):
        if any(ch in txt for ch in ("Ã", "Â", "â")):
            try:
                fixed = txt.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
                if fixed:
                    txt = fixed
                    continue
            except Exception:
                pass
        break
    replacements = {
        "Donn?es": "Données",
        "donn?es": "données",
    }
    for bad, good in replacements.items():
        txt = txt.replace(bad, good)
    return txt


def _fix_ui_obj(value: object) -> object:
    if isinstance(value, str):
        return _fix_ui_text(value)
    if isinstance(value, list):
        return [_fix_ui_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_fix_ui_obj(v) for v in value)
    return value


def _patch_streamlit_text_rendering() -> None:
    names = [
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "error",
        "success",
        "button",
        "download_button",
        "text_input",
        "selectbox",
        "radio",
        "metric",
    ]
    for name in names:
        original = getattr(st, name, None)
        if original is None or getattr(original, "__name__", "") == "wrapped_ui_text":
            continue

        def wrapped_ui_text(*args, __orig=original, **kwargs):
            if args:
                fixed = list(args)
                fixed[0] = _fix_ui_obj(fixed[0])
                args = tuple(fixed)
            for key in ("label", "placeholder", "help", "value", "caption", "options"):
                if key in kwargs:
                    kwargs[key] = _fix_ui_obj(kwargs[key])
            return __orig(*args, **kwargs)

        setattr(st, name, wrapped_ui_text)


_patch_streamlit_text_rendering()


def _resolve_logo_path() -> Path | None:
    # Priority order:
    # 1) canonical path in assets/
    # 2) common root filenames (useful on GitHub/Streamlit Cloud)
    preferred = Path("assets/abb_logo.png")
    if preferred.exists():
        return preferred

    for candidate in ("ALBARID.png", "albarid.png", "logo.png"):
        p = Path(candidate)
        if p.exists():
            return p

    assets = Path("assets")
    if not assets.exists():
        return None
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files = sorted(assets.glob(ext))
        if files:
            return files[0]
    return None


LOGO_PATH = _resolve_logo_path()

ISIN_MAP = {
    "quotidien": {
        "OCT": {
            "MA0000038960",
            "MA0000040396",
            "MA0000040768",
            "MA0000041717",
            "MA0000037616",
            "MA0000041394",
            "MA0000042152",
            "MA0000037962",
            "MA0000038002",
            "MA0000036261",
            "MA0000038754",
            "MA0000040024",
            "MA0000037715",
            "MA0000037624",
            "MA0000038655",
        },
        "OMLT": {
            "MA0000042186",
            "MA0000041329",
            "MA0000041261",
            "MA0000040214",
            "MA0000038978",
            "MA0000038309",
            "MA0000038267",
            "MA0000038200",
            "MA0000030785",
            "MA0000035917",
            "MA0000036915",
            "MA0000030280",
            "MA0000040016",
            "MA0000042210",
            "MA0000039695",
        },
        "Diversifiés": {
            "MA0000030470",
            "MA0000042202",
            "MA0000040065",
            "MA0000038986",
            "MA0000038358",
            "MA0000038259",
            "MA0000038077",
            "MA0000030520",
            "MA0000036501",
        },
    },
    "hebdomadaire": {
        "OCT": set(),
        "OMLT": {
            "MA0000042079",
            "MA0000041170",
            "MA0000041014",
            "MA0000039190",
            "MA0000037475",
            "MA0000037087",
            "MA0000035099",
        },
        "Diversifiés": {
            "MA0000042087",
            "MA0000042004",
            "MA0000041725",
            "MA0000039554",
            "MA0000038408",
            "MA0000037665",
            "MA0000037640",
            "MA0000036634",
            "MA0000036782",
            "MA0000039398",
        },
    },
}

# Normalize any legacy/mojibake category key to the canonical label.
for _freq in ("quotidien", "hebdomadaire"):
    _cats = ISIN_MAP.get(_freq, {})
    for _k in list(_cats.keys()):
        if "Diversifi" in _k and _k != "Diversifiés":
            _cats["Diversifiés"] = _cats.pop(_k)

if "active_page" not in st.session_state:
    st.session_state.active_page = "OCT"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def _auth_screen() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(900px 520px at 50% 0%, rgba(255,255,255,0.10), rgba(255,255,255,0)),
            radial-gradient(500px 300px at 85% 20%, rgba(242,211,0,0.10), rgba(242,211,0,0)),
            linear-gradient(135deg, #3a2a25 0%, #57423a 48%, #2b211d 100%);
        }
        .auth-shell {
          min-height: 90vh;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .auth-card {
          width: 420px;
          max-width: 100%;
          background: rgba(255,255,255,0.14);
          border: 1px solid rgba(255,255,255,0.28);
          border-radius: 16px;
          padding: 20px 18px 18px;
          color: #fff;
          box-shadow: 0 16px 40px rgba(0,0,0,0.35);
          backdrop-filter: blur(10px);
        }
        .auth-title { font-size: 30px; font-weight: 800; margin-bottom: 4px; color: #ffffff; text-align: center; }
        .auth-sub { color: #f7efea; margin-bottom: 10px; text-align: center; }
        .auth-brand { color: #ffe788; font-weight: 700; margin-bottom: 12px; text-align: center; }
        .auth-logo { display: flex; justify-content: center; margin-bottom: 8px; }
        .auth-logo img { width: 72px; height: 72px; object-fit: contain; }
        </style>
        <div class="auth-shell">
          <div class="auth-card">
            <div class="auth-logo" id="auth-logo-anchor"></div>
            <div class="auth-title">Suivi des OPCVM</div>
            <div class="auth-brand">Al Barid Bank</div>
            <div class="auth-sub">Connexion sécurisée à la plateforme interne</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if LOGO_PATH and LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=72)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="sara")
            password = st.text_input("Code", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
    if submitted:
        if username == APP_USER and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Identifiants invalides")
    st.stop()


if not st.session_state.authenticated:
    _auth_screen()


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        :root {
          --abb-bg: #f2efec;
          --abb-card: #ffffff;
          --abb-yellow: #f2d300;
          --abb-ink: #1f2a37;
          --abb-muted: #5b6674;
          --abb-border: #e5ded8;
          --abb-choco: #5b463f;
        }

        .stApp {
          background:
            radial-gradient(1100px 520px at 80% -20%, rgba(242,211,0,0.22) 0%, rgba(242,211,0,0) 55%),
            radial-gradient(900px 500px at 10% -10%, rgba(91,70,63,0.16) 0%, rgba(91,70,63,0) 60%),
            linear-gradient(180deg, #f5f1ee 0%, var(--abb-bg) 100%);
          color: var(--abb-ink);
          font-family: "Manrope", sans-serif;
        }

        [data-testid="stSidebar"] {
          background:
            radial-gradient(500px 240px at 10% 5%, rgba(255,255,255,0.10), rgba(255,255,255,0)),
            linear-gradient(180deg, rgba(74,56,50,0.96) 0%, rgba(56,42,37,0.96) 100%);
          backdrop-filter: blur(8px);
          border-right: 1px solid rgba(255,255,255,0.12);
        }

        h1, h2, h3 {
          color: var(--abb-ink);
          letter-spacing: -0.02em;
        }

        [data-testid="stMetric"] {
          background: var(--abb-card);
          border: 1px solid var(--abb-border);
          border-radius: 14px;
          padding: 10px 12px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        .abb-banner {
          background: linear-gradient(120deg, rgba(255,255,255,0.96) 0%, rgba(255,247,203,0.95) 100%);
          border: 1px solid var(--abb-border);
          border-left: 6px solid var(--abb-yellow);
          border-radius: 14px;
          padding: 12px 16px;
          margin-bottom: 10px;
        }

        .abb-banner-title {
          font-size: 1.05rem;
          font-weight: 800;
          color: var(--abb-ink);
          margin-bottom: 4px;
        }

        .abb-banner-sub {
          font-size: 0.92rem;
          color: var(--abb-muted);
        }

        .stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
          border-radius: 10px !important;
          border: 1px solid #d8cfc9 !important;
          background: #fffdf7 !important;
          color: #243042 !important;
          font-weight: 700 !important;
        }

        .stDownloadButton > button {
          border-radius: 10px !important;
          background: #fff7cc !important;
          border: 1px solid #e8d77f !important;
          color: #1f2a37 !important;
          font-weight: 700 !important;
        }

        [data-testid="stDataFrame"] {
          border: 1px solid var(--abb-border);
          border-radius: 10px;
          overflow: hidden;
        }

        .side-nav-title {
          font-size: 0.95rem;
          font-weight: 800;
          color: #f2e8e3;
          margin-top: 8px;
          margin-bottom: 6px;
          letter-spacing: 0.02em;
        }

        .side-brand {
          display: flex; align-items: center; gap: 10px;
          padding: 8px 4px 6px;
          border-bottom: 1px dashed rgba(255,255,255,0.25);
          margin-bottom: 8px;
        }
        .side-brand-text { font-weight: 800; color: #fff7e8; }
        .kpi-up { color: #0a8f2e; font-weight: 700; }
        .kpi-down { color: #c00000; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    c1, c2 = st.columns([1, 9])
    with c1:
        if LOGO_PATH and LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=78)
    with c2:
        st.markdown(
            """
            <div class="abb-banner">
              <div class="abb-banner-title">Al Barid Bank</div>
              <div class="abb-banner-sub">Suivi des OPCVM • Plateforme interne de pilotage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


_inject_theme()


def _norm_col(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace("\u00a0", " ")
    # Repair common mojibake sequences before normalization.
    text = (
        text.replace("Ã©", "e")
        .replace("Ã¨", "e")
        .replace("Ãª", "e")
        .replace("Ã«", "e")
        .replace("Ã ", "a")
        .replace("Ã¢", "a")
        .replace("Ã¹", "u")
        .replace("Ã»", "u")
        .replace("Ã´", "o")
        .replace("Ã®", "i")
        .replace("Ã¯", "i")
        .replace("â€™", "'")
        .replace("â€¢", "")
    )
    # Remove accents so matching works with both accented/non-accented headers.
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def _to_num(value: object) -> float | None:
    if value is None:
        return None
    txt = str(value).strip().replace("\u00a0", "")
    if not txt:
        return None
    txt = txt.replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def _format_amount(value: object) -> str:
    num = _to_num(value)
    if num is None:
        return str(value)
    return f"{num:,.2f}"


def _format_percent(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    num = _to_num(value)
    if num is None:
        return raw
    pct = num
    if "%" not in raw:
        if abs(num) <= 1:
            pct = num * 100
    return f"{pct:.2f}%"


def _format_perf_for_kpi(value: object) -> str:
    return _format_percent(value)


def _format_table(df: pd.DataFrame, perf_col: str) -> pd.DataFrame:
    out = df.copy()
    if "AN" in out.columns:
        out["AN"] = out["AN"].map(_format_amount)
    if "VL" in out.columns:
        out["VL"] = out["VL"].map(_format_amount)
    if "YTD" in out.columns:
        out["YTD"] = out["YTD"].map(_format_percent)
    if perf_col in out.columns:
        out[perf_col] = out[perf_col].map(_format_percent)
    return out


def _standardize_asfim_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    for col in out.columns:
        n = _norm_col(col)
        if n == "societe de gestion":
            rename_map[col] = "Société de Gestion"
        elif n == "periodicite vl":
            rename_map[col] = "Périodicité VL"
        elif n == "code isin":
            rename_map[col] = "Code ISIN"
        elif n == "opcvm":
            rename_map[col] = "OPCVM"
        elif n == "souscripteurs" or n == "souscripteur":
            rename_map[col] = "Souscripteurs"
        elif n == "an":
            rename_map[col] = "AN"
        elif n == "vl":
            rename_map[col] = "VL"
        elif n in {"ytd", "yield", "yld"}:
            rename_map[col] = "YTD"
        elif n == "maturite":
            rename_map[col] = "Maturité"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _col_by_norm(df: pd.DataFrame, target: str) -> str | None:
    t = _norm_col(target)
    for c in df.columns:
        if _norm_col(c) == t:
            return c
    return None


def _detect_headers(raw: pd.DataFrame, frequency: str) -> tuple[int, dict[int, str]] | tuple[None, None]:
    aliases = {
        "Code ISIN": {"code isin", "isin"},
        "OPCVM": {"opcvm"},
        "Société de Gestion": {"societe de gestion", "société de gestion"},
        "Périodicité VL": {"periodicite vl", "périodicité vl", "periodicite", "périodicité"},
        "Souscripteurs": {"souscripteurs", "souscripteur"},
        "AN": {"an"},
        "VL": {"vl"},
        "YTD": {"ytd", "yield", "yld"},
        "1 jour": {"1 jour", "1j", "1 journee", "1 journée"},
        "1 semaine": {"1 semaine", "1 sem", "1semaine"},
    }

    perf_required = "1 jour" if frequency == "quotidien" else "1 semaine"

    for r in range(min(30, len(raw))):
        vals = raw.iloc[r].tolist()
        mapped: dict[int, str] = {}
        for i, v in enumerate(vals):
            n = _norm_col(v)
            for canonical, syns in aliases.items():
                if n in syns:
                    mapped[i] = canonical
                    break
        found = set(mapped.values())
        required = {
            "Code ISIN",
            "OPCVM",
            "Société de Gestion",
            "Périodicité VL",
            "Souscripteurs",
            "AN",
            "VL",
            "YTD",
            perf_required,
        }
        if required.issubset(found):
            return r, mapped
    return None, None


@st.cache_data(show_spinner=False)
def parse_asfim_file(path: str, frequency: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    perf_col = "1 jour" if frequency == "quotidien" else "1 semaine"

    for sheet in xls.sheet_names:
        raw = xls.parse(sheet_name=sheet, header=None, dtype=str).fillna("")
        header_row, mapped = _detect_headers(raw, frequency)
        if header_row is None or mapped is None:
            continue

        keep = sorted(mapped.items(), key=lambda x: x[0])
        idxs = [i for i, _ in keep]
        body = raw.iloc[header_row + 1 :].copy()
        body = body.iloc[:, idxs]
        body.columns = [name for _, name in keep]
        body = body.fillna("")
        body = body[body["Code ISIN"].astype(str).str.strip() != ""]

        out = body[
            [
                "Code ISIN",
                "OPCVM",
                "Société de Gestion",
                "Périodicité VL",
                "Souscripteurs",
                "AN",
                "VL",
                "YTD",
                perf_col,
            ]
        ].copy()

        perf_name = "Performance quotidienne" if frequency == "quotidien" else "Performance hebdomadaire"
        out = out.rename(columns={perf_col: perf_name})
        out = _standardize_asfim_columns(out)
        out["performance_num"] = out[perf_name].map(_to_num)
        return out

    return pd.DataFrame(
        columns=[
            "Code ISIN",
            "OPCVM",
            "Société de Gestion",
            "Périodicité VL",
            "Souscripteurs",
            "AN",
            "VL",
            "YTD",
            "Performance quotidienne" if frequency == "quotidien" else "Performance hebdomadaire",
            "performance_num",
        ]
    )


def _latest_file_for_date(frequency: str, date_key: str) -> str | None:
    records = get_asfim_records(frequency=frequency, date_key=date_key)
    for rec in records:
        path = Path(str(rec.get("storage_path", "")))
        if path.exists():
            return str(path)
    return None


def _build_export_excel(df: pd.DataFrame, perf_col: str) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Données", index=False)
        wb = writer.book
        ws = writer.sheets["Données"]

        header_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#C00000", "border": 1})
        green_fmt = wb.add_format({"font_color": "#008000"})
        red_fmt = wb.add_format({"font_color": "#C00000"})

        for c, name in enumerate(df.columns):
            ws.write(0, c, name, header_fmt)
            ws.set_column(c, c, max(12, len(name) + 2))

        if perf_col in df.columns:
            c = df.columns.get_loc(perf_col)
            for r in range(1, len(df) + 1):
                raw = str(df.iloc[r - 1, c])
                num = _to_num(raw)
                fmt = None
                if num is not None:
                    if num > 0:
                        fmt = green_fmt
                    elif num < 0:
                        fmt = red_fmt
                ws.write(r, c, raw, fmt)

    return buffer.getvalue()


def _fund_history(frequency: str, category: str, isin: str) -> pd.DataFrame:
    dates = list_asfim_dates(frequency)
    perf_col = "Performance quotidienne" if frequency == "quotidien" else "Performance hebdomadaire"
    rows: list[dict[str, object]] = []

    for d in sorted(dates):
        path = _latest_file_for_date(frequency, d)
        if not path:
            continue
        df = parse_asfim_file(path, frequency)
        if df.empty:
            continue
        allowed = ISIN_MAP[frequency].get(category, set())
        df = df[df["Code ISIN"].astype(str).str.strip().isin(allowed)]
        item = df[df["Code ISIN"].astype(str).str.strip() == isin]
        if item.empty:
            continue
        val = item.iloc[0][perf_col]
        rows.append({"Date": d, "performance_num": _to_num(val), "Valeur": str(val)})

    return pd.DataFrame(rows)


def _render_category_page(category: str) -> None:
    st.subheader(category)

    freq_ui = st.radio("Type ASFIM", ["Quotidien", "Hebdomadaire"], horizontal=True, key=f"freq_{category}")
    frequency = "quotidien" if freq_ui == "Quotidien" else "hebdomadaire"

    allowed_isins = ISIN_MAP[frequency].get(category, set())
    if not allowed_isins:
        st.warning("Aucun code ISIN configurÃ© pour cette combinaison catÃ©gorie/type.")
        return

    dates = list_asfim_dates(frequency)
    if not dates:
        st.info(f"Aucun fichier ASFIM {frequency} dans l'historique.")
        return

    selected_date = st.selectbox("Date", dates, key=f"date_{category}_{frequency}")
    file_path = _latest_file_for_date(frequency, selected_date)
    if not file_path:
        st.warning("Aucun fichier physique trouvÃ© pour cette date.")
        return

    raw_df = parse_asfim_file(file_path, frequency)
    if raw_df.empty:
        st.error("Impossible de lire les colonnes attendues dans ce fichier ASFIM.")
        return

    perf_col = "Performance quotidienne" if frequency == "quotidien" else "Performance hebdomadaire"

    df = raw_df[raw_df["Code ISIN"].astype(str).str.strip().isin(allowed_isins)].copy()
    if df.empty:
        st.info("Aucune ligne correspondante aux codes ISIN configurÃ©s.")
        return

    df = df.sort_values(by=["performance_num", "OPCVM"], ascending=[True, True], na_position="last")

    best = df.dropna(subset=["performance_num"]).sort_values("performance_num", ascending=False).head(1)
    worst = df.dropna(subset=["performance_num"]).sort_values("performance_num", ascending=True).head(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        if not best.empty:
            st.metric("Plus performant", str(best.iloc[0]["OPCVM"]))
            st.caption(_format_perf_for_kpi(best.iloc[0][perf_col]))
            st.markdown('<div class="kpi-up">â–² tendance positive</div>', unsafe_allow_html=True)
        else:
            st.metric("Plus performant", "N/A")
    with c2:
        if not worst.empty:
            st.metric("Moins performant", str(worst.iloc[0]["OPCVM"]))
            st.caption(_format_perf_for_kpi(worst.iloc[0][perf_col]))
            st.markdown('<div class="kpi-down">â–¼ tendance nÃ©gative</div>', unsafe_allow_html=True)
        else:
            st.metric("Moins performant", "N/A")
    with c3:
        st.metric("DerniÃ¨re date de mise Ã  jour", selected_date)

    soc_col = _col_by_norm(df, "Société de Gestion")
    per_col = _col_by_norm(df, "Périodicité VL")
    display_cols = ["Code ISIN", "OPCVM"]
    if soc_col:
        display_cols.append(soc_col)
    if per_col:
        display_cols.append(per_col)
    display_cols += ["Souscripteurs", "AN", "VL", "YTD", perf_col]
    out = df[display_cols].copy()
    out = _standardize_asfim_columns(out)
    out_display = _format_table(out, perf_col)

    def color_perf(v: object) -> str:
        num = _to_num(v)
        if num is None:
            return ""
        if num > 0:
            return "color: #008000"
        if num < 0:
            return "color: #C00000"
        return ""

    st.dataframe(out_display.style.applymap(color_perf, subset=[perf_col]), use_container_width=True)

    st.download_button(
        "TÃ©lÃ©charger en Excel",
        data=_build_export_excel(out_display, perf_col),
        file_name=f"{category}_{frequency}_{selected_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### DÃ©tail d'un fonds")
    options = [f"{r['OPCVM']} ({r['Code ISIN']})" for _, r in out.iterrows()]
    isin_lookup = {f"{r['OPCVM']} ({r['Code ISIN']})": str(r["Code ISIN"]) for _, r in out.iterrows()}
    selected = st.selectbox("SÃ©lectionner un fonds", options=options, key=f"fund_{category}_{frequency}")
    isin = isin_lookup[selected]

    hist = _fund_history(frequency, category, isin)
    if hist.empty:
        st.info("Historique insuffisant pour ce fonds.")
        return

    today = _format_percent(hist.iloc[-1]["Valeur"])
    prev = _format_percent(hist.iloc[-2]["Valeur"]) if len(hist) > 1 else "N/A"
    d1, d2 = st.columns(2)
    d1.metric("Performance du jour", str(today))
    d2.metric("Performance jour prÃ©cÃ©dent", str(prev))

    chart = hist[["Date", "performance_num"]].dropna().set_index("Date")
    if not chart.empty:
        st.line_chart(chart)


def _mati_pivot_days(date_c1: date) -> int:
    return mati(date_c1, 1)


def _norm_bam_col(v: str) -> str:
    t = str(v).strip().lower().replace("\u00a0", " ")
    t = t.replace("’", "'")
    # Remove accents robustly (é -> e, etc.) for BAM header matching.
    t = "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))
    t = re.sub(r"\s+", " ", t)
    return t


def _parse_dt_any(v: object) -> date | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    for dayfirst in (True, False):
        d = pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
        if pd.notna(d):
            return d.date()
    return None


@st.cache_data(show_spinner=False)
def _parse_bam_curve_file(path: str) -> tuple[pd.DataFrame, str | None]:
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet, dtype=str).fillna("")
        if df.empty:
            continue
        cols = list(df.columns)
        m_col = None
        t_col = None
        v_col = None
        for c in cols:
            n = _norm_bam_col(str(c))
            if "date d'echeance" in n or "date d'cheance" in n or "echeance" in n:
                m_col = c
            if "taux moyen pondere" in n or n == "taux":
                t_col = c
            if "date de la valeur" in n or "date de valeur" in n:
                v_col = c
        if not (m_col and t_col):
            continue
        work = df[[m_col, t_col] + ([v_col] if v_col else [])].copy()
        work.columns = ["DateEcheance", "Taux"] + (["DateValeur"] if v_col else [])
        work = work[work["DateEcheance"].astype(str).str.strip() != ""]
        if work.empty:
            continue
        if "DateValeur" in work.columns:
            vals = work["DateValeur"].map(_parse_dt_any).dropna()
            if vals.empty:
                continue
            mode_vals = vals.mode()
            if mode_vals.empty:
                continue
            date_valeur = mode_vals.iloc[0]
        else:
            continue
        work["DateEcheance_dt"] = work["DateEcheance"].map(_parse_dt_any)
        work["maturity_days"] = work["DateEcheance_dt"].map(
            lambda d: (d - date_valeur).days if d is not None else None
        )
        work["rate_num"] = work["Taux"].map(_to_num)
        work = work.dropna(subset=["maturity_days", "rate_num"])
        work = work[work["maturity_days"] > 0]
        if work.empty:
            continue
        # Convert percent-like values to decimal rates for interpolation.
        work["rate_dec"] = work["rate_num"].map(lambda x: x / 100.0 if x > 1 else x)
        return work[["maturity_days", "rate_dec"]].sort_values("maturity_days"), date_valeur.strftime("%Y-%m-%d")
    return pd.DataFrame(columns=["maturity_days", "rate_dec"]), None


TARGET_MATS = [
    ("13 s", 13 * 7),
    ("26 s", 26 * 7),
    ("52 s", 52 * 7),
    ("2 ans", 2 * 365),
    ("5 ans", 5 * 365),
    ("10 ans", 10 * 365),
    ("15 ans", 15 * 365),
    ("20 ans", 20 * 365),
    ("30 ans", 30 * 365),
]


def _latest_bam_file_for_date(date_key: str) -> str | None:
    records = get_bam_records(date_key=date_key)
    for rec in records:
        p = Path(str(rec.get("storage_path", "")))
        if p.exists():
            return str(p)
    return None


@st.cache_data(show_spinner=False)
def _build_bam_curve_points(date_key: str) -> dict[str, float] | None:
    path = _latest_bam_file_for_date(date_key)
    if not path:
        return None
    curve, dstr = _parse_bam_curve_file(path)
    if curve.empty or not dstr:
        return None
    mt = [int(v) for v in curve["maturity_days"].tolist()]
    tx = [float(v) for v in curve["rate_dec"].tolist()]
    date_c1 = datetime.strptime(dstr, "%Y-%m-%d").date()
    pivot = _mati_pivot_days(date_c1)
    out: dict[str, float] = {}
    for label, days in TARGET_MATS:
        out[label] = calcul_taux(days, mt, tx, date_c1, pivot)
    return out


def _build_bam_compare_export(selected_j: str, selected_j1: str) -> bytes | None:
    j_curve = _build_bam_curve_points(selected_j)
    j1_curve = _build_bam_curve_points(selected_j1)
    if not j_curve or not j1_curve:
        return None

    cols = [label for label, _ in TARGET_MATS]
    j_vals = [j_curve.get(c) for c in cols]
    j1_vals = [j1_curve.get(c) for c in cols]
    var_vals = [(a - b) if a is not None and b is not None else None for a, b in zip(j_vals, j1_vals)]

    def fmt_pct(x: object) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x) * 100:.3f}%"

    df = pd.DataFrame(
        [j_vals, j1_vals, var_vals],
        columns=cols,
        index=[selected_j, selected_j1, "VAR"],
    ).reset_index()
    df = df.rename(columns={"index": "Maturité"})
    for c in cols:
        df[c] = df[c].map(fmt_pct)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Courbe BAM", index=False)
    return buffer.getvalue()


def _find_previous_valid_bam_date(selected_j: str, bam_dates: list[str]) -> str | None:
    """Return the closest previous BAM date that can be parsed into curve points."""
    if selected_j not in bam_dates:
        return None
    j_index = bam_dates.index(selected_j)
    for d in bam_dates[j_index + 1 :]:
        if _build_bam_curve_points(d):
            return d
    return None


def _curve_reco_comment_from_history(
    dates: list[str],
    start_index: int,
    cols: list[str],
    window: int = 5,
) -> tuple[str, str]:
    """Build recommendation/comment based on historical BAM transitions."""
    if len(dates) < 2 or start_index >= len(dates) - 1:
        return "Données", "Historique BAM insuffisant pour générer un commentaire."

    short_labels = ["13 s", "26 s", "52 s"]
    long_labels = ["10 ans", "15 ans", "20 ans", "30 ans"]
    avg_vars: list[float] = []
    slope_deltas: list[float] = []

    end_index = min(len(dates) - 1, start_index + max(1, window))
    for i in range(start_index, end_index):
        d_j = dates[i]
        d_j1 = dates[i + 1]
        c_j = _build_bam_curve_points(d_j)
        c_j1 = _build_bam_curve_points(d_j1)
        if not c_j or not c_j1:
            continue

        pair_vars = []
        for k in cols:
            a = c_j.get(k)
            b = c_j1.get(k)
            if a is not None and b is not None:
                pair_vars.append(a - b)
        if not pair_vars:
            continue
        avg_vars.append(sum(pair_vars) / len(pair_vars))

        short_vals = [c_j.get(k) - c_j1.get(k) for k in short_labels if c_j.get(k) is not None and c_j1.get(k) is not None]
        long_vals = [c_j.get(k) - c_j1.get(k) for k in long_labels if c_j.get(k) is not None and c_j1.get(k) is not None]
        if short_vals and long_vals:
            slope_deltas.append((sum(long_vals) / len(long_vals)) - (sum(short_vals) / len(short_vals)))

    if not avg_vars:
        return "Données", "Impossible de calculer la tendance historique."

    mean_hist = sum(avg_vars) / len(avg_vars)
    up_count = sum(1 for v in avg_vars if v > 0)
    down_count = sum(1 for v in avg_vars if v < 0)

    if mean_hist > 0:
        reco = f"Historique: pression haussière des taux ({up_count}/{len(avg_vars)} séances), posture prudente."
    elif mean_hist < 0:
        reco = f"Historique: détente des taux ({down_count}/{len(avg_vars)} séances), posture plus constructive."
    else:
        reco = "Historique: stabilité globale des taux."

    if slope_deltas:
        slope_mean = sum(slope_deltas) / len(slope_deltas)
        if slope_mean > 0:
            com = "Historique: pentification dominante (long terme évolue plus que court terme)."
        elif slope_mean < 0:
            com = "Historique: aplatissement dominant (court terme évolue plus que long terme)."
        else:
            com = "Historique: pente globalement stable."
    else:
        com = "Historique: données insuffisantes pour qualifier la pente."

    return reco, com


def _asfim_daily_fund_timeseries() -> tuple[dict[str, list[tuple[str, float]]], dict[str, str]]:
    dates = list_asfim_dates("quotidien")
    data: dict[str, list[tuple[str, float]]] = {}
    names: dict[str, str] = {}
    if not dates:
        return data, names
    allowed_all = set().union(*ISIN_MAP["quotidien"].values())
    for d in dates:
        path = _latest_file_for_date("quotidien", d)
        if not path:
            continue
        df = parse_asfim_file(path, "quotidien")
        if df.empty or "Performance quotidienne" not in df.columns:
            continue
        df = df[df["Code ISIN"].astype(str).str.strip().isin(allowed_all)]
        for _, r in df.iterrows():
            isin = str(r["Code ISIN"]).strip()
            perf = _to_num(r["Performance quotidienne"])
            if perf is None:
                continue
            names[isin] = str(r["OPCVM"])
            data.setdefault(isin, []).append((d, perf))
    return data, names


def _correlation_insights(curve_metric_by_date: dict[str, float]) -> tuple[str | None, float | None, str | None, float | None]:
    if not curve_metric_by_date:
        return None, None, None, None
    fund_ts, names = _asfim_daily_fund_timeseries()
    if not fund_ts:
        return None, None, None, None

    s_curve = pd.Series(curve_metric_by_date, name="curve")
    corr_scores: list[tuple[str, float]] = []
    for isin, points in fund_ts.items():
        s_fund = pd.Series({d: v for d, v in points}, name="fund")
        merged = pd.concat([s_fund, s_curve], axis=1, join="inner").dropna()
        if len(merged) < 3:
            continue
        corr = merged["fund"].corr(merged["curve"])
        if pd.notna(corr):
            corr_scores.append((f"{names.get(isin, isin)} ({isin})", float(corr)))
    if not corr_scores:
        return None, None, None, None

    most = max(corr_scores, key=lambda x: abs(x[1]))
    least = min(corr_scores, key=lambda x: abs(x[1]))
    return most[0], most[1] * 100.0, least[0], least[1] * 100.0


def _render_curve_page() -> None:
    st.subheader("Suivi de la courbe")
    dates = list_bam_dates()
    if len(dates) < 2:
        st.info("Il faut au moins 2 dates BAM pour comparer J et J-1.")
        return

    selected_j = st.selectbox("Date J", dates, index=0)
    j_index = dates.index(selected_j)
    if j_index + 1 >= len(dates):
        st.warning("Choisir une date J qui a une date précédente J-1.")
        return
    selected_j1 = dates[j_index + 1]

    j_curve = _build_bam_curve_points(selected_j)
    j1_curve = _build_bam_curve_points(selected_j1)
    if not j_curve or not j1_curve:
        st.error("Impossible de construire les courbes interpolées pour J/J-1.")
        return

    cols = [label for label, _ in TARGET_MATS]
    j_vals = [j_curve.get(c) for c in cols]
    j1_vals = [j1_curve.get(c) for c in cols]
    var_vals = [(a - b) if a is not None and b is not None else None for a, b in zip(j_vals, j1_vals)]

    table = pd.DataFrame([j_vals, j1_vals, var_vals], columns=cols, index=[selected_j, selected_j1, "VAR"]).reset_index()
    table = table.rename(columns={"index": "Maturité"})

    def fmt_pct(x: object) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x) * 100:.3f}%"

    display = table.copy()
    for c in cols:
        display[c] = display[c].map(fmt_pct)

    def style_var(v: object) -> str:
        n = _to_num(v)
        if n is None:
            return ""
        if n < 0:
            return "color: #C00000; font-weight: bold;"
        if n > 0:
            return "color: #008000; font-weight: bold;"
        return ""

    def _build_curve_compare_excel(df_display: pd.DataFrame, maturity_cols: list[str]) -> bytes:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_display.to_excel(writer, sheet_name="Comparaison", index=False)
            wb = writer.book
            ws = writer.sheets["Comparaison"]

            yellow_header = wb.add_format(
                {"bold": True, "bg_color": "#FFD966", "font_color": "#000000", "border": 1, "align": "center"}
            )
            yellow_first_col = wb.add_format({"bold": True, "bg_color": "#FFD966", "border": 1})
            default_cell = wb.add_format({"border": 1, "align": "center"})
            green_var = wb.add_format({"font_color": "#008000", "bold": True, "border": 1, "align": "center"})
            red_var = wb.add_format({"font_color": "#C00000", "bold": True, "border": 1, "align": "center"})

            # Header row (maturities + first column title) in yellow.
            for c, name in enumerate(df_display.columns):
                ws.write(0, c, name, yellow_header)
                ws.set_column(c, c, max(12, len(name) + 2))

            # Data rows.
            for r in range(1, len(df_display) + 1):
                first_col_val = str(df_display.iloc[r - 1, 0])
                ws.write(r, 0, first_col_val, yellow_first_col)
                for c in range(1, len(df_display.columns)):
                    raw = str(df_display.iloc[r - 1, c])
                    fmt = default_cell
                    if first_col_val == "VAR":
                        n = _to_num(raw)
                        if n is not None:
                            fmt = green_var if n > 0 else red_var
                    ws.write(r, c, raw, fmt)

        return buffer.getvalue()

    table_col, btn_col = st.columns([5, 1])
    with table_col:
        st.dataframe(display.style.applymap(style_var, subset=cols), use_container_width=True)
    with btn_col:
        st.download_button(
            "Télécharger\nExcel",
            data=_build_curve_compare_excel(display, cols),
            file_name=f"Courbe_J_vs_J-1_{selected_j}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    reco, com = _curve_reco_comment_from_history(dates, j_index, cols)
    st.markdown(f"**Recommandations:** {reco}")
    st.markdown(f"**Commentaires:** {com}")

    curve_metric_by_date: dict[str, float] = {}
    for d in dates:
        curve = _build_bam_curve_points(d)
        if not curve:
            continue
        vals = [curve.get(k) for k in cols if curve.get(k) is not None]
        if vals:
            curve_metric_by_date[d] = sum(vals) / len(vals)
    most_name, most_corr, least_name, least_corr = _correlation_insights(curve_metric_by_date)
    st.markdown("### Corrélation à la courbe BAM")
    if most_name is None:
        st.info("Données insuffisantes pour calculer les corrélations.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Plus corrélé", most_name)
        c1.caption(f"Corrélation: {most_corr:.2f}%")
        c2.metric("Moins corrélé", least_name or "N/A")
        if least_corr is not None:
            c2.caption(f"Corrélation: {least_corr:.2f}%")


def _category_from_isin(frequency: str, isin: str) -> str | None:
    for cat, values in ISIN_MAP[frequency].items():
        if isin in values:
            return cat
    return None


def _latest_universe_df() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frequency in ["quotidien", "hebdomadaire"]:
        dates = list_asfim_dates(frequency)
        if not dates:
            continue
        path = _latest_file_for_date(frequency, dates[0])
        if not path:
            continue
        df = parse_asfim_file(path, frequency)
        if df.empty:
            continue
        perf_col = "Performance quotidienne" if frequency == "quotidien" else "Performance hebdomadaire"
        allowed = set().union(*ISIN_MAP[frequency].values())
        df = df[df["Code ISIN"].astype(str).str.strip().isin(allowed)].copy()
        if df.empty:
            continue
        df["Frequency"] = frequency
        df["Date"] = dates[0]
        df["Category"] = df["Code ISIN"].astype(str).str.strip().map(lambda x: _category_from_isin(frequency, x))
        df["PerfLabel"] = perf_col
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


with st.sidebar:
    if LOGO_PATH and LOGO_PATH.exists():
        st.markdown('<div class="side-brand"><img src="data:image/png;base64,{}" width="34"/><div class="side-brand-text">Al Barid Bank</div></div>'.format(base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")), unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-brand"><div class="side-brand-text">Al Barid Bank</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav-title">Navigation</div>', unsafe_allow_html=True)
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
    pages = ["OCT", "OMLT", "Diversifiés", "Suivi de la courbe", "Analyse", "Export"]
    st.markdown('<div class="side-nav-title">Pages</div>', unsafe_allow_html=True)
    for p in pages:
        label = f"▸ {p}" if p == st.session_state.active_page else p
        if st.button(label, key=f"nav_{p}", use_container_width=True):
            st.session_state.active_page = p
            st.rerun()
    page = st.session_state.active_page


_render_brand_header()

if page == "OCT":
    _render_category_page("OCT")
elif page == "OMLT":
    _render_category_page("OMLT")
elif page == "Diversifiés":
    _render_category_page("Diversifiés")
elif page == "Suivi de la courbe":
    _render_curve_page()
elif page == "Analyse":
    st.subheader("Analyse")
    universe = _latest_universe_df()
    if universe.empty:
        st.info("Aucune donnÃ©e ASFIM disponible pour l'analyse.")
    else:
        work = universe.dropna(subset=["performance_num"]).copy()
        if work.empty:
            st.info("DonnÃ©es de performance indisponibles.")
        else:
            best = work.loc[work["performance_num"].idxmax()]
            worst = work.loc[work["performance_num"].idxmin()]
            # DÃ©fensif: performance la plus proche de 0 (faible amplitude)
            defensive = work.assign(abs_perf=work["performance_num"].abs()).sort_values("abs_perf").iloc[0]
            # Offensif: performance la plus Ã©levÃ©e
            offensive = best

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Plus performant", f"{best['OPCVM']} ({best['Code ISIN']})")
            c1.caption(f"{best['Category']} | {best['Frequency']} | Perf: {_format_percent(best[best['PerfLabel']])}")
            c1.markdown('<div class="kpi-up">â–² performance forte</div>', unsafe_allow_html=True)
            c2.metric("Moins performant", f"{worst['OPCVM']} ({worst['Code ISIN']})")
            c2.caption(f"{worst['Category']} | {worst['Frequency']} | Perf: {_format_percent(worst[worst['PerfLabel']])}")
            c2.markdown('<div class="kpi-down">â–¼ performance faible</div>', unsafe_allow_html=True)
            c3.metric("Fonds offensif", f"{offensive['OPCVM']} ({offensive['Code ISIN']})")
            c3.caption(f"{offensive['Category']} | {offensive['Frequency']} | Perf: {_format_percent(offensive[offensive['PerfLabel']])}")
            c4.metric("Fonds dÃ©fensif", f"{defensive['OPCVM']} ({defensive['Code ISIN']})")
            c4.caption(f"{defensive['Category']} | {defensive['Frequency']} | Perf: {_format_percent(defensive[defensive['PerfLabel']])}")

            st.markdown("### DÃ©tails analyse")
            show_cols = [
                "Code ISIN",
                "OPCVM",
                "Société de Gestion",
                "Category",
                "Frequency",
                "Date",
                "AN",
                "VL",
                "YTD",
            ]
            # Afficher la colonne perf selon frÃ©quence dans un format unifiÃ©.
            details = work.copy()
            details["Performance"] = details.apply(lambda r: r[r["PerfLabel"]], axis=1)
            show_cols.append("Performance")
            details_display = _format_table(details[show_cols], "Performance")
            st.dataframe(details_display, use_container_width=True)
elif page == "Export":
    st.subheader("Export")
    c_refresh, _ = st.columns([1, 5])
    with c_refresh:
        if st.button("Mise à jour", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("### Section ASFIM - Upload Historique")
    frequency_ui = st.radio(
        "Type de fichier",
        ["ASFIM Quotidien", "ASFIM Hebdomadaire"],
        horizontal=True,
    )
    frequency = "quotidien" if frequency_ui == "ASFIM Quotidien" else "hebdomadaire"

    uploaded_files = st.file_uploader(
        "Uploader des fichiers ASFIM (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="asfim_multi_upload",
    )

    batch_date_key = st.text_input(
        "Date du lot (optionnel, fallback si la date n'est pas détectée)",
        placeholder="ex: 2026-02-11",
    )

    if st.button("Enregistrer dans l’historique", use_container_width=True):
        if not uploaded_files:
            st.warning("Aucun fichier uploadé.")
        else:
            result = add_asfim_files(uploaded_files, frequency=frequency, batch_date_key=batch_date_key or None)
            saved_count = len(result["saved"])
            error_count = len(result["errors"])

            if saved_count:
                st.success(f"{saved_count} fichier(s) ASFIM enregistré(s).")
                for item in result["saved"]:
                    st.caption(
                        f"- {item['filename']} | date={item['date_key']} | type={item['frequency']} | source_date={item['date_source']}"
                    )

            if error_count:
                st.error(f"{error_count} fichier(s) non enregistrés.")
                for err in result["errors"]:
                    st.caption(f"- {err['filename']} : {err['error']}")

    with st.expander("Archive historique ASFIM", expanded=False):
        st.markdown("### Historique ASFIM")
        summary_rows = summarize_asfim_history()
        if not summary_rows:
            st.info("Aucun historique ASFIM enregistré.")
        else:
            for row in summary_rows:
                st.write(f"- **{row['Type']}** | Date: `{row['Date']}` | Fichiers: {row['Nombre de fichiers']}")

        st.markdown("### Détail des fichiers stockés")
        detail_frequency = st.selectbox("Type", ["quotidien", "hebdomadaire"], index=0)
        available_dates = list_asfim_dates(detail_frequency)

        if not available_dates:
            st.info("Aucune date disponible pour ce type.")
        else:
            detail_date = st.selectbox("Date", available_dates)
            if st.button("Voir fichiers stockés", use_container_width=True):
                files = list_asfim_files(detail_frequency, detail_date)
                if not files:
                    st.info("Aucun fichier pour cette date.")
                else:
                    for f in files:
                        st.write(
                            f"- `{f['filename']}` | original: `{f['original_filename']}` | path: `{f['storage_path']}` | upload: {f['uploaded_at']}"
                        )

    st.markdown("### Section BAM - Upload Historique")
    bam_uploaded_files = st.file_uploader(
        "Uploader des fichiers BAM (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="bam_multi_upload",
    )
    bam_batch_date_key = st.text_input(
        "Date du lot BAM (optionnel, fallback si la date n'est pas détectée)",
        placeholder="ex: 2026-02-12",
        key="bam_batch_date",
    )

    if st.button("Enregistrer historique BAM", use_container_width=True):
        if not bam_uploaded_files:
            st.warning("Aucun fichier BAM uploadé.")
        else:
            result = add_bam_files(bam_uploaded_files, batch_date_key=bam_batch_date_key or None)
            saved_count = len(result["saved"])
            error_count = len(result["errors"])

            if saved_count:
                st.success(f"{saved_count} fichier(s) BAM enregistré(s).")
            if error_count:
                st.error(f"{error_count} fichier(s) BAM non enregistrés.")

    with st.expander("Archive historique BAM", expanded=False):
        bam_summary = summarize_bam_history()
        if not bam_summary:
            st.info("Aucun historique BAM enregistré.")
        else:
            for row in bam_summary:
                st.write(f"- Date: `{row['Date']}` | Fichiers: {row['Nombre de fichiers']}")

        bam_dates_detail = list_bam_dates()
        if bam_dates_detail:
            bam_date = st.selectbox("Date BAM", bam_dates_detail, key="bam_detail_date")
            if st.button("Voir fichiers BAM stock?s", use_container_width=True):
                files = list_bam_files(bam_date)
                if not files:
                    st.info("Aucun fichier BAM pour cette date.")
                else:
                    for f in files:
                        st.write(f"- `{f['filename']}` | path: `{f['storage_path']}` | upload: {f['uploaded_at']}")


    st.markdown("### Export courbe BAM")
    bam_dates = list_bam_dates()
    if not bam_dates:
        st.info("Aucun historique BAM disponible.")
    elif len(bam_dates) < 2:
        st.info("Il faut au moins 2 dates BAM pour exporter J vs J-1.")
    else:
        last_asfim_q = list_asfim_dates("quotidien")
        last_asfim_h = list_asfim_dates("hebdomadaire")
        c1, c2, c3 = st.columns(3)
        c1.metric("Derni?re date BAM", bam_dates[0])
        c2.metric("Derni?re date ASFIM quotidien", last_asfim_q[0] if last_asfim_q else "N/A")
        c3.metric("Derni?re date ASFIM hebdomadaire", last_asfim_h[0] if last_asfim_h else "N/A")

        selected_j = st.selectbox("Date BAM J (export)", bam_dates, key="exp_bam_j")
        if not _build_bam_curve_points(selected_j):
            st.warning(
                "La date J s?lectionn?e existe dans l'archive, mais son fichier BAM n'est pas lisible "
                "(colonnes/date de valeur). Choisis une autre date J."
            )
        else:
            selected_j1 = _find_previous_valid_bam_date(selected_j, bam_dates)
            if not selected_j1:
                st.warning("Aucune date J-1 valide trouv?e dans l'historique BAM pour cette date J.")
            else:
                st.caption(f"J-1 utilis? automatiquement: {selected_j1}")
                data = _build_bam_compare_export(selected_j, selected_j1)
                if data is None:
                    st.warning("Impossible de construire l'export BAM pour ces dates.")
                else:
                    st.download_button(
                        "T?l?charger courbe BAM (J vs J-1)",
                        data=data,
                        file_name=f"Courbe_BAM_{selected_j}_vs_{selected_j1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
    st.caption("Historique cumulatif: chaque nouveau fichier est ajout?, les anciens sont conserv?s apr?s red?marrage.")

