"""Real Exchange Rate (RER) explorer built on live IMF data.

Layout of this file (top to bottom):
    constants -> country table -> fetch (IMF API) -> fallback (backup CSV)
    -> compute (S, RER, Real Index) -> signal (valuation zones)
    -> plot (Plotly) -> explain (plain-language text) -> UI (Streamlit)
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
IMF_BASE = "https://api.imf.org/external/sdmx/3.0"
CPI_FLOW = f"{IMF_BASE}/data/dataflow/IMF.STA/CPI/~"   # "~" = latest version
ER_FLOW = f"{IMF_BASE}/data/dataflow/IMF.STA/ER/~"
CODELIST_URL = f"{IMF_BASE}/structure/codelist/IMF/CL_COUNTRY/+"
HTTP_TIMEOUT = 25            # seconds per call
CACHE_TTL = 6 * 3600         # IMF data refreshed at most every 6 hours
API_COOLDOWN = 90            # after a failure, skip live calls for this many seconds
START_PERIOD = pd.Period("1976-01", "M")
BACKUP_PATH = Path(__file__).resolve().parent / "imf_backup.csv"
BACKUP_ISOS = ["USA", "G163", "JPN", "GBR", "CHN", "CHE", "AUS", "CAN", "HKG", "SGP", "MEX"]
EURO_AREA = "G163"
DEFAULT_C1, DEFAULT_C2 = "GBR", "USA"

# Valuation zones: (upper bound of z, label, key). Lower bound is the previous bound.
Z_VERY_CHEAP, Z_CHEAP, Z_EXPENSIVE, Z_VERY_EXPENSIVE = -1.5, -0.5, 0.5, 1.5
ZONES = [
    (Z_VERY_CHEAP, "🟢🟢 VERY CHEAP", "very_cheap"),
    (Z_CHEAP, "🟢 CHEAP", "cheap"),
    (Z_EXPENSIVE, "🟡 FAIR", "fair"),
    (Z_VERY_EXPENSIVE, "🔴 EXPENSIVE", "expensive"),
    (math.inf, "🔴🔴 VERY EXPENSIVE", "very_expensive"),
]
ZONE_COLORS = {"very_cheap": "#1b8a3b", "cheap": "#4caf50", "fair": "#e0a800",
               "expensive": "#e5533d", "very_expensive": "#b3261e"}

# ISO3 | ISO2 (for the flag) | short name | currency code
_COUNTRY_TABLE = """
G163|EU|Euro area|EUR
AFG|AF|Afghanistan|AFN
ALB|AL|Albania|ALL
DZA|DZ|Algeria|DZD
AGO|AO|Angola|AOA
ATG|AG|Antigua and Barbuda|XCD
ARG|AR|Argentina|ARS
ARM|AM|Armenia|AMD
ABW|AW|Aruba|AWG
AUS|AU|Australia|AUD
AUT|AT|Austria|ATS
AZE|AZ|Azerbaijan|AZN
BHS|BS|Bahamas|BSD
BHR|BH|Bahrain|BHD
BGD|BD|Bangladesh|BDT
BRB|BB|Barbados|BBD
BLR|BY|Belarus|BYN
BEL|BE|Belgium|BEF
BLZ|BZ|Belize|BZD
BEN|BJ|Benin|XOF
BTN|BT|Bhutan|BTN
BOL|BO|Bolivia|BOB
BIH|BA|Bosnia and Herzegovina|BAM
BWA|BW|Botswana|BWP
BRA|BR|Brazil|BRL
BRN|BN|Brunei|BND
BGR|BG|Bulgaria|BGN
BFA|BF|Burkina Faso|XOF
BDI|BI|Burundi|BIF
CPV|CV|Cabo Verde|CVE
KHM|KH|Cambodia|KHR
CMR|CM|Cameroon|XAF
CAN|CA|Canada|CAD
CAF|CF|Central African Republic|XAF
TCD|TD|Chad|XAF
CHL|CL|Chile|CLP
CHN|CN|China|CNY
COL|CO|Colombia|COP
COM|KM|Comoros|KMF
COD|CD|Congo (DR)|CDF
COG|CG|Congo (Rep.)|XAF
CRI|CR|Costa Rica|CRC
CIV|CI|Côte d'Ivoire|XOF
HRV|HR|Croatia|HRK
CYP|CY|Cyprus|CYP
CZE|CZ|Czechia|CZK
DNK|DK|Denmark|DKK
DJI|DJ|Djibouti|DJF
DMA|DM|Dominica|XCD
DOM|DO|Dominican Republic|DOP
ECU|EC|Ecuador|USD
EGY|EG|Egypt|EGP
SLV|SV|El Salvador|USD
GNQ|GQ|Equatorial Guinea|XAF
ERI|ER|Eritrea|ERN
EST|EE|Estonia|EEK
SWZ|SZ|Eswatini|SZL
ETH|ET|Ethiopia|ETB
FJI|FJ|Fiji|FJD
FIN|FI|Finland|FIM
FRA|FR|France|FRF
GAB|GA|Gabon|XAF
GMB|GM|Gambia|GMD
GEO|GE|Georgia|GEL
DEU|DE|Germany|DEM
GHA|GH|Ghana|GHS
GRC|GR|Greece|GRD
GRD|GD|Grenada|XCD
GTM|GT|Guatemala|GTQ
GIN|GN|Guinea|GNF
GNB|GW|Guinea-Bissau|XOF
GUY|GY|Guyana|GYD
HTI|HT|Haiti|HTG
HND|HN|Honduras|HNL
HKG|HK|Hong Kong|HKD
HUN|HU|Hungary|HUF
ISL|IS|Iceland|ISK
IND|IN|India|INR
IDN|ID|Indonesia|IDR
IRN|IR|Iran|IRR
IRQ|IQ|Iraq|IQD
IRL|IE|Ireland|IEP
ISR|IL|Israel|ILS
ITA|IT|Italy|ITL
JAM|JM|Jamaica|JMD
JPN|JP|Japan|JPY
JOR|JO|Jordan|JOD
KAZ|KZ|Kazakhstan|KZT
KEN|KE|Kenya|KES
KIR|KI|Kiribati|AUD
KOR|KR|South Korea|KRW
KWT|KW|Kuwait|KWD
KGZ|KG|Kyrgyz Republic|KGS
LAO|LA|Laos|LAK
LVA|LV|Latvia|LVL
LBN|LB|Lebanon|LBP
LSO|LS|Lesotho|LSL
LBR|LR|Liberia|LRD
LBY|LY|Libya|LYD
LTU|LT|Lithuania|LTL
LUX|LU|Luxembourg|LUF
MAC|MO|Macao|MOP
MDG|MG|Madagascar|MGA
MWI|MW|Malawi|MWK
MYS|MY|Malaysia|MYR
MDV|MV|Maldives|MVR
MLI|ML|Mali|XOF
MLT|MT|Malta|MTL
MRT|MR|Mauritania|MRU
MUS|MU|Mauritius|MUR
MEX|MX|Mexico|MXN
MDA|MD|Moldova|MDL
MNG|MN|Mongolia|MNT
MNE|ME|Montenegro|EUR
MAR|MA|Morocco|MAD
MOZ|MZ|Mozambique|MZN
MMR|MM|Myanmar|MMK
NAM|NA|Namibia|NAD
NPL|NP|Nepal|NPR
NLD|NL|Netherlands|NLG
NZL|NZ|New Zealand|NZD
NIC|NI|Nicaragua|NIO
NER|NE|Niger|XOF
NGA|NG|Nigeria|NGN
MKD|MK|North Macedonia|MKD
NOR|NO|Norway|NOK
OMN|OM|Oman|OMR
PAK|PK|Pakistan|PKR
PAN|PA|Panama|PAB
PNG|PG|Papua New Guinea|PGK
PRY|PY|Paraguay|PYG
PER|PE|Peru|PEN
PHL|PH|Philippines|PHP
POL|PL|Poland|PLN
PRT|PT|Portugal|PTE
QAT|QA|Qatar|QAR
ROU|RO|Romania|RON
RUS|RU|Russia|RUB
RWA|RW|Rwanda|RWF
WSM|WS|Samoa|WST
STP|ST|São Tomé and Príncipe|STN
SAU|SA|Saudi Arabia|SAR
SEN|SN|Senegal|XOF
SRB|RS|Serbia|RSD
SYC|SC|Seychelles|SCR
SLE|SL|Sierra Leone|SLE
SGP|SG|Singapore|SGD
SVK|SK|Slovak Republic|SKK
SVN|SI|Slovenia|SIT
SLB|SB|Solomon Islands|SBD
ZAF|ZA|South Africa|ZAR
ESP|ES|Spain|ESP
LKA|LK|Sri Lanka|LKR
SDN|SD|Sudan|SDG
SUR|SR|Suriname|SRD
SWE|SE|Sweden|SEK
CHE|CH|Switzerland|CHF
SYR|SY|Syria|SYP
TJK|TJ|Tajikistan|TJS
TZA|TZ|Tanzania|TZS
THA|TH|Thailand|THB
TLS|TL|Timor-Leste|USD
TGO|TG|Togo|XOF
TON|TO|Tonga|TOP
TTO|TT|Trinidad and Tobago|TTD
TUN|TN|Tunisia|TND
TUR|TR|Türkiye|TRY
TKM|TM|Turkmenistan|TMT
UGA|UG|Uganda|UGX
UKR|UA|Ukraine|UAH
ARE|AE|United Arab Emirates|AED
GBR|GB|United Kingdom|GBP
USA|US|United States|USD
URY|UY|Uruguay|UYU
UZB|UZ|Uzbekistan|UZS
VUT|VU|Vanuatu|VUV
VEN|VE|Venezuela|VES
VNM|VN|Vietnam|VND
YEM|YE|Yemen|YER
ZMB|ZM|Zambia|ZMW
ZWE|ZW|Zimbabwe|ZWL
"""
# Euro members: the IMF exchange-rate series stops when they adopted the euro, so they carry their
# legacy currency code (DEM, FRF...). Use the Euro area (G163) for anything after 1998.
EURO_MEMBERS = {"AUT", "BEL", "CYP", "DEU", "ESP", "EST", "FIN", "FRA", "GRC", "HRV", "IRL", "ITA", "LTU", "LUX",
                "LVA", "MLT", "NLD", "PRT", "SVK", "SVN"}
COUNTRY_INFO: dict[str, tuple[str, str, str]] = {}
for _line in _COUNTRY_TABLE.strip().splitlines():
    _iso3, _iso2, _name, _ccy = _line.split("|")
    COUNTRY_INFO[_iso3] = (_iso2, _name, _ccy)


class DataProblem(Exception):
    """A data/calculation problem that we can explain to the user in plain words."""


# ----------------------------------------------------------------------------
# COUNTRY LABELS
# ----------------------------------------------------------------------------
def flag(iso: str) -> str:
    """Flag emoji from the ISO2 code (🏳️ if unknown)."""
    if iso not in COUNTRY_INFO:
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in COUNTRY_INFO[iso][0])


def cname(iso: str, imf_names: dict | None = None) -> str:
    if iso in COUNTRY_INFO:
        return COUNTRY_INFO[iso][1]
    return (imf_names or {}).get(iso, iso)


def ccy(iso: str) -> str:
    return COUNTRY_INFO[iso][2] if iso in COUNTRY_INFO else "LCU"  # LCU = local currency unit


def label(iso: str, imf_names: dict | None = None, star: bool = False) -> str:
    """'🇬🇧 United Kingdom (GBP)' - used everywhere a country is mentioned."""
    return f"{'⭐ ' if star else ''}{flag(iso)} {cname(iso, imf_names)} ({ccy(iso)})"


# ----------------------------------------------------------------------------
# FETCH: IMF SDMX 3.0 API
# ----------------------------------------------------------------------------
@st.cache_resource
def _shared_api_state() -> dict:
    """One dict shared by all sessions/reruns: avoids waiting on a dead API again and again."""
    return {"down_until": 0.0}


_api_state = _shared_api_state()


def _http_json(url: str, params: dict | None = None) -> dict:
    """GET + JSON with a timeout and ONE retry. Raises on any failure."""
    last: Exception | None = None
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, headers={"Accept": "application/json"},
                             timeout=HTTP_TIMEOUT)
            if r.status_code == 204:
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # network error, bad status, bad JSON
            last = exc
            if attempt == 0:
                time.sleep(0.5)
    raise RuntimeError(f"IMF request failed: {last}")


def _parse_sdmx(payload: dict) -> dict[str, dict[pd.Period, float]]:
    """SDMX-JSON 2.0 -> {INDEX_TYPE_or_INDICATOR: {month: value}}.

    The time axis lives in structures[0].dimensions.observation[0].values and each
    observation is keyed by its position in that list; value is element 0.
    """
    data = payload.get("data") or {}
    if not data.get("dataSets"):
        return {}
    struct = data["structures"][0]["dimensions"]
    times = [v["value"] for v in struct["observation"][0]["values"]]
    sdims = struct["series"]
    out: dict[str, dict[pd.Period, float]] = {}
    for key, series in data["dataSets"][0].get("series", {}).items():
        ids = [sdims[i]["values"][int(k)]["id"] for i, k in enumerate(key.split(":"))]
        obs = {}
        for idx, o in series["observations"].items():
            t = times[int(idx)]
            if "-M" not in t or o[0] is None:
                continue
            try:
                obs[pd.Period(t.replace("-M", "-"), "M")] = float(o[0])
            except (ValueError, TypeError):
                continue
        out[ids[1]] = obs
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_catalog() -> dict:
    """Countries available live: names + which have CPI and exchange-rate data."""
    names_payload = _http_json(CODELIST_URL)
    names = {c["id"]: c["name"] for c in names_payload["data"]["codelists"][0]["codes"]}

    def codes(url: str) -> set[str]:
        p = _http_json(url, {"lastNObservations": 1})
        d = p["data"]
        dim0 = d["structures"][0]["dimensions"]["series"][0]["values"]
        return {dim0[int(k.split(":")[0])]["id"] for k in d["dataSets"][0].get("series", {})}

    cpi = codes(f"{CPI_FLOW}/*.*._T.IX.M")
    fx = codes(f"{ER_FLOW}/*.XDC_USD.PA_RT.M")
    return {"names": names, "isos": sorted(cpi & fx)}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_live(iso: str) -> pd.DataFrame:
    """Monthly CPI (P) and local-currency-per-USD rate (E) for one economy, from the IMF.

    Raises RuntimeError / DataProblem on failure (failures are NOT cached).
    """
    cpi = _parse_sdmx(_http_json(f"{CPI_FLOW}/{iso}.*._T.IX.M"))
    p = cpi.get("CPI") or cpi.get("HICP")           # euro area only publishes HICP
    if not p:
        raise DataProblem(f"The IMF has no monthly price index (CPI) for {iso}.")
    if iso == "USA":
        e = {m: 1.0 for m in p}                      # USD per USD = 1 by definition
    else:
        fx = _parse_sdmx(_http_json(f"{ER_FLOW}/{iso}.XDC_USD.PA_RT.M"))
        e = fx.get("XDC_USD")
        if not e:
            raise DataProblem(f"The IMF has no monthly exchange rate against the USD for {iso}.")
    return _to_frame(pd.Series(p), pd.Series(e), iso == "USA")


def _to_frame(p: pd.Series, e: pd.Series, constant_e: bool = False) -> pd.DataFrame:
    """Align P and E on a monthly PeriodIndex, from 1976, no future months, no duplicates."""
    now = pd.Period(pd.Timestamp.today(), "M")
    df = pd.DataFrame({"P": p, "E": e})
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= START_PERIOD) & (df.index <= now)]
    if constant_e and len(df):                       # USA: E = 1 in EVERY month, even where CPI is missing
        df = df.reindex(pd.period_range(df.index[0], df.index[-1], freq="M"))
        df["E"] = 1.0
    return df


# ----------------------------------------------------------------------------
# FALLBACK: saved CSV
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_backup() -> pd.DataFrame:
    """Long CSV (iso3, series, date, value) -> long DataFrame with a month column."""
    df = pd.read_csv(BACKUP_PATH, dtype={"iso3": str, "series": str, "date": str})
    df["month"] = pd.PeriodIndex(df["date"], freq="M")
    return df


def backup_last_month() -> str:
    return str(load_backup()["month"].max())


def backup_frame(iso: str) -> pd.DataFrame | None:
    b = load_backup()
    b = b[b["iso3"] == iso]
    if b.empty:
        return None
    p = b[b["series"] == "CPI"].set_index("month")["value"]
    e = b[b["series"] == "FX"].set_index("month")["value"]
    return _to_frame(p, e, iso == "USA")


@dataclass
class Loaded:
    iso: str
    df: pd.DataFrame | None = None
    source: str = "live"            # "live" or "backup"
    problem: str = ""               # plain-language reason when df is None
    live_error: str = ""            # why live failed when we used the backup


def load_country(iso: str) -> Loaded:
    """Live IMF first; saved CSV if the API is down; never raises."""
    live_err, api_failed = "", False
    if time.time() >= _api_state["down_until"]:
        try:
            df = fetch_live(iso)
            if len(df.dropna()) == 0:
                raise DataProblem("The IMF returned no overlapping data.")
            return Loaded(iso, df, "live")
        except DataProblem as exc:                   # API is up, this series just isn't there
            live_err = str(exc)
        except Exception as exc:                     # network/timeout/bad response
            _api_state["down_until"] = time.time() + API_COOLDOWN
            live_err, api_failed = f"live data unavailable ({exc})", True
    else:
        live_err, api_failed = "live data unavailable (recent failure)", True
    try:
        bdf = backup_frame(iso)
    except Exception:
        bdf = None
    if bdf is not None and len(bdf.dropna()):
        return Loaded(iso, bdf, "backup", live_error=live_err)
    if api_failed:
        return Loaded(iso, None, problem=f"The IMF website is not responding right now, and {cname(iso)} is not part "
                                         "of the saved backup data.")
    return Loaded(iso, None, problem=live_err.replace(f" for {iso}.", f" for {cname(iso)}.") or "No data was found.")


def api_down() -> bool:
    return time.time() < _api_state["down_until"]


def get_catalog() -> tuple[dict, list[str], bool]:
    """(imf_names, list of ISO codes to offer, live?) - backup list if the API is down."""
    if time.time() >= _api_state["down_until"]:
        try:
            cat = fetch_catalog()
            isos = [i for i in cat["isos"] if i == EURO_AREA or not re.fullmatch(r"G\d{3}", i)]  # drop IMF aggregates (G001 World...)
            if "USA" not in isos:
                isos.append("USA")
            return cat["names"], isos, True
        except Exception:
            _api_state["down_until"] = time.time() + API_COOLDOWN
    return {}, list(BACKUP_ISOS), False


# ----------------------------------------------------------------------------
# COMPUTE
# ----------------------------------------------------------------------------
@dataclass
class Result:
    iso1: str
    iso2: str
    c1: str
    c2: str
    table: pd.DataFrame                 # valid months only
    n_window: int = 0                   # calendar months in the window
    n_valid: int = 0
    n_dropped: int = 0
    drop_reasons: dict = field(default_factory=dict)
    rer_star: float = float("nan")
    sd: float = float("nan")            # std of the Real Index (index points); nan if undefined
    source1: str = "live"
    source2: str = "live"
    warnings: list = field(default_factory=list)

    @property
    def per(self) -> str:               # "USD per GBP"
        return f"{self.c2} per {self.c1}"


def combine(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month S, P1, P2 (+E1, E2). Invalid (<=0, inf, NaN) values become NaN."""
    idx = d1.index.union(d2.index)
    out = pd.DataFrame(index=idx)
    out["E1"], out["P1"] = d1["E"].reindex(idx), d1["P"].reindex(idx)
    out["E2"], out["P2"] = d2["E"].reindex(idx), d2["P"].reindex(idx)
    for col in ("E1", "P1", "E2", "P2"):
        s = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        out[col] = s.where(s > 0)
    out["S"] = out["E2"] / out["E1"]                 # units of currency 2 per unit of currency 1
    return out


def compute(d1: pd.DataFrame, d2: pd.DataFrame, iso1: str, iso2: str,
            start: pd.Period, end: pd.Period, src1="live", src2="live") -> Result:
    """RER(t) = S * P1/P2 ; RER* = mean over valid months in the window ; Real Index = 100*RER/RER*."""
    if start > end:
        raise DataProblem("The start month is after the end month. Please pick a start that is earlier than the end.")
    full = combine(d1, d2)
    window = pd.period_range(start, end, freq="M")
    w = full.reindex(window)
    need = {"E1": f"exchange rate of {cname(iso1)}", "E2": f"exchange rate of {cname(iso2)}",
            "P1": f"prices (CPI) of {cname(iso1)}", "P2": f"prices (CPI) of {cname(iso2)}"}
    valid_mask = w[["S", "P1", "P2"]].notna().all(axis=1)
    reasons = {txt: int(w[col].isna().sum()) for col, txt in need.items() if w[col].isna().any()}
    res = Result(iso1, iso2, ccy(iso1), ccy(iso2), pd.DataFrame(), n_window=len(window),
                 n_valid=int(valid_mask.sum()), source1=src1, source2=src2)
    res.n_dropped = res.n_window - res.n_valid
    res.drop_reasons = reasons
    if res.n_valid == 0:
        return res
    t = w[valid_mask].copy()
    t["RER"] = t["S"] * t["P1"] / t["P2"]
    res.rer_star = float(t["RER"].mean())
    t["RI"] = 100.0 * t["RER"] / res.rer_star
    res.sd = float(t["RI"].std(ddof=1)) if len(t) > 1 else float("nan")
    if not math.isfinite(res.sd) or res.sd < 1e-9:
        res.sd = float("nan")
        t["z"] = np.nan
    else:
        t["z"] = (t["RI"] - 100.0) / res.sd
    res.table = t
    if t["RI"].max() / max(t["RI"].min(), 1e-12) > 50:
        res.warnings.append("Extreme swings (likely hyperinflation or a currency reform). Math is right but hard to "
                            "read - try a shorter window.")
    return res


# ----------------------------------------------------------------------------
# SIGNAL
# ----------------------------------------------------------------------------
def zone_for(z: float) -> tuple[str, str]:
    for upper, lab, key in ZONES:
        if z < upper:
            return lab, key
    return ZONES[-1][1], ZONES[-1][2]


def signal(res: Result) -> dict | None:
    """z = (latest Real Index - 100) / std(Real Index). None if undefined (std unavailable)."""
    t = res.table
    if t.empty or not math.isfinite(res.sd):
        return None
    latest = float(t["RI"].iloc[-1])
    z = (latest - 100.0) / res.sd
    lab, key = zone_for(z)
    pct = float((t["RI"] < latest).mean() * 100)
    return {"z": z, "label": lab, "key": key, "latest": latest, "percentile": pct,
            "dist": latest - 100.0, "month": t.index[-1]}


def scenarios(sig: dict, c1: str, c2: str) -> tuple[str, str]:
    """One-line verdicts for the two scenarios, always naming both currencies."""
    k = sig["key"]
    if k in ("very_cheap", "cheap"):
        a = f"✅ **Good entry.** The {c1} is cheap in real terms and has room to recover."
        b = f"⚠️ **Poor timing.** If the cheap {c1} recovers, your {c1} loan gets more expensive to repay."
    elif k in ("expensive", "very_expensive"):
        a = f"⚠️ **Poor timing.** You'd buy a pricey {c1} and lose if it falls back to its average."
        b = f"✅ **Good entry.** The {c1} is pricey; your {c1} loan gets cheaper to repay if it falls back."
    else:
        a = b = f"⚖️ **Neutral.** The {c1} is near its normal level against the {c2}: no currency edge."
    return a, b


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------
FONT = "Plus Jakarta Sans, Inter, Arial, sans-serif"
C_REAL, C_NOM = "#4338ca", "#f59e0b"


def make_chart(res: Result, view: str) -> go.Figure:
    t = res.table
    x = t.index.to_timestamp()
    per, c1, c2 = res.per, res.c1, res.c2
    ri_name = f"Real Index ({c1} vs {c2}, 100 = period average)"
    s_name = f"S = {per}"
    ri_hover = "Real Index: %{y:.1f}  (above 100 = " + c1 + " expensive)<extra></extra>"
    s_hover = per + ": %{y:,.4f}<extra></extra>"
    both = view == "Both"
    fig = make_subplots(specs=[[{"secondary_y": both}]])

    if view in ("Real", "Both"):
        fig.add_trace(go.Scatter(x=x, y=t["RI"], name=ri_name, mode="lines", line=dict(color=C_REAL, width=3),
                                 hovertemplate=ri_hover), secondary_y=False)
    if view in ("Nominal", "Both"):
        fig.add_trace(go.Scatter(x=x, y=t["S"], name=s_name, mode="lines",
                                 line=dict(color=C_NOM, width=3 if view == "Nominal" else 2.2),
                                 hovertemplate=s_hover), secondary_y=True if both else False)

    if view in ("Real", "Both"):          # shapes after traces so Plotly knows the subplot
        if math.isfinite(res.sd):
            sd = res.sd
            fig.add_hrect(y0=100 - 0.5 * sd, y1=100 + 0.5 * sd, fillcolor="rgba(234,179,8,0.14)", line_width=0, layer="below", row=1, col=1)
            fig.add_hrect(y0=100 + 0.5 * sd, y1=100 + 1.5 * sd, fillcolor="rgba(239,68,68,0.10)", line_width=0, layer="below", row=1, col=1)
            fig.add_hrect(y0=100 - 1.5 * sd, y1=100 - 0.5 * sd, fillcolor="rgba(34,197,94,0.10)", line_width=0, layer="below", row=1, col=1)
            for k, txt in ((1.5, "+1.5 SD · very expensive"), (0.5, "+0.5 SD"), (-0.5, "−0.5 SD"), (-1.5, "−1.5 SD · very cheap")):
                fig.add_hline(y=100 + k * sd, line_width=1, line_dash="dot", line_color="rgba(100,116,139,0.6)",
                              annotation_text=txt, annotation_position="top left",
                              annotation_font=dict(size=11, color="#64748b"), row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="#334155", line_width=2,
                      annotation_text="100 = average", annotation_position="bottom right",
                      annotation_font=dict(size=12, color="#334155"), row=1, col=1)
    if view == "Real":
        title, ytitle = f"Is the {c1} cheap or expensive vs the {c2}?", "Real Index (100 = period average)"
    elif view == "Nominal":
        title, ytitle = f"Market rate: {per}", f"S = {per}"
    else:
        title, ytitle = f"Real Index vs market rate ({per})", "Real Index (100 = period average)"
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=21, family=FONT)),
                      height=540, hovermode="x unified", font=dict(size=14, family=FONT),
                      legend=dict(orientation="h", y=-0.16, font=dict(size=13)),
                      margin=dict(l=10, r=10, t=70, b=10),
                      hoverlabel=dict(font=dict(family=FONT, size=14)))
    fig.update_yaxes(title_text=ytitle, secondary_y=False, title_font=dict(size=14), gridcolor="#eef2f7")
    if both:
        fig.update_yaxes(title_text=f"S = {per} (currency units)", secondary_y=True, showgrid=False,
                         title_font=dict(size=14, color=C_NOM), tickfont=dict(color=C_NOM))
    fig.update_xaxes(title_text=None, showgrid=False, hoverformat="%b %Y",
                     rangeselector=dict(buttons=[dict(count=5, label="5Y", step="year", stepmode="backward"),
                                                 dict(count=10, label="10Y", step="year", stepmode="backward"),
                                                 dict(step="all", label="All")]))
    return fig


# ----------------------------------------------------------------------------
# EXPLAIN
# ----------------------------------------------------------------------------
def fmt(x: float) -> str:
    """Readable number for both tiny and huge values."""
    if x is None or not math.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}"
    return f"{x:.6g}"


def fmt_banner(x: float) -> str:
    """Headline number: 2 decimals for normal values (1.34), more precision for tiny ones."""
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4g}"


def mlabel(p: pd.Period) -> str:
    return p.strftime("%b %Y")


def explain(res: Result, view: str, lab1: str, lab2: str) -> str:
    t, c1, c2 = res.table, res.c1, res.c2
    last, ri_last = t.index[-1], float(t["RI"].iloc[-1])
    hi_m, lo_m = t["RI"].idxmax(), t["RI"].idxmin()
    dropped = ""
    if res.n_dropped:
        why = ", ".join(f"no {txt}" for txt in res.drop_reasons)
        dropped = f" · {res.n_dropped} skipped ({why})"
    where = "above 100 → pricier than usual" if ri_last > 100 else "below 100 → cheaper than usual"
    how = {
        "Real": f"Blue line above the dashed 100 = {c1} expensive; below = cheap. Shaded bands = normal swings.",
        "Nominal": f"Orange line = market price of 1 {c1} in {c2}. Not inflation-adjusted, so it can't tell cheap from expensive.",
        "Both": f"Blue (left) = Real Index; orange (right) = {res.per}. Gaps between them come from inflation.",
    }[view]
    return f"""
- 📅 **Period:** {mlabel(t.index[0])} – {mlabel(t.index[-1])} · **{res.n_valid} valid months**{dropped}
- 💱 **S (market rate):** 1 {c1} = **{fmt(float(t['S'].iloc[-1]))} {c2}** in {mlabel(last)}
- 📊 **Real Index:** **{ri_last:.1f}** now ({where}) · high {float(t['RI'].max()):.1f} ({mlabel(hi_m)}) · low {float(t['RI'].min()):.1f} ({mlabel(lo_m)})
- 👀 **How to read it:** {how}
"""


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.55; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero { background: linear-gradient(120deg,#1e1b4b 0%,#3730a3 55%,#0e7490 100%); color:#fff; border-radius:22px;
        padding: 1.6rem 2rem; margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(55,48,163,.25); }
.hero-t { font-size: 2.4rem; font-weight: 800; letter-spacing:-.02em; }
.hero-s { font-size: 1.1rem; opacity: .85; margin-top:.2rem; }
.dir-banner { background:#fff; border:2px solid #e0e7ff; border-radius:18px; padding: 1rem 1.4rem; margin: .2rem 0 .8rem 0;
              text-align:center; color:#1e1b4b; font-size: 2.2rem; font-weight: 800; line-height:1.2;
              box-shadow: 0 4px 18px rgba(30,27,75,.08); }
.dir-banner .pre { color:#6366f1; font-size:1rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; display:block; }
.dir-banner small { display:block; font-size:1rem; font-weight:500; color:#64748b; margin-top:.3rem; }
.step-title { font-size: 1.7rem; font-weight: 800; margin: 2rem 0 .5rem 0; color: inherit; letter-spacing:-.01em; }
.step-num { display:inline-block; background: linear-gradient(135deg,#4338ca,#0e7490); color:white; border-radius:12px;
            width:2.4rem; height:2.4rem; line-height:2.4rem; text-align:center; margin-right:.7rem; }
.stButton > button, .stDownloadButton > button { font-size: 1.2rem; padding: .8rem 1.5rem; border-radius: 14px; font-weight: 700;
            border: none; background: linear-gradient(135deg,#4338ca,#0e7490); color: #fff; box-shadow: 0 4px 14px rgba(67,56,202,.3); }
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.1); color:#fff; }
div[data-baseweb="select"] > div { font-size: 1.15rem; min-height: 3.1rem; border-radius: 12px; }
div[role="radiogroup"] { gap: .6rem; }
div[role="radiogroup"] label { font-size: 1.25rem !important; padding: .5rem 1.1rem; border-radius: 12px; border:1.5px solid #c7d2fe; }
.sig-card { border-radius: 20px; padding: 1.3rem 1.7rem; color: white; margin: .4rem 0 .8rem 0; box-shadow: 0 8px 24px rgba(0,0,0,.12); }
.sig-card .lbl { font-size:.95rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:.85; }
.sig-card .big { font-size: 2.5rem; font-weight: 800; }
.sig-card .sub { font-size: 1.1rem; opacity:.95; }
.meter { display:flex; height:12px; border-radius:8px; overflow:hidden; margin: 1rem 0 .3rem 0; position:relative;
         box-shadow: 0 0 0 2px rgba(255,255,255,.85); }
.meter span { flex:1; }
.meter-wrap { position:relative; }
.meter-pin { position:absolute; top:-6px; width:6px; height:24px; background:#fff; border-radius:3px; box-shadow:0 0 0 2px #111827; }
.meter-lbl { display:flex; justify-content:space-between; font-size:.8rem; opacity:.9; }
.tiles { display:flex; gap:.8rem; margin:.3rem 0 .8rem 0; flex-wrap:wrap; }
.tile { flex:1; min-width:150px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px; padding:.8rem 1rem; color:#0f172a; }
.tile .k { font-size:.85rem; color:#64748b; font-weight:600; }
.tile .v { font-size:1.6rem; font-weight:800; }
.scen { border-radius:16px; padding:1rem 1.2rem; background:#f8fafc; border:1px solid #e2e8f0; color:#0f172a; height:100%; margin-bottom:.9rem; }
.scen h4 { margin:0 0 .4rem 0; font-size:1.1rem; }
.caveats { background:#fffbeb; border:1px solid #fde68a; color:#78350f; padding:.8rem 1.1rem; border-radius:14px; font-size:.98rem; }
.chip { display:inline-block; background:#fef3c7; border-radius:999px; padding:.15rem .7rem; margin:.2rem .25rem; }
</style>
"""


def step_header(n: int, text: str) -> None:
    st.markdown(f'<div class="step-title"><span class="step-num">{n}</span>{text}</div>', unsafe_allow_html=True)


def friendly_stop(title: str, body: str) -> None:
    st.error(f"**{title}**\n\n{body}")


def swap_countries() -> None:
    """Callback for the swap button: flips Country 1 and 2 and records an explanation."""
    a, b = st.session_state.get("c1"), st.session_state.get("c2")
    st.session_state["c1"], st.session_state["c2"] = b, a
    st.session_state["swap_note"] = (a, b)


def _md(md: str) -> str:
    """Tiny **bold** -> <b> converter for text placed inside HTML cards."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)


def render_signal(res: Result, lab1: str, lab2: str, container) -> None:
    with container:
        c1, c2 = res.c1, res.c2
        st.markdown("### 🧭 Is it a good moment?")
        sig = signal(res)
        if sig is None:
            st.info("Needs at least **2 valid months** to measure normal swings. Widen your dates in Step 2.")
        else:
            color = ZONE_COLORS[sig["key"]]
            pin = (min(max(sig["z"], -2.5), 2.5) + 2.5) / 5 * 100
            segs = "".join(f'<span style="background:{ZONE_COLORS[k]}"></span>' for _, _, k in ZONES)
            st.markdown(
                f'<div class="sig-card" style="background:linear-gradient(135deg,{color},{color}cc)">'
                f'<div class="lbl">The {c1} in real terms vs the {c2}</div><div class="big">{sig["label"]}</div>'
                f'<div class="sub">z-score {sig["z"]:+.2f} · {mlabel(sig["month"])}</div>'
                f'<div class="meter-wrap"><div class="meter">{segs}</div>'
                f'<div class="meter-pin" style="left:calc({pin:.1f}% - 3px); top:10px"></div></div>'
                f'<div class="meter-lbl"><span>Very cheap</span><span>Fair</span><span>Very expensive</span></div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="tiles"><div class="tile"><div class="k">Real Index now</div><div class="v">{sig["latest"]:.1f}</div></div>'
                f'<div class="tile"><div class="k">Distance from average (100)</div><div class="v">{sig["dist"]:+.1f}%</div></div>'
                f'<div class="tile"><div class="k">Pricier than … of months</div><div class="v">{sig["percentile"]:.0f}%</div></div></div>',
                unsafe_allow_html=True)
            st.markdown(f"Today the {c1} is more expensive in real terms than in **{sig['percentile']:.0f}%** of the months in your period.")
            a, b = scenarios(sig, c1, c2)
            col1, col2 = st.columns(2)
            col1.markdown(f'<div class="scen"><h4>💼 Invest in {c1} assets using a {c2} loan</h4>{_md(a)}</div>',
                          unsafe_allow_html=True)
            col2.markdown(f'<div class="scen"><h4>💼 Invest in {c2} assets using a {c1} loan</h4>{_md(b)}</div>',
                          unsafe_allow_html=True)
        st.markdown(
            f"**In plain words:** “expensive in real terms” means 1 {c1} buys fewer goods abroad than usual. "
            f"Real rates tend to drift back to their long-run average. "
            f"We judge “unusual” against this pair's own swings, since some pairs are calmer than others.")
        st.markdown(
            '<div class="caveats"><b>⚠️ Keep in mind</b><br>'
            '<span class="chip">Ignores interest rates, which can matter more</span>'
            '<span class="chip">Reversion takes years, not months</span>'
            '<span class="chip">Changes with your dates</span>'
            '<span class="chip">Pegs (e.g. HKD) can stay off for decades</span>'
            '<span class="chip">Educational, not financial advice</span></div>', unsafe_allow_html=True)


def build_csv(res: Result) -> bytes:
    t = res.table
    c1, c2 = res.c1, res.c2
    both_live = res.source1 == "live" and res.source2 == "live"
    out = pd.DataFrame({
        "date": t.index.strftime("%Y-%m"),
        f"E1_{c1}_per_USD": t["E1"].values,
        f"E2_{c2}_per_USD": t["E2"].values,
        f"S_{c2}_per_{c1}": t["S"].values,
        f"P1_CPI_{res.iso1}": t["P1"].values,
        f"P2_CPI_{res.iso2}": t["P2"].values,
        "RER": t["RER"].values,
        "RER_star": res.rer_star,
        "Real_Index": t["RI"].values,
        "z_score": t["z"].values,
        "data_source": "live" if both_live else "backup",
    })
    return out.to_csv(index=False).encode("utf-8")


def render_math(res: Result, lab1: str, lab2: str) -> None:
    t, c1, c2 = res.table, res.c1, res.c2
    months = list(t.index[::-1])
    m = st.selectbox("Pick a month", months, format_func=mlabel)
    r = t.loc[m]
    st.markdown(f"""
| Step | Formula | Value |
|---|---|---|
| E1, E2 | IMF rates per 1 USD (USA = 1) | {fmt(r.E1)} {c1} · {fmt(r.E2)} {c2} |
| **S** | E2 / E1 | **{fmt(r.S)} {c2} per {c1}** |
| P1, P2 | CPI of {cname(res.iso1)}, {cname(res.iso2)} | {fmt(r.P1)} · {fmt(r.P2)} |
| **RER** | S × P1 / P2 | **{fmt(r.RER)}** |
| **RER\\*** | average RER, {res.n_valid} valid months | **{fmt(res.rer_star)}** |
| **Real Index** | 100 × RER / RER\\* | **{r.RI:.2f}** |
""")
    avg = float(t["RI"].mean())
    ok = abs(avg - 100) < 1e-6
    st.markdown(f"**Live check:** Average of Real Index in your window = **{avg:.2f}** {'✅' if ok else '❌'}")
    st.caption("Each CPI has its own base year; it cancels out because the index divides by its own average.")


GLOSSARY = {
    "CPI": "Price of a typical shopping basket; its growth is inflation.",
    "Nominal exchange rate": "Market price of one currency in another (e.g. 1.34 USD per GBP).",
    "Real exchange rate": "The nominal rate adjusted for prices: what a currency really buys abroad.",
    "Appreciation": "A currency becomes worth more.",
    "Depreciation": "A currency becomes worth less.",
    "Standard deviation": "How far a number usually strays from its average.",
}


# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate Explorer", page_icon="💱", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-t">💱 Real Exchange Rate Explorer</div>'
                '<div class="hero-s">Is a currency cheap or expensive once inflation is counted? Live IMF data · 4 steps</div></div>',
                unsafe_allow_html=True)

    banner_slot = st.container()      # filled once we know the numbers
    notice_slot = st.container()

    imf_names, isos, catalog_live = get_catalog()
    star = [i for i in BACKUP_ISOS if i in isos]
    rest = sorted([i for i in isos if i not in BACKUP_ISOS], key=lambda i: cname(i, imf_names))
    options = star + rest
    for k in ("c1", "c2"):                      # keep a previous pick visible (e.g. Argentina while the API is down)
        v = st.session_state.get(k)
        if v in COUNTRY_INFO and v not in options:
            options.append(v)
    star_set = set(BACKUP_ISOS)

    def fmt_opt(i: str) -> str:
        return label(i, imf_names, star=i in star_set)

    # ---- Step 1
    step_header(1, "Choose your two countries")
    for k, d in (("c1", DEFAULT_C1), ("c2", DEFAULT_C2)):
        if st.session_state.get(k) not in options:
            st.session_state[k] = d if d in options else options[0]
    col_a, col_b, col_c = st.columns([4, 3, 4])
    col_a.selectbox("Country 1 · the currency you check", options, format_func=fmt_opt, key="c1")
    col_c.selectbox("Country 2 · priced in", options, format_func=fmt_opt, key="c2")
    col_b.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    col_b.button("🔄 Swap countries", on_click=swap_countries, width="stretch")
    st.caption("⭐ Most traded currencies: listed first, work even if the IMF is offline.")
    if not catalog_live:
        st.info("IMF list offline: showing the 11 ⭐ currencies only.")
    iso1, iso2 = st.session_state["c1"], st.session_state["c2"]
    lab1, lab2 = label(iso1, imf_names), label(iso2, imf_names)

    note = st.session_state.pop("swap_note", None)
    if note:
        a, b = note
        st.success(f"🔄 Swapped: now {ccy(a)} per {ccy(b)} (the inverse rate). The signal now rates the {ccy(b)}.")

    signal_slot = st.container()

    if iso1 == iso2:
        friendly_stop("You picked the same country twice",
                      "Pick two different countries in Step 1.")
        return

    # ---- load data
    with st.spinner("Fetching IMF data…"):
        L1, L2 = load_country(iso1), load_country(iso2)
    if any(L.source == "backup" or (L.df is None and api_down()) for L in (L1, L2)):
        with notice_slot:
            st.warning(f"⚠️ Live IMF data unavailable, using saved data up to {backup_last_month()}")

    for L in (L1, L2):
        if L.df is None:
            others = ", ".join(label(i, imf_names) for i in BACKUP_ISOS if i not in (iso1, iso2))
            friendly_stop(f"We could not get data for {label(L.iso, imf_names)}",
                          f"{L.problem}\n\n**Try instead:** {others}.")
            return

    # ---- valid-month range (used to limit the pickers)
    full = combine(L1.df, L2.df)
    valid = full.index[full[["S", "P1", "P2"]].notna().all(axis=1)]
    if len(valid) == 0:
        friendly_stop("These two countries have no months in common",
                      f"The IMF has no month with prices and rates for both {lab1} and {lab2}. "
                      "**Try another pair** (⭐ ones have the longest history)."
                      + (" Euro members' own rates stop in 1998, so they can't be paired with the Euro area."
                         if {iso1, iso2} & (EURO_MEMBERS | {EURO_AREA}) else ""))
        return
    month_opts = [p for p in pd.period_range(valid[0], valid[-1], freq="M")]
    month_strs = [str(p) for p in month_opts]

    # ---- Step 2
    step_header(2, "Choose the period")
    pair = tuple(sorted((iso1, iso2)))
    if st.session_state.get("pair_sig") != pair or st.session_state.get("start_m") not in month_strs \
            or st.session_state.get("end_m") not in month_strs:
        st.session_state["pair_sig"] = pair
        st.session_state["start_m"], st.session_state["end_m"] = month_strs[0], month_strs[-1]
    mfmt = lambda s: mlabel(pd.Period(s, "M"))
    d1, d2 = st.columns(2)
    d1.selectbox("From (first month)", month_strs, format_func=mfmt, key="start_m")
    d2.selectbox("To (last month)", month_strs, format_func=mfmt, key="end_m")
    st.caption(f"Available: {mlabel(month_opts[0])} – {mlabel(month_opts[-1])}")
    start, end = pd.Period(st.session_state["start_m"], "M"), pd.Period(st.session_state["end_m"], "M")

    try:
        res = compute(L1.df, L2.df, iso1, iso2, start, end, L1.source, L2.source)
    except DataProblem as exc:
        friendly_stop("Please check your dates", str(exc))
        return
    if res.n_valid == 0:
        friendly_stop("No usable months in this period",
                      "Every month in your window is missing a price or rate. **Widen the period** in Step 2.")
        return

    # ---- always-visible direction banner
    s_latest = float(res.table["S"].iloc[-1])
    with banner_slot:
        st.markdown(f'<div class="dir-banner"><span class="pre">{res.c2} per {res.c1}</span>'
                    f'You are viewing: 1 {res.c1} = {fmt_banner(s_latest)} {res.c2}'
                    f'<small>{lab1} → {lab2} · {mlabel(res.table.index[-1])}</small></div>', unsafe_allow_html=True)

    if res.n_dropped:
        why = "; ".join(f"{n} had no {txt}" for txt, n in res.drop_reasons.items())
        st.info(f"ℹ️ {res.n_dropped} of {res.n_window} months skipped ({why}). Using {res.n_valid}.")
    for w in res.warnings:
        st.warning("⚠️ " + w)
    if res.n_valid == 1:
        st.info("Only one month selected: normal swings can't be measured.")

    render_signal(res, lab1, lab2, signal_slot)

    # ---- Step 3
    step_header(3, "Choose the view")
    view_label = st.radio("View", ["📈 Real", "💱 Nominal", "🔀 Both"], horizontal=True, key="view_label",
                          label_visibility="collapsed")
    view = view_label.split(" ")[1]
    try:
        st.plotly_chart(make_chart(res, view), width="stretch")
    except Exception:
        friendly_stop("The chart could not be drawn", "Please try another period or country pair.")

    with st.container(border=True):
        st.markdown("### 📝 In plain words")
        st.markdown(explain(res, view, lab1, lab2))

    with st.expander("🧮 Show me the math"):
        render_math(res, lab1, lab2)

    # ---- Step 4
    step_header(4, "Download")
    st.download_button("⬇️ Download CSV", build_csv(res),
                       file_name=f"RER_{res.iso1}_{res.iso2}_{start}_{end}.csv", mime="text/csv")

    with st.expander("📖 Glossary"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** - {v}")

    st.caption("Source: IMF (CPI & Exchange Rates) via api.imf.org · Educational use only.")


def run() -> None:
    """Top-level guard: any unexpected error becomes a friendly message, never a traceback."""
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        st.error("**Something unexpected happened.** Please change a country or date and try again, or reload the page. "
                 f"(Technical note for your instructor: {type(exc).__name__})")


run()
