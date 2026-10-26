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
        res.warnings.append("The Real Index moves by a factor of more than 50 in this window, which usually means "
                            "hyperinflation or a currency redenomination. The numbers are mathematically correct "
                            "but hard to interpret - try a shorter window.")
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
    """Plain text for the two carry-style scenarios, always naming both currencies."""
    k = sig["key"]
    if k in ("very_cheap", "cheap"):
        a = (f"✅ **Historically a better entry point.** The {c1} is cheap in real terms, so it has room to "
             f"recover toward its average. Buying {c1} assets now means buying low.")
        b = (f"⚠️ **Not an attractive moment.** You would be buying {c2} assets (and selling {c1}) while the "
             f"{c1} is historically cheap - if the {c1} recovers, repaying a {c1} loan gets more expensive.")
    elif k in ("expensive", "very_expensive"):
        a = (f"⚠️ **Not a good moment.** You would be buying {c1} when it is historically pricey and could lose "
             f"if it returns to its average.")
        b = (f"✅ **Historically a better entry point.** The {c1} is expensive in real terms, so a {c1} loan is "
             f"'costly money' that could get cheaper to repay if the {c1} falls back toward its average.")
    else:
        a = f"⚖️ **Neutral.** The {c1} trades close to its normal level against the {c2}; the exchange rate gives no edge."
        b = f"⚖️ **Neutral.** The {c1} trades close to its normal level against the {c2}; the exchange rate gives no edge."
    return a, b


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------
def make_chart(res: Result, view: str) -> go.Figure:
    t = res.table
    x = t.index.to_timestamp()
    per, c1, c2 = res.per, res.c1, res.c2
    ri_name = f"Real Index ({c1} in real terms vs {c2}, 100 = period average)"
    s_name = f"S = {per}"
    ri_hover = ("%{x|%b %Y}<br>Real Index: %{y:.1f}<br>(above 100 = " + c1 + " expensive in real terms)<extra></extra>")
    s_hover = "%{x|%b %Y}<br>" + per + ": %{y:,.4f}<extra></extra>"
    both = view == "Both"
    fig = make_subplots(specs=[[{"secondary_y": both}]])

    if view in ("Real", "Both"):
        fig.add_trace(go.Scatter(x=x, y=t["RI"], name=ri_name, mode="lines", line=dict(color="#1f5fbf", width=3),
                                 hovertemplate=ri_hover), secondary_y=False)
    if view in ("Nominal", "Both"):
        fig.add_trace(go.Scatter(x=x, y=t["S"], name=s_name, mode="lines",
                                 line=dict(color="#e07b00", width=3 if view == "Nominal" else 2),
                                 hovertemplate=s_hover), secondary_y=True if both else False)

    if view in ("Real", "Both"):          # shapes after traces so Plotly knows the subplot
        if math.isfinite(res.sd):
            sd = res.sd
            fig.add_hrect(y0=100 - 0.5 * sd, y1=100 + 0.5 * sd, fillcolor="rgba(224,168,0,0.13)", line_width=0, layer="below", row=1, col=1)
            fig.add_hrect(y0=100 + 0.5 * sd, y1=100 + 1.5 * sd, fillcolor="rgba(229,83,61,0.10)", line_width=0, layer="below", row=1, col=1)
            fig.add_hrect(y0=100 - 1.5 * sd, y1=100 - 0.5 * sd, fillcolor="rgba(76,175,80,0.10)", line_width=0, layer="below", row=1, col=1)
            for k, txt in ((1.5, "+1.5 SD"), (0.5, "+0.5 SD"), (-0.5, "−0.5 SD"), (-1.5, "−1.5 SD")):
                fig.add_hline(y=100 + k * sd, line_width=1, line_dash="dot", line_color="rgba(120,120,120,0.55)",
                              annotation_text=txt, annotation_position="top left", annotation_font_size=11, row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="#444", line_width=2,
                      annotation_text="100 = average", annotation_position="bottom right", row=1, col=1)
    if view == "Real":
        title, ytitle = f"Real Index: is the {c1} cheap or expensive vs the {c2}?", "Real Index (100 = average of this period)"
    elif view == "Nominal":
        title, ytitle = f"Nominal exchange rate: {per}", f"S = {per}"
    else:
        title, ytitle = f"Real Index and nominal rate: {c1} vs {c2}", "Real Index (100 = average of this period)"
    fig.update_layout(title=dict(text=title, font=dict(size=22)), height=520, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18, font=dict(size=14)),
                      margin=dict(l=10, r=10, t=70, b=10), font=dict(size=14))
    fig.update_yaxes(title_text=ytitle, secondary_y=False, title_font=dict(size=15))
    if both:
        fig.update_yaxes(title_text=f"S = {per} (currency units)", secondary_y=True, showgrid=False,
                         title_font=dict(size=15))
    fig.update_xaxes(title_text="Month")
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
    s_last = float(t["S"].iloc[-1])
    dropped_txt = "no months were dropped."
    if res.n_dropped:
        why = "; ".join(f"{n} month(s) had no {txt}" for txt, n in res.drop_reasons.items())
        dropped_txt = f"{res.n_dropped} month(s) were dropped because a needed number was missing ({why})."
    direction = ("above 100 → real appreciation (more expensive)" if ri_last > 100
                 else "below 100 → real depreciation (cheaper)")
    how = {
        "Real": f"The blue line is the Real Index. The dashed line at 100 is the period average; when the line is above it, "
                f"the {c1} is more expensive in real terms. Shaded bands show normal swings (±0.5 and ±1.5 standard deviations).",
        "Nominal": f"The orange line is the plain market price: how many {c2} you get for 1 {c1}. It is NOT adjusted for "
                   f"inflation, so it can drift for years without telling you whether the {c1} is cheap or expensive.",
        "Both": f"The blue line (left axis) is the Real Index; the orange line (right axis) is the plain price in {c2} per {c1}. "
                f"Where the two disagree, inflation differences are doing the work.",
    }[view]
    return f"""
**a. What you selected.** Country 1 = {lab1}, Country 2 = {lab2}. Period used: **{mlabel(t.index[0])} to {mlabel(t.index[-1])}**
({res.n_window} calendar months in your window). **{res.n_valid} valid months** used; {dropped_txt}

**b. What S means.** S is how many **{c2}** you need to buy **1 {c1}**. Latest value ({mlabel(last)}): **1 {c1} = {fmt(s_last)} {c2}**.

**c. What the Real Index means.** It compares prices-adjusted exchange rates with this period's average (= 100). Above 100 = {lab1.split(' (')[0]}
is more expensive in real terms than its average in this period (real appreciation); below 100 = cheaper.
Right now ({mlabel(last)}) it is **{ri_last:.1f}** ({direction}). Highest: **{float(t['RI'].max()):.1f}** in {mlabel(hi_m)};
lowest: **{float(t['RI'].min()):.1f}** in {mlabel(lo_m)}.

**d. How to read this chart.** {how}
"""


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
CSS = """
<style>
html, body, [class*="css"] { font-size: 19px; }
.stApp h1 { font-size: 2.6rem; }
.step-title { font-size: 1.9rem; font-weight: 800; margin: 1.6rem 0 .4rem 0; }
.step-num { background:#1f5fbf; color:white; border-radius:50%; padding:2px 15px; margin-right:12px; }
.stButton > button, .stDownloadButton > button { font-size: 1.35rem; padding: .85rem 1.6rem; border-radius: 12px; font-weight: 700; }
div[data-baseweb="select"] > div { font-size: 1.2rem; min-height: 3.2rem; }
div[role="radiogroup"] label { font-size: 1.35rem !important; padding: .4rem 1rem; }
.dir-banner { background:#0b3d91; color:white; padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: .6rem;
              font-size: 2.3rem; font-weight: 800; text-align:center; line-height:1.25; }
.dir-banner small { display:block; font-size:1.05rem; font-weight:500; opacity:.9; margin-top:.3rem; }
.sig-card { border-radius: 18px; padding: 1.2rem 1.6rem; color: white; margin: .5rem 0 1rem 0; }
.sig-card .big { font-size: 2.6rem; font-weight: 900; }
.sig-card .sub { font-size: 1.25rem; }
.caveats { border-left: 8px solid #e0a800; background:#fff7dd; color:#3a2f00; padding:1rem 1.3rem; border-radius:10px; font-size:1.05rem; }
.caveats li { margin-bottom:.3rem; }
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


def render_signal(res: Result, lab1: str, lab2: str, container) -> None:
    with container:
        c1, c2 = res.c1, res.c2
        st.markdown("### 🧭 Is it a good moment? (valuation signal)")
        sig = signal(res)
        if sig is None:
            st.info("The signal needs at least **2 valid months with some variation** in your date window to measure the "
                    "normal swings of this pair. Please widen your dates in Step 2.")
        else:
            color = ZONE_COLORS[sig["key"]]
            st.markdown(f'<div class="sig-card" style="background:{color}"><div class="big">{sig["label"]}</div>'
                        f'<div class="sub">The <b>{c1}</b> in real terms vs the <b>{c2}</b> · z-score {sig["z"]:+.2f} · '
                        f'{mlabel(sig["month"])}</div></div>', unsafe_allow_html=True)
            p = sig["percentile"]
            st.markdown(f"**Today the {c1} is more expensive in real terms than in {p:.0f}% of the months in your period.** "
                        f"Its Real Index is {sig['latest']:.1f}, which is **{sig['dist']:+.1f}%** away from 100 (its average).")
            a, b = scenarios(sig, c1, c2)
            col1, col2 = st.columns(2)
            col1.markdown(f"#### 💼 Invest in {c1} assets using a {c2} loan\n{a}")
            col2.markdown(f"#### 💼 Invest in {c2} assets using a {c1} loan\n{b}")
        st.markdown(
            f"**In plain words:** “Expensive in real terms” means that, after correcting for inflation, one {c1} buys fewer goods "
            f"abroad than it usually does. Real exchange rates tend to drift back toward their long-run average. We compare "
            f"against this pair's own normal swings (standard deviation) because some pairs are naturally calmer than others.")
        st.markdown(
            '<div class="caveats"><b>⚠️ Important caveats</b><ul>'
            "<li>This ignores <b>interest rate differences</b>, which can matter more than the exchange rate.</li>"
            "<li>Reversion usually takes <b>years, not months</b>.</li>"
            "<li>The signal <b>changes with the date window</b> you choose.</li>"
            "<li><b>Pegged currencies</b> (like HKD) may stay “cheap” or “expensive” for decades.</li>"
            "<li>This is <b>educational, not financial advice</b>.</li></ul></div>", unsafe_allow_html=True)


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
    m = st.selectbox("Pick a month to walk through", months, format_func=mlabel)
    r = t.loc[m]
    st.markdown(f"""
**Month: {mlabel(m)}**

1. **E1** = {fmt(r.E1)} {c1} per 1 USD  ·  **E2** = {fmt(r.E2)} {c2} per 1 USD  (IMF; the USA is always 1)
2. **S = E2 / E1** = {fmt(r.E2)} / {fmt(r.E1)} = **{fmt(r.S)}** {c2} per {c1}
3. **P1** (CPI of {cname(res.iso1)}) = {fmt(r.P1)}  ·  **P2** (CPI of {cname(res.iso2)}) = {fmt(r.P2)}
4. **RER = S × P1 / P2** = {fmt(r.S)} × {fmt(r.P1)} / {fmt(r.P2)} = **{fmt(r.RER)}**
5. **RER\\*** = average of RER over the {res.n_valid} valid months = **{fmt(res.rer_star)}**
6. **Real Index = 100 × RER / RER\\*** = 100 × {fmt(r.RER)} / {fmt(res.rer_star)} = **{r.RI:.2f}**
""")
    avg = float(t["RI"].mean())
    ok = abs(avg - 100) < 1e-6
    st.markdown(f"**Live check:** Average of Real Index in your window = **{avg:.2f}** {'✅' if ok else '❌'}")
    st.caption("CPI levels use each country's own base year; that does not matter because the Real Index divides by its own average.")


GLOSSARY = {
    "CPI (Consumer Price Index)": "An index of the prices of a typical shopping basket; its growth is inflation.",
    "Nominal exchange rate": "The market price of one currency in another, with no inflation adjustment (e.g. 1.34 USD per GBP).",
    "Real exchange rate": "The nominal rate corrected for price differences between countries, so it shows what a currency really buys.",
    "Appreciation": "A currency becomes worth more (nominally, or in real terms when its prices rise faster than abroad's).",
    "Depreciation": "A currency becomes worth less (nominally, or in real terms when it gets cheaper relative to abroad).",
    "Standard deviation": "A measure of how far a number usually strays from its average; bigger means wilder swings.",
}


# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate Explorer", page_icon="💱", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("💱 Real Exchange Rate Explorer")
    st.caption("Live IMF data · Follow the 4 numbered steps · No economics knowledge needed")

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
    col_a.selectbox("Country 1 (the currency you are looking at)", options, format_func=fmt_opt, key="c1")
    col_c.selectbox("Country 2 (what you measure it in)", options, format_func=fmt_opt, key="c2")
    col_b.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    col_b.button("🔄 Swap countries", on_click=swap_countries, width="stretch")
    st.caption("⭐ **Most traded currencies** are listed first and keep working even when the IMF website is down.")
    if not catalog_live:
        st.info("The IMF country list could not be loaded, so only the 11 ⭐ most traded currencies are offered.")
    iso1, iso2 = st.session_state["c1"], st.session_state["c2"]
    lab1, lab2 = label(iso1, imf_names), label(iso2, imf_names)

    note = st.session_state.pop("swap_note", None)
    if note:
        a, b = note
        st.success(f"🔄 Swapped! Country 1 is now {label(b, imf_names)} and Country 2 is now {label(a, imf_names)}. "
                   f"The exchange rate is now quoted as {ccy(a)} per {ccy(b)} (the inverse), and a high Real Index now "
                   f"means the {ccy(b)} is expensive instead of the {ccy(a)}.")

    signal_slot = st.container()

    if iso1 == iso2:
        friendly_stop("You picked the same country twice",
                      "Real exchange rates compare two different economies. Please change Country 1 or Country 2 in Step 1.")
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
                          f"{L.problem}\n\nThis can happen when the IMF website is busy or the country does not publish "
                          f"monthly prices or exchange rates. **What to try next:** pick one of the most traded currencies, "
                          f"for example {others}.")
            return

    # ---- valid-month range (used to limit the pickers)
    full = combine(L1.df, L2.df)
    valid = full.index[full[["S", "P1", "P2"]].notna().all(axis=1)]
    if len(valid) == 0:
        friendly_stop("These two countries have no months in common",
                      f"{lab1} and {lab2} never have prices and exchange rates for the same month in the IMF data. "
                      "**What to try next:** choose another pair - the ⭐ countries have the longest histories."
                      + (" (Euro-area members stopped publishing their own exchange rate when they adopted the euro in "
                         "1999, so they cannot be compared with the Euro area. Compare either one with another currency instead.)"
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
    st.caption(f"Data available for this pair: {mlabel(month_opts[0])} to {mlabel(month_opts[-1])}.")
    start, end = pd.Period(st.session_state["start_m"], "M"), pd.Period(st.session_state["end_m"], "M")

    try:
        res = compute(L1.df, L2.df, iso1, iso2, start, end, L1.source, L2.source)
    except DataProblem as exc:
        friendly_stop("Please check your dates", str(exc))
        return
    if res.n_valid == 0:
        friendly_stop("No usable months in this period",
                      "In the window you chose, at least one of the four needed numbers (two exchange rates, two price "
                      "indexes) is missing for every month. **What to try next:** widen the period in Step 2.")
        return

    # ---- always-visible direction banner
    s_latest = float(res.table["S"].iloc[-1])
    with banner_slot:
        st.markdown(f'<div class="dir-banner">You are viewing: 1 {res.c1} = {fmt_banner(s_latest)} {res.c2}'
                    f'<small>{lab1} priced in {lab2} · latest data {mlabel(res.table.index[-1])} · '
                    f'quoted as {res.c2} per {res.c1}</small></div>', unsafe_allow_html=True)

    if res.n_dropped:
        why = "; ".join(f"{n} had no {txt}" for txt, n in res.drop_reasons.items())
        st.info(f"ℹ️ {res.n_dropped} of the {res.n_window} months in your period were left out because a number was missing "
                f"({why}). {res.n_valid} valid months are used.")
    for w in res.warnings:
        st.warning("⚠️ " + w)
    if res.n_valid == 1:
        st.info("Only one valid month is selected, so 'normal swings' (standard deviation) cannot be measured.")

    render_signal(res, lab1, lab2, signal_slot)

    # ---- Step 3
    step_header(3, "Choose what to see")
    view_label = st.radio("View", ["📈 Real", "💱 Nominal", "🔀 Both"], horizontal=True, key="view_label",
                          label_visibility="collapsed")
    view = view_label.split(" ")[1]
    try:
        st.plotly_chart(make_chart(res, view), width="stretch")
    except Exception:
        friendly_stop("The chart could not be drawn", "Please try another period or country pair.")

    with st.container(border=True):
        st.markdown("### 📝 What this means (in plain language)")
        st.markdown(explain(res, view, lab1, lab2))

    with st.expander("🧮 Show me the math"):
        render_math(res, lab1, lab2)

    # ---- Step 4
    step_header(4, "Download your data")
    st.download_button("⬇️ Download CSV (every valid month)", build_csv(res),
                       file_name=f"RER_{res.iso1}_{res.iso2}_{start}_{end}.csv", mime="text/csv")

    with st.expander("📖 Glossary"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** - {v}")

    st.caption("Source: IMF International Financial Statistics (CPI, Exchange Rates) via api.imf.org · Educational use only.")


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
