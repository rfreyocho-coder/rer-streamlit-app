# 💱 Real Exchange Rate Explorer

**▶️ Open the live app: https://rer-app-app-hszbk2cqnd5597x2pulu2z.streamlit.app/**

A Streamlit app that turns live IMF data into a plain-language answer to: *"Is this currency cheap or expensive against another one, after correcting for inflation?"* Built for an MBA assignment; the interface assumes the user knows no economics.

## What the app does

1. **Step 1 – pick two economies** (any IMF country plus the Euro area). Default: 🇬🇧 United Kingdom vs 🇺🇸 United States.
2. **A valuation signal card** says whether Country 1's currency is VERY CHEAP … VERY EXPENSIVE in real terms.
3. **Step 2 – pick a period** with month pickers limited to the available data (from January 1976).
4. **Step 3 – pick a view**: Real, Nominal or Both (two y-axes). The chart has a dashed line at 100 and shaded ±0.5 / ±1.5 standard-deviation bands.
5. **Plain-language explanation**, a "Show me the math" walk-through (with a live check that the Real Index averages 100), a glossary, and a **Step 4 CSV download**.

A big banner always states the direction, e.g. *"You are viewing: 1 GBP = 1.34 USD"*.

## Formulas

| Symbol | Meaning |
|---|---|
| `E1(t)`, `E2(t)` | IMF monthly rate, **local currency units per 1 USD** (`XDC_USD`). For the USA, `E = 1`. |
| `S(t) = E2(t) / E1(t)` | Units of **Country 2** currency per 1 unit of **Country 1** currency (UK/US → USD per GBP) |
| `P1(t)`, `P2(t)` | Monthly CPI index (euro area: HICP) |
| `RER(t) = S(t) · P1(t) / P2(t)` | Real exchange rate |
| `RER*` | Mean of `RER(t)` over the **valid months of the selected window** (recomputed on every change) |
| `Real Index(t) = 100 · RER(t) / RER*` | Averages exactly 100 over the window |

*Valid months* are months inside the window where `S`, `P1` and `P2` all exist and are positive. Others are dropped and the app says how many and why. Nominal `S(t)` is always plotted in real currency units, never normalised. CPI base years differ by country; this is harmless because the Real Index divides by its own average.

## Valuation-signal methodology

`z = (latest Real Index − 100) / SD(Real Index over the window)` (sample standard deviation).
Zones (constants at the top of `app.py`): z < −1.5 🟢🟢 VERY CHEAP · −1.5…−0.5 🟢 CHEAP · −0.5…0.5 🟡 FAIR · 0.5…1.5 🔴 EXPENSIVE · > 1.5 🔴🔴 VERY EXPENSIVE (all for Country 1's currency in real terms vs Country 2). Thresholds scale with each pair's own volatility. The percentile is the share of months in the window with a lower Real Index than today. Two scenarios (invest in C1 assets with a C2 loan, and the reverse) are translated from the zone.
**Caveats (also shown in-app):** ignores interest-rate differences; reversion takes years; the signal depends on the window; pegged currencies (e.g. HKD) can stay cheap/expensive for decades; educational, not financial advice.

## Data and backup

* Source: IMF SDMX 3.0 API, `https://api.imf.org/external/sdmx/3.0` (no key). Dataflows `IMF.STA/CPI` (key `{ISO3}.CPI._T.IX.M`; euro area `G163.HICP._T.IX.M`) and `IMF.STA/ER` (key `{ISO3}.XDC_USD.PA_RT.M`). URLs need a version segment, e.g. `/data/dataflow/IMF.STA/CPI/~/GBR.CPI._T.IX.M`. Country names come from codelist `IMF/CL_COUNTRY`.
* Calls are cached with `st.cache_data` (6 h), use a 25 s timeout and one retry.
* **Backup:** `imf_backup.csv` (long format `iso3, series, date, value`, ~0.3 MB) holds the full monthly CPI and FX history from 1976 for the 11 ⭐ economies: USA, Euro area, JPN, GBR, CHN, CHE, AUS, CAN, HKG, SGP, MEX. If the live API fails or times out, the app switches to this file and shows a yellow banner *"Live IMF data unavailable, using saved data up to [month]"*. If a country outside the backup is chosen while the API is down, the app suggests one of the 11. The CSV `data_source` column says `live` or `backup`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (branch `main`, files at the top level).
2. Go to <https://share.streamlit.io> → **Create app** → select this repo, branch `main`, main file `app.py`.
3. **Advanced settings → Python 3.12**; leave **Secrets** empty.
4. Click **Deploy**.

## Data caveats

* Euro-area members' IMF exchange-rate series end in 1998 (national currencies); they carry their legacy code (DEM, FRF…) and cannot be paired with the Euro area, whose data starts in 1999. Use the Euro area for the euro era.
* Australia's monthly CPI is new (≈ 28 months), Hong Kong/Singapore start ≈ 2004, China ≈ 1992, the euro area HICP in Dec 1999. Short windows give weak signals.
* Hyperinflation economies (Venezuela, Zimbabwe, Argentina in some periods) produce extreme values; the app warns but still calculates.
* Months after today and months with missing/non-positive values are dropped; duplicates keep the last value.
* Pegged currencies and central-bank interventions can keep the real rate away from "normal" for decades.
