"""
Battery usefulness dashboard for home PV / grid data.

What it does
------------
- Reads a CSV like your `alldata.csv` with 15-minute power data.
- Simulates a battery that charges from PV surplus and discharges to reduce grid import.
- Lets you input:
  - capacity per battery (kWh)
  - power per battery (kW)
  - number of batteries
  - charge/discharge efficiency
  - initial/minimum SOC
  - optional import/export electricity prices
  - EV charging threshold, EV charging power, and EV reward per kWh
- Shows date-range or single-day graphs for:
  - battery state of charge (SOC)
  - charging/discharging power
  - grid import/export before and after battery
  - PV production and consumption
- Infers EV/car charging from high-consumption periods.
- Adds an adjustable extra reward for each kWh charged into cars.
- Calculates usefulness metrics.

Run
---
pip install streamlit pandas plotly
streamlit run battery_dashboard.py

Put `alldata.csv` in the same folder, or upload it in the sidebar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import numpy as np
import pandas as pd


DATETIME_COL = "DateTime"
PV_COL_CANDIDATES = ["Productie (kW)", "PV (kW)", "Production (kW)", "Solar (kW)"]
LOAD_COL_CANDIDATES = ["Verbruik (kW)", "Consumption (kW)", "Load (kW)"]
GRID_EXPORT_COL_CANDIDATES = ["Naar net (kW)", "Export (kW)", "To grid (kW)"]
GRID_IMPORT_COL_CANDIDATES = ["Van net (kW)", "Import (kW)", "From grid (kW)"]


@dataclass(frozen=True)
class BatteryConfig:
    capacity_per_battery_kwh: float = 5.0
    power_per_battery_kw: float = 2.5
    number_of_batteries: int = 1
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc_percent: float = 0.0
    minimum_soc_percent: float = 0.0

    @property
    def total_capacity_kwh(self) -> float:
        return max(0.0, self.capacity_per_battery_kwh * self.number_of_batteries)

    @property
    def total_power_kw(self) -> float:
        return max(0.0, self.power_per_battery_kw * self.number_of_batteries)


@dataclass(frozen=True)
class EVChargingConfig:
    """Configuration for inferring car charging from measured house consumption."""

    enabled: bool = True
    threshold_kw: float = 9.0
    charging_power_kw: float = 7.0
    reward_per_kwh: float = 0.14
    detection_mode: str = "fixed_above_threshold"
    cap_to_measured_load: bool = True


@dataclass(frozen=True)
class ColumnMap:
    timestamp: str
    pv_kw: Optional[str]
    load_kw: Optional[str]
    grid_export_kw: Optional[str]
    grid_import_kw: Optional[str]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column, case-insensitive and whitespace-tolerant."""
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        found = normalized.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _to_number(series: pd.Series) -> pd.Series:
    """Convert Dutch/English numeric text to float."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_home_data(source: str | Path | BinaryIO) -> Tuple[pd.DataFrame, ColumnMap]:
    """Load and normalize home-energy data."""
    # Your file is semicolon-separated. If a different CSV is used later, this tries a fallback.
    try:
        df = pd.read_csv(source, sep=";", encoding="utf-8-sig")
        if len(df.columns) == 1:
            raise ValueError("Only one column found; retrying with comma separator")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=",", encoding="utf-8-sig")

    df.columns = [str(c).strip() for c in df.columns]

    if DATETIME_COL in df.columns:
        timestamp = pd.to_datetime(df[DATETIME_COL], format="%d-%m-%Y %H:%M", errors="coerce")
        if timestamp.isna().any():
            timestamp = pd.to_datetime(df[DATETIME_COL], dayfirst=True, errors="coerce")
    elif "Date" in df.columns and "Time" in df.columns:
        timestamp = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), dayfirst=True, errors="coerce")
    else:
        raise ValueError("No timestamp column found. Expected `DateTime`, or `Date` + `Time`.")

    df = df.assign(timestamp=timestamp).dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    colmap = ColumnMap(
        timestamp="timestamp",
        pv_kw=_find_col(df, PV_COL_CANDIDATES),
        load_kw=_find_col(df, LOAD_COL_CANDIDATES),
        grid_export_kw=_find_col(df, GRID_EXPORT_COL_CANDIDATES),
        grid_import_kw=_find_col(df, GRID_IMPORT_COL_CANDIDATES),
    )

    if colmap.pv_kw is None or colmap.load_kw is None:
        if colmap.grid_export_kw is None or colmap.grid_import_kw is None:
            raise ValueError(
                "Need either PV+load columns or grid import+export columns. "
                "For your file, expected `Productie (kW)`, `Verbruik (kW)`, `Naar net (kW)`, `Van net (kW)`."
            )

    numeric_cols = [c for c in [colmap.pv_kw, colmap.load_kw, colmap.grid_export_kw, colmap.grid_import_kw] if c]
    for col in numeric_cols:
        df[col] = _to_number(df[col]).fillna(0.0).clip(lower=0.0)

    # Interval duration in hours. Your file is normally 15 minutes. Large gaps are treated as one normal interval
    # so a missing day does not accidentally become 24 hours of energy.
    forward_delta = df["timestamp"].shift(-1) - df["timestamp"]
    valid = forward_delta[(forward_delta > pd.Timedelta(0)) & (forward_delta <= pd.Timedelta(hours=1))]
    median_delta = valid.median() if len(valid) else pd.Timedelta(minutes=15)
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        median_delta = pd.Timedelta(minutes=15)
    dt_hours = forward_delta.dt.total_seconds() / 3600.0
    median_hours = median_delta.total_seconds() / 3600.0
    dt_hours = dt_hours.where((dt_hours > 0) & (dt_hours <= max(1.0, 2 * median_hours)), median_hours)
    df["dt_h"] = dt_hours.fillna(median_hours)

    # Construct a clean net surplus/deficit. This avoids simultaneous import/export in one interval.
    if colmap.pv_kw and colmap.load_kw:
        df["pv_kw"] = df[colmap.pv_kw]
        df["load_kw"] = df[colmap.load_kw]
        df["baseline_surplus_kw"] = (df["pv_kw"] - df["load_kw"]).clip(lower=0.0)
        df["baseline_deficit_kw"] = (df["load_kw"] - df["pv_kw"]).clip(lower=0.0)
    else:
        df["pv_kw"] = np.nan
        df["load_kw"] = np.nan
        df["baseline_surplus_kw"] = df[colmap.grid_export_kw].clip(lower=0.0)
        df["baseline_deficit_kw"] = df[colmap.grid_import_kw].clip(lower=0.0)

    if colmap.grid_export_kw:
        df["raw_grid_export_kw"] = df[colmap.grid_export_kw]
    else:
        df["raw_grid_export_kw"] = df["baseline_surplus_kw"]

    if colmap.grid_import_kw:
        df["raw_grid_import_kw"] = df[colmap.grid_import_kw]
    else:
        df["raw_grid_import_kw"] = df["baseline_deficit_kw"]

    return df, colmap


def apply_ev_charging_model(df: pd.DataFrame, config: EVChargingConfig) -> pd.DataFrame:
    """
    Infer car charging from measured load.

    The input data usually contains one total consumption/load column. This function creates:
    - ev_charging_kw: inferred car charging power
    - non_ev_house_load_kw: measured load minus inferred car charging

    Detection modes:
    - fixed_above_threshold: if total load is above the threshold, count a fixed charging power.
    - excess_above_threshold: count only the part of load above the threshold, capped by charging power.
    """
    out = df.copy()

    if "load_kw" not in out.columns or out["load_kw"].isna().all() or not config.enabled:
        out["ev_charging_kw"] = 0.0
        out["non_ev_house_load_kw"] = out["load_kw"].fillna(0.0) if "load_kw" in out.columns else 0.0
        return out

    load = out["load_kw"].fillna(0.0).clip(lower=0.0)
    threshold = max(0.0, float(config.threshold_kw))
    charging_power = max(0.0, float(config.charging_power_kw))

    if config.detection_mode == "excess_above_threshold":
        ev_kw = (load - threshold).clip(lower=0.0).clip(upper=charging_power)
    else:
        ev_kw = pd.Series(np.where(load > threshold, charging_power, 0.0), index=out.index, dtype="float64")

    if config.cap_to_measured_load:
        ev_kw = np.minimum(ev_kw, load)

    out["ev_charging_kw"] = pd.Series(ev_kw, index=out.index).fillna(0.0).clip(lower=0.0)
    out["non_ev_house_load_kw"] = (load - out["ev_charging_kw"]).clip(lower=0.0)
    return out


def simulate_battery(df: pd.DataFrame, config: BatteryConfig) -> pd.DataFrame:
    """
    Simulate a behind-the-meter battery.

    Model assumptions:
    - The battery charges only from PV surplus.
    - The battery discharges only to reduce house grid import.
    - It does not charge from the grid or do price arbitrage.
    - Power values are average kW over each interval.
    """
    out = df.copy()
    capacity_kwh = config.total_capacity_kwh
    power_kw = config.total_power_kw
    charge_eff = max(0.0001, min(1.0, config.charge_efficiency))
    discharge_eff = max(0.0001, min(1.0, config.discharge_efficiency))

    if capacity_kwh <= 0 or power_kw <= 0 or config.number_of_batteries <= 0:
        out["battery_charge_kw"] = 0.0
        out["battery_discharge_kw"] = 0.0
        out["soc_kwh"] = 0.0
        out["soc_percent"] = 0.0
        out["grid_import_after_kw"] = out["baseline_deficit_kw"]
        out["grid_export_after_kw"] = out["baseline_surplus_kw"]
        out["battery_losses_kw_equiv"] = 0.0
        return out

    soc = capacity_kwh * max(0.0, min(100.0, config.initial_soc_percent)) / 100.0
    reserve = capacity_kwh * max(0.0, min(100.0, config.minimum_soc_percent)) / 100.0
    soc = max(reserve, min(capacity_kwh, soc))

    charge_kw = np.zeros(len(out))
    discharge_kw = np.zeros(len(out))
    soc_kwh = np.zeros(len(out))
    grid_import_after_kw = np.zeros(len(out))
    grid_export_after_kw = np.zeros(len(out))
    losses_kw_equiv = np.zeros(len(out))

    surplus = out["baseline_surplus_kw"].to_numpy(dtype=float)
    deficit = out["baseline_deficit_kw"].to_numpy(dtype=float)
    dt_h = out["dt_h"].to_numpy(dtype=float)

    for i in range(len(out)):
        dt = max(float(dt_h[i]), 1e-9)
        pv_surplus_kw = max(0.0, float(surplus[i]))
        demand_deficit_kw = max(0.0, float(deficit[i]))

        # Charge from PV surplus. charge_input_kwh is AC energy taken from PV surplus.
        free_capacity_kwh = max(0.0, capacity_kwh - soc)
        max_charge_input_kwh = min(pv_surplus_kw, power_kw) * dt
        charge_input_kwh = min(max_charge_input_kwh, free_capacity_kwh / charge_eff)
        soc += charge_input_kwh * charge_eff
        actual_charge_kw = charge_input_kwh / dt
        remaining_surplus_kw = max(0.0, pv_surplus_kw - actual_charge_kw)

        # Discharge to cover house deficit. discharge_output_kwh is AC energy delivered to the house.
        available_output_kwh = max(0.0, soc - reserve) * discharge_eff
        max_discharge_output_kwh = min(demand_deficit_kw, power_kw) * dt
        discharge_output_kwh = min(max_discharge_output_kwh, available_output_kwh)
        soc -= discharge_output_kwh / discharge_eff
        actual_discharge_kw = discharge_output_kwh / dt
        remaining_deficit_kw = max(0.0, demand_deficit_kw - actual_discharge_kw)

        charge_kw[i] = actual_charge_kw
        discharge_kw[i] = actual_discharge_kw
        soc_kwh[i] = soc
        grid_import_after_kw[i] = remaining_deficit_kw
        grid_export_after_kw[i] = remaining_surplus_kw
        losses_kw_equiv[i] = max(0.0, actual_charge_kw - actual_discharge_kw)

    out["battery_charge_kw"] = charge_kw
    out["battery_discharge_kw"] = discharge_kw
    out["soc_kwh"] = soc_kwh
    out["soc_percent"] = np.where(capacity_kwh > 0, 100.0 * soc_kwh / capacity_kwh, 0.0)
    out["grid_import_after_kw"] = grid_import_after_kw
    out["grid_export_after_kw"] = grid_export_after_kw
    out["battery_losses_kw_equiv"] = losses_kw_equiv
    return out


def summarize(
    df: pd.DataFrame,
    config: BatteryConfig,
    import_price: float = 0.0,
    export_price: float = 0.0,
    ev_reward_per_kwh: float = 0.0,
) -> dict[str, float]:
    """Summarize energy, battery usefulness, and inferred EV charging for the supplied period."""
    if len(df) == 0:
        return {}

    dt = df["dt_h"]
    pv_kwh = float((df["pv_kw"].fillna(0) * dt).sum())
    load_kwh = float((df["load_kw"].fillna(0) * dt).sum())
    ev_charging_kwh = float((df.get("ev_charging_kw", pd.Series(0.0, index=df.index)) * dt).sum())
    non_ev_house_load_kwh = float((df.get("non_ev_house_load_kw", df["load_kw"].fillna(0.0)) * dt).sum())
    baseline_import_kwh = float((df["baseline_deficit_kw"] * dt).sum())
    baseline_export_kwh = float((df["baseline_surplus_kw"] * dt).sum())
    after_import_kwh = float((df["grid_import_after_kw"] * dt).sum())
    after_export_kwh = float((df["grid_export_after_kw"] * dt).sum())
    charged_kwh = float((df["battery_charge_kw"] * dt).sum())
    discharged_kwh = float((df["battery_discharge_kw"] * dt).sum())
    avoided_import_kwh = baseline_import_kwh - after_import_kwh
    reduced_export_kwh = baseline_export_kwh - after_export_kwh
    battery_losses_kwh = max(0.0, charged_kwh - discharged_kwh)

    if pv_kwh <= 0 and "raw_grid_export_kw" in df.columns:
        pv_kwh = baseline_export_kwh
    if load_kwh <= 0 and "raw_grid_import_kw" in df.columns:
        load_kwh = baseline_import_kwh

    total_capacity = config.total_capacity_kwh
    cycles = discharged_kwh / total_capacity if total_capacity > 0 else 0.0
    battery_value = avoided_import_kwh * import_price - reduced_export_kwh * export_price
    ev_reward = ev_charging_kwh * max(0.0, ev_reward_per_kwh)

    return {
        "PV generation kWh": pv_kwh,
        "Total measured consumption kWh": load_kwh,
        "Non-EV house consumption kWh": non_ev_house_load_kwh,
        "EV charging kWh": ev_charging_kwh,
        "EV share of consumption %": 100 * ev_charging_kwh / load_kwh if load_kwh > 0 else 0.0,
        "Grid import before battery kWh": baseline_import_kwh,
        "Grid import after battery kWh": after_import_kwh,
        "Grid export before battery kWh": baseline_export_kwh,
        "Grid export after battery kWh": after_export_kwh,
        "Avoided grid import kWh": avoided_import_kwh,
        "Battery charge energy kWh": charged_kwh,
        "Battery discharge energy kWh": discharged_kwh,
        "Battery losses kWh": battery_losses_kwh,
        "Equivalent full cycles": cycles,
        "Self-consumption before %": 100 * (pv_kwh - baseline_export_kwh) / pv_kwh if pv_kwh > 0 else 0.0,
        "Self-consumption after %": 100 * (pv_kwh - after_export_kwh) / pv_kwh if pv_kwh > 0 else 0.0,
        "Self-sufficiency before %": 100 * (load_kwh - baseline_import_kwh) / load_kwh if load_kwh > 0 else 0.0,
        "Self-sufficiency after %": 100 * (load_kwh - after_import_kwh) / load_kwh if load_kwh > 0 else 0.0,
        "Estimated value": battery_value,
        "EV charging reward": ev_reward,
        "Estimated value including EV reward": battery_value + ev_reward,
    }

def _fmt_kwh(x: float) -> str:
    return f"{x:,.1f} kWh"


def _fmt_percent(x: float) -> str:
    return f"{x:,.1f}%"


def _fmt_money(x: float) -> str:
    return f"€{x:,.2f}"


def run_streamlit_app() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Home battery usefulness", layout="wide")
    st.title("Home battery usefulness simulator")
    st.caption(
        "Simulates a battery that charges from PV surplus and discharges to reduce grid import. "
        "The model uses net surplus/deficit per interval, so it does not charge and discharge at the same time."
    )

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        default_file = Path("alldata.csv")
        use_default = uploaded is None and default_file.exists()
        if uploaded is None and not use_default:
            st.info("Upload your CSV, or place `alldata.csv` next to this script.")
            st.stop()

        st.header("Battery")
        capacity_per_battery = st.number_input("Capacity per battery (kWh)", min_value=0.0, value=5.0, step=0.5)
        power_per_battery = st.number_input("Power per battery (kW)", min_value=0.0, value=2.5, step=0.5)
        number_of_batteries = st.number_input("Number of batteries", min_value=0, value=1, step=1)
        charge_eff = st.slider("Charge efficiency", min_value=0.50, max_value=1.00, value=0.95, step=0.01)
        discharge_eff = st.slider("Discharge efficiency", min_value=0.50, max_value=1.00, value=0.95, step=0.01)
        initial_soc = st.slider("Initial SOC at simulation start (%)", min_value=0, max_value=100, value=0, step=5)
        min_soc = st.slider("Minimum reserve SOC (%)", min_value=0, max_value=80, value=0, step=5)

        st.header("Optional economics")
        import_price = st.number_input("Import price (€/kWh)", min_value=0.0, value=0.30, step=0.01)
        export_price = st.number_input("Export compensation (€/kWh)", min_value=0.0, value=0.05, step=0.01)

        st.header("EV / car charging")
        ev_enabled = st.checkbox(
            "Infer car charging from high consumption",
            value=True,
            help="Uses the measured total consumption column to estimate when the car is charging.",
        )
        ev_threshold_kw = st.number_input(
            "Consumption threshold for car charging (kW)",
            min_value=0.0,
            value=9.0,
            step=0.5,
            help="When measured consumption is above this level, the app assumes car charging is active.",
        )
        ev_charging_power_kw = st.number_input(
            "Car charging power when active (kW)",
            min_value=0.0,
            value=7.0,
            step=0.5,
        )
        ev_reward_per_kwh = st.number_input(
            "Extra reward for car charging (€/kWh)",
            min_value=0.0,
            value=0.14,
            step=0.01,
        )
        ev_detection_label = st.selectbox(
            "Car charging detection method",
            ["Fixed power above threshold", "Only excess above threshold"],
            help=(
                "Fixed: if consumption is above the threshold, count the chosen charging power. "
                "Excess: count only the kW above the threshold, capped by the chosen charging power."
            ),
        )
        ev_cap_to_load = st.checkbox(
            "Cap car charging to measured consumption",
            value=True,
            help="Prevents inferred car charging from being higher than the measured total consumption in an interval.",
        )

        st.header("Sizing comparison")
        max_sweep_batteries = st.number_input(
            "Compare 0 to this many batteries", min_value=0, max_value=50, value=max(5, int(number_of_batteries)), step=1
        )

    data_source = uploaded if uploaded is not None else default_file
    df, colmap = load_home_data(data_source)

    ev_cfg = EVChargingConfig(
        enabled=bool(ev_enabled),
        threshold_kw=ev_threshold_kw,
        charging_power_kw=ev_charging_power_kw,
        reward_per_kwh=ev_reward_per_kwh,
        detection_mode="excess_above_threshold" if ev_detection_label == "Only excess above threshold" else "fixed_above_threshold",
        cap_to_measured_load=bool(ev_cap_to_load),
    )
    df = apply_ev_charging_model(df, ev_cfg)
    if ev_enabled and colmap.load_kw is None:
        st.warning("EV charging inference needs a measured consumption/load column. No EV charging was inferred.")

    min_date = df["timestamp"].dt.date.min()
    max_date = df["timestamp"].dt.date.max()

    st.sidebar.header("Period")
    picker_mode = st.sidebar.radio("Picker mode", ["Single day", "Date range"], horizontal=True)
    carry_soc = st.sidebar.checkbox(
        "Carry SOC from previous data",
        value=True,
        help="Recommended. The full file is simulated first, then the selected period is displayed. If off, SOC resets at the start of the selected period.",
    )

    if picker_mode == "Single day":
        chosen_day = st.sidebar.date_input("Day", value=max_date, min_value=min_date, max_value=max_date)
        start_ts = pd.Timestamp(chosen_day)
        end_ts = start_ts + pd.Timedelta(days=1)
    else:
        chosen_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if not isinstance(chosen_range, tuple) or len(chosen_range) != 2:
            st.warning("Select a start and end date.")
            st.stop()
        start_ts = pd.Timestamp(chosen_range[0])
        end_ts = pd.Timestamp(chosen_range[1]) + pd.Timedelta(days=1)

    cfg = BatteryConfig(
        capacity_per_battery_kwh=capacity_per_battery,
        power_per_battery_kw=power_per_battery,
        number_of_batteries=int(number_of_batteries),
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        initial_soc_percent=initial_soc,
        minimum_soc_percent=min_soc,
    )

    if carry_soc:
        sim_all = simulate_battery(df, cfg)
        view = sim_all[(sim_all["timestamp"] >= start_ts) & (sim_all["timestamp"] < end_ts)].copy()
        summary_scope = view
    else:
        selected = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
        view = simulate_battery(selected, cfg)
        summary_scope = view

    if view.empty:
        st.warning("No data in the selected period.")
        st.stop()

    metrics = summarize(
        summary_scope,
        cfg,
        import_price=import_price,
        export_price=export_price,
        ev_reward_per_kwh=ev_cfg.reward_per_kwh,
    )

    st.subheader("Battery setup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total capacity", f"{cfg.total_capacity_kwh:,.1f} kWh")
    c2.metric("Total power", f"{cfg.total_power_kw:,.1f} kW")
    c3.metric("Batteries", f"{cfg.number_of_batteries}")
    c4.metric("Round-trip efficiency", _fmt_percent(100 * cfg.charge_efficiency * cfg.discharge_efficiency))

    st.subheader("Usefulness summary for selected period")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avoided grid import", _fmt_kwh(metrics["Avoided grid import kWh"]))
    m2.metric(
        "Import reduction",
        _fmt_percent(
            100 * metrics["Avoided grid import kWh"] / metrics["Grid import before battery kWh"]
            if metrics["Grid import before battery kWh"] > 0
            else 0
        ),
    )
    m3.metric("Equivalent full cycles", f"{metrics['Equivalent full cycles']:,.2f}")
    m4.metric("Battery value", _fmt_money(metrics["Estimated value"]))

    ev1, ev2, ev3, ev4 = st.columns(4)
    ev1.metric("Energy to cars", _fmt_kwh(metrics["EV charging kWh"]))
    ev2.metric("EV share of consumption", _fmt_percent(metrics["EV share of consumption %"]))
    ev3.metric("EV charging reward", _fmt_money(metrics["EV charging reward"]))
    ev4.metric("Total value incl. EV", _fmt_money(metrics["Estimated value including EV reward"]))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Self-consumption before", _fmt_percent(metrics["Self-consumption before %"]))
    m6.metric("Self-consumption after", _fmt_percent(metrics["Self-consumption after %"]))
    m7.metric("Self-sufficiency before", _fmt_percent(metrics["Self-sufficiency before %"]))
    m8.metric("Self-sufficiency after", _fmt_percent(metrics["Self-sufficiency after %"]))

    st.subheader("Graphs")

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=view["timestamp"], y=view["soc_percent"], name="SOC (%)", mode="lines"))
    fig_soc.update_layout(yaxis_title="SOC (%)", xaxis_title="Time", hovermode="x unified", height=330)
    st.plotly_chart(fig_soc, use_container_width=True)

    fig_battery = go.Figure()
    fig_battery.add_trace(go.Scatter(x=view["timestamp"], y=view["battery_charge_kw"], name="Charge from PV surplus (kW)", mode="lines"))
    fig_battery.add_trace(go.Scatter(x=view["timestamp"], y=-view["battery_discharge_kw"], name="Discharge to house (kW)", mode="lines"))
    fig_battery.update_layout(
        yaxis_title="Battery power (kW, discharge shown negative)",
        xaxis_title="Time",
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_battery, use_container_width=True)

    fig_grid = go.Figure()
    fig_grid.add_trace(go.Scatter(x=view["timestamp"], y=view["baseline_deficit_kw"], name="Import before battery (kW)", mode="lines"))
    fig_grid.add_trace(go.Scatter(x=view["timestamp"], y=view["grid_import_after_kw"], name="Import after battery (kW)", mode="lines"))
    fig_grid.add_trace(go.Scatter(x=view["timestamp"], y=-view["baseline_surplus_kw"], name="Export before battery (kW)", mode="lines"))
    fig_grid.add_trace(go.Scatter(x=view["timestamp"], y=-view["grid_export_after_kw"], name="Export after battery (kW)", mode="lines"))
    fig_grid.update_layout(
        yaxis_title="Grid power (kW, export shown negative)",
        xaxis_title="Time",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(fig_grid, use_container_width=True)

    if view["pv_kw"].notna().any() and view["load_kw"].notna().any():
        fig_pv_load = go.Figure()
        fig_pv_load.add_trace(go.Scatter(x=view["timestamp"], y=view["pv_kw"], name="PV production (kW)", mode="lines"))
        fig_pv_load.add_trace(go.Scatter(x=view["timestamp"], y=view["load_kw"], name="Total measured consumption (kW)", mode="lines"))
        if "non_ev_house_load_kw" in view.columns:
            fig_pv_load.add_trace(
                go.Scatter(x=view["timestamp"], y=view["non_ev_house_load_kw"], name="Consumption excluding cars (kW)", mode="lines")
            )
        fig_pv_load.update_layout(yaxis_title="Power (kW)", xaxis_title="Time", hovermode="x unified", height=380)
        st.plotly_chart(fig_pv_load, use_container_width=True)

    if "ev_charging_kw" in view.columns and ev_cfg.enabled:
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=view["timestamp"], y=view["ev_charging_kw"], name="Inferred car charging (kW)", mode="lines"))
        fig_ev.add_trace(go.Scatter(x=view["timestamp"], y=view["load_kw"], name="Total measured consumption (kW)", mode="lines"))
        fig_ev.update_layout(
            yaxis_title="Power (kW)",
            xaxis_title="Time",
            hovermode="x unified",
            height=340,
        )
        st.plotly_chart(fig_ev, use_container_width=True)

        daily_ev = (
            view.assign(date=view["timestamp"].dt.date, ev_kwh=view["ev_charging_kw"] * view["dt_h"])
            .groupby("date", as_index=False)["ev_kwh"]
            .sum()
        )
        if not daily_ev.empty:
            fig_ev_daily = go.Figure()
            fig_ev_daily.add_trace(go.Bar(x=daily_ev["date"], y=daily_ev["ev_kwh"], name="Car charging energy (kWh)"))
            fig_ev_daily.update_layout(
                yaxis_title="Car charging energy (kWh/day)",
                xaxis_title="Date",
                hovermode="x unified",
                height=320,
            )
            st.plotly_chart(fig_ev_daily, use_container_width=True)

    st.subheader("Sizing comparison")
    sweep_rows = []
    for n in range(int(max_sweep_batteries) + 1):
        sweep_cfg = BatteryConfig(
            capacity_per_battery_kwh=capacity_per_battery,
            power_per_battery_kw=power_per_battery,
            number_of_batteries=n,
            charge_efficiency=charge_eff,
            discharge_efficiency=discharge_eff,
            initial_soc_percent=initial_soc,
            minimum_soc_percent=min_soc,
        )
        if carry_soc:
            sweep_sim = simulate_battery(df, sweep_cfg)
            sweep_view = sweep_sim[(sweep_sim["timestamp"] >= start_ts) & (sweep_sim["timestamp"] < end_ts)].copy()
        else:
            selected = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
            sweep_view = simulate_battery(selected, sweep_cfg)
        sweep_metrics = summarize(
            sweep_view,
            sweep_cfg,
            import_price=import_price,
            export_price=export_price,
            ev_reward_per_kwh=ev_cfg.reward_per_kwh,
        )
        sweep_rows.append(
            {
                "Batteries": n,
                "Capacity kWh": sweep_cfg.total_capacity_kwh,
                "Power kW": sweep_cfg.total_power_kw,
                "Avoided import kWh": sweep_metrics.get("Avoided grid import kWh", 0.0),
                "Grid import after kWh": sweep_metrics.get("Grid import after battery kWh", 0.0),
                "Grid export after kWh": sweep_metrics.get("Grid export after battery kWh", 0.0),
                "Self-consumption after %": sweep_metrics.get("Self-consumption after %", 0.0),
                "Self-sufficiency after %": sweep_metrics.get("Self-sufficiency after %", 0.0),
                "Equivalent full cycles": sweep_metrics.get("Equivalent full cycles", 0.0),
                "Battery value": sweep_metrics.get("Estimated value", 0.0),
                "EV reward": sweep_metrics.get("EV charging reward", 0.0),
                "Total value incl. EV": sweep_metrics.get("Estimated value including EV reward", 0.0),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_df["Batteries"], y=sweep_df["Avoided import kWh"], name="Avoided import (kWh)", mode="lines+markers"))
    fig_sweep.update_layout(
        yaxis_title="Avoided grid import (kWh)",
        xaxis_title="Number of batteries",
        hovermode="x unified",
        height=330,
    )
    st.plotly_chart(fig_sweep, use_container_width=True)
    st.dataframe(sweep_df, use_container_width=True, hide_index=True)

    st.subheader("Detailed metrics")
    detail = pd.DataFrame(
        [
            {"Metric": key, "Value": value}
            for key, value in metrics.items()
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    export_cols = [
        "timestamp",
        "pv_kw",
        "load_kw",
        "ev_charging_kw",
        "non_ev_house_load_kw",
        "baseline_deficit_kw",
        "baseline_surplus_kw",
        "battery_charge_kw",
        "battery_discharge_kw",
        "soc_kwh",
        "soc_percent",
        "grid_import_after_kw",
        "grid_export_after_kw",
        "dt_h",
    ]
    st.download_button(
        "Download simulated selected period as CSV",
        data=view[[col for col in export_cols if col in view.columns]].to_csv(index=False).encode("utf-8"),
        file_name="battery_simulation_selected_period.csv",
        mime="text/csv",
    )

    with st.expander("Column detection and model notes"):
        st.write(
            {
                "timestamp": colmap.timestamp,
                "PV": colmap.pv_kw,
                "load": colmap.load_kw,
                "grid export": colmap.grid_export_kw,
                "grid import": colmap.grid_import_kw,
            }
        )
        st.markdown(
            """
            **Model assumptions**
            - Battery charges only from PV surplus and discharges only to cover house demand.
            - No grid charging or dynamic tariff optimization is included.
            - EV/car charging is inferred from total measured consumption using your threshold and charging-power settings.
            - EV charging is treated as part of the measured load; the battery model still uses the total net surplus/deficit per interval.
            - The extra EV reward is calculated as `inferred EV kWh × reward €/kWh`.
            - Large timestamp gaps are treated as one normal interval to avoid over-counting missing data.
            """
        )


if __name__ == "__main__":
    run_streamlit_app()
