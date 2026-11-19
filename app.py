import io
import math
import os
from datetime import date, datetime, timedelta, timezone
from time import monotonic, sleep

try:
    import truststore
except ImportError:
    truststore = None

# Use the operating system's trusted CA store for HTTPS requests when installed.
if truststore is not None:
    truststore.inject_into_ssl()

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

G = 9.80665
EARTH_RADIUS_M = 6_378_137.0
EXCEL_MAX_ROWS = 1_048_576
MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS - 1
FR24_MIN_REQUEST_SPACING_S = 7.0  # stays below the Explorer tier's 10 requests/minute cap

# Approximate geographic bounding boxes (south, north, west, east), in degrees.
# Initial aircraft positions are sampled uniformly inside the selected rectangle.
REGIONS = {
    "Switzerland": {"bounds": (45.80, 47.82, 5.95, 10.50), "center": (46.82, 8.23), "zoom": 6.4},
    "Israel": {"bounds": (29.45, 33.35, 34.25, 35.90), "center": (31.40, 35.00), "zoom": 7.0},
    "Austria": {"bounds": (46.35, 49.05, 9.50, 17.20), "center": (47.60, 13.70), "zoom": 6.0},
    "Germany": {"bounds": (47.25, 55.10, 5.85, 15.10), "center": (51.10, 10.40), "zoom": 5.5},
    "France": {"bounds": (41.30, 51.20, -5.30, 9.80), "center": (46.60, 2.50), "zoom": 5.2},
    "United Kingdom": {"bounds": (49.80, 58.70, -8.20, 1.90), "center": (54.50, -3.00), "zoom": 5.2},
    "United States (continental)": {"bounds": (25.00, 49.50, -124.80, -66.90), "center": (39.50, -98.35), "zoom": 2.8},
}

PRESETS = pd.DataFrame([
    {
        "aircraft_type": "Transportation",
        "fr24_icao_types": "A320,A20N,A321,A21N,B772,B773,B77L,B77W",
        "count": 3,
        "speed_min_mps": 100.0,
        "speed_max_mps": 230.0,
        "altitude_m": 8000.0,
        "heading_deg": 45.0,
        "profile": "Straight",
        "max_turn_rate_dps": 1.5,
        "pitch_deg": 2.0,
        "pitch_amplitude_deg": 0.5,
    },
    {
        "aircraft_type": "Fighter",
        "fr24_icao_types": "",
        "count": 2,
        "speed_min_mps": 170.0,
        "speed_max_mps": 300.0,
        "altitude_m": 150.0,
        "heading_deg": 120.0,
        "profile": "S-turn",
        "max_turn_rate_dps": 16.0,
        "pitch_deg": 5.0,
        "pitch_amplitude_deg": 4.0,
    },
    {
        "aircraft_type": "UAV",
        "fr24_icao_types": "HRM9",
        "count": 2,
        "speed_min_mps": 30.0,
        "speed_max_mps": 55.0,
        "altitude_m": 6000.0,
        "heading_deg": 120.0,
        "profile": "Constant turn",
        "max_turn_rate_dps": 3.0,
        "pitch_deg": 2.0,
        "pitch_amplitude_deg": 1.5,
    },
])


def make_time_axis(duration_s: float, sample_rate_hz: float) -> np.ndarray:
    n = int(math.floor(duration_s * sample_rate_hz)) + 1
    t = np.arange(n, dtype=float) / sample_rate_hz
    if n:
        t[-1] = min(t[-1], duration_s)
    return t


def generate_aircraft_trajectory(
    aircraft_id: str,
    aircraft_type: str,
    duration_s: float,
    sample_rate_hz: float,
    start_epoch_s: float,
    start_lat_deg: float,
    start_lon_deg: float,
    altitude_m: float,
    speed_mps: float,
    heading_deg: float,
    pitch_deg: float,
    pitch_amplitude_deg: float,
    profile: str,
    max_turn_rate_dps: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one synthetic trajectory in a local NED frame.

    Heading is clockwise from north; pitch is positive nose-up; altitude is
    positive upward. This is a kinematic data generator, not a flight-control
    or validated aircraft-performance model.
    """
    t = make_time_axis(duration_s, sample_rate_hz)
    dt = 1.0 / sample_rate_hz
    n = len(t)

    phase = rng.uniform(0, 2 * np.pi)
    if profile == "Straight":
        yaw_rate_dps = np.zeros(n)
    elif profile == "Constant turn":
        sign = rng.choice([-1.0, 1.0])
        yaw_rate_dps = np.full(n, sign * max_turn_rate_dps)
    elif profile == "S-turn":
        yaw_rate_dps = max_turn_rate_dps * np.sin(
            2 * np.pi * t / max(duration_s / 2, dt) + phase
        )
    else:  # Waypoint-like smooth heading changes
        yaw_rate_dps = max_turn_rate_dps * (
            0.65 * np.sin(2 * np.pi * t / max(duration_s, dt) + phase)
            + 0.25 * np.sin(4 * np.pi * t / max(duration_s, dt) + phase / 2)
        )
        yaw_rate_dps = np.clip(yaw_rate_dps, -max_turn_rate_dps, max_turn_rate_dps)

    yaw_rate_rps = np.deg2rad(yaw_rate_dps)
    heading_rad = np.deg2rad(heading_deg) + np.concatenate(
        ([0.0], np.cumsum((yaw_rate_rps[1:] + yaw_rate_rps[:-1]) * 0.5 * dt))
    )

    speed_delta = rng.uniform(0.01, 0.04) * speed_mps
    speed_phase = rng.uniform(0, 2 * np.pi)
    speed = speed_mps + speed_delta * np.sin(
        2 * np.pi * t / max(duration_s, dt) + speed_phase
    )
    pitch_phase = rng.uniform(0, 2 * np.pi)
    pitch = pitch_deg + pitch_amplitude_deg * np.sin(
        2 * np.pi * t / max(duration_s, dt) + pitch_phase
    )

    pitch_rad = np.deg2rad(pitch)
    v_n = speed * np.cos(pitch_rad) * np.cos(heading_rad)
    v_e = speed * np.cos(pitch_rad) * np.sin(heading_rad)
    v_d = -speed * np.sin(pitch_rad)

    north = np.zeros(n)
    east = np.zeros(n)
    down = np.zeros(n)
    if n > 1:
        north[1:] = np.cumsum((v_n[1:] + v_n[:-1]) * 0.5 * dt)
        east[1:] = np.cumsum((v_e[1:] + v_e[:-1]) * 0.5 * dt)
        down[1:] = np.cumsum((v_d[1:] + v_d[:-1]) * 0.5 * dt)

    lat0 = np.deg2rad(start_lat_deg)
    latitude = start_lat_deg + np.rad2deg(north / EARTH_RADIUS_M)
    lon_scale = EARTH_RADIUS_M * max(abs(np.cos(lat0)), 1e-6)
    longitude = start_lon_deg + np.rad2deg(east / lon_scale)
    altitude = altitude_m - down

    horizontal_speed = np.maximum(speed * np.cos(pitch_rad), 0.1)
    roll = np.rad2deg(np.arctan(horizontal_speed * yaw_rate_rps / G))

    # Unix Epoch seconds (UTC). Keep fractional seconds for sub-second sampling rates.
    epoch_tags = start_epoch_s + t

    return pd.DataFrame({
        "simulation_time_s": t,
        "time_tag_epoch_s": epoch_tags,
        "aircraft_id": aircraft_id,
        "aircraft_type": aircraft_type,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "altitude_m": altitude,
        "north_m": north,
        "east_m": east,
        "speed_mps": speed,
        "heading_deg": np.mod(np.rad2deg(heading_rad), 360.0),
        "pitch_deg": pitch,
        "roll_deg": roll,
        "yaw_rate_deg_s": yaw_rate_dps,
    })


def generate_simultaneous_batch(
    aircraft_table: pd.DataFrame,
    duration_s: float,
    sample_rate_hz: float,
    region_name: str,
    seed: int | None,
    start_slot_interval_s: float,
):
    rng = np.random.default_rng(seed)
    shared_start_epoch_s = datetime.now(timezone.utc).timestamp()
    south, north, west, east = REGIONS[region_name]["bounds"]
    all_tracks = []
    summary_rows = []
    aircraft_number = 0
    class_start_slots = {}
    class_counts = {}
    for row in aircraft_table.to_dict("records"):
        aircraft_type = str(row["aircraft_type"]).strip()
        class_counts[aircraft_type] = class_counts.get(aircraft_type, 0) + int(row["count"])
    for aircraft_type, class_count in class_counts.items():
        slots = np.arange(class_count, dtype=float) * start_slot_interval_s
        rng.shuffle(slots)
        class_start_slots[aircraft_type] = list(slots)

    for row in aircraft_table.to_dict("records"):
        aircraft_type = str(row["aircraft_type"]).strip()
        count = int(row["count"])
        for _ in range(count):
            aircraft_number += 1
            aircraft_id = f"AC-{aircraft_number:04d}"
            start_delay_s = float(class_start_slots[aircraft_type].pop())
            aircraft_start_epoch_s = shared_start_epoch_s + start_delay_s
            speed = float(rng.uniform(row["speed_min_mps"], row["speed_max_mps"]))
            # Give every aircraft its own initial location within the selected region.
            start_lat = float(rng.uniform(south, north))
            start_lon = float(rng.uniform(west, east))
            heading = float(row["heading_deg"]) % 360.0
            pitch = float(row["pitch_deg"])

            track = generate_aircraft_trajectory(
                aircraft_id=aircraft_id,
                aircraft_type=aircraft_type,
                duration_s=duration_s,
                sample_rate_hz=sample_rate_hz,
                start_epoch_s=aircraft_start_epoch_s,
                start_lat_deg=start_lat,
                start_lon_deg=start_lon,
                altitude_m=float(row["altitude_m"]),
                speed_mps=speed,
                heading_deg=heading,
                pitch_deg=pitch,
                pitch_amplitude_deg=float(row["pitch_amplitude_deg"]),
                profile=str(row["profile"]),
                max_turn_rate_dps=float(row["max_turn_rate_dps"]),
                rng=rng,
            )
            track["aircraft_elapsed_s"] = track["simulation_time_s"]
            track["start_delay_s"] = start_delay_s
            track["simulation_time_s"] += start_delay_s
            all_tracks.append(track)
            summary_rows.append({
                "aircraft_id": aircraft_id,
                "aircraft_type": aircraft_type,
                "start_delay_s": start_delay_s,
                "aircraft_start_epoch_s": aircraft_start_epoch_s,
                "initial_latitude_deg": start_lat,
                "initial_longitude_deg": start_lon,
                "initial_altitude_m": float(row["altitude_m"]),
                "initial_speed_mps": speed,
                "initial_heading_deg": heading,
                "mean_pitch_deg": pitch,
                "profile": str(row["profile"]),
                "samples": len(track),
                "duration_s": duration_s,
                "shared_start_epoch_s": shared_start_epoch_s,
            })

    combined = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    if not combined.empty:
        combined.sort_values(
            ["time_tag_epoch_s", "aircraft_id"], kind="stable", inplace=True
        )
        combined.reset_index(drop=True, inplace=True)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.sort_values(
            ["aircraft_start_epoch_s", "aircraft_id"], kind="stable", inplace=True
        )
        summary.reset_index(drop=True, inplace=True)
    return combined, summary


def get_fr24_token() -> tuple[str | None, str | None]:
    """Read an FR24 token and report its source without exposing its value."""
    try:
        token = st.secrets.get("FR24_API_TOKEN")
    except Exception:
        token = None
    if token:
        return token, "Streamlit secrets"
    token = os.getenv("FR24_API_TOKEN")
    if token:
        return token, "environment variable"
    return None, None


def fetch_fr24_historical_positions(
    start_epoch_s: int,
    duration_s: int,
    interval_s: int,
    region_name: str,
    class_type_codes: dict[str, set[str]],
    api_token: str,
) -> pd.DataFrame:
    """Fetch sampled historical positions and retain configured ICAO type codes."""
    south, north, west, east = REGIONS[region_name]["bounds"]
    bounds = f"{north},{south},{west},{east}"
    headers = {
        "Accept": "application/json",
        "Accept-Version": "v1",
        "Authorization": f"Bearer {api_token}",
    }
    rows = []
    request_epochs = range(start_epoch_s, start_epoch_s + duration_s + 1, interval_s)
    last_request_at = None

    for requested_epoch in request_epochs:
        for attempt in range(3):
            if last_request_at is not None:
                elapsed = monotonic() - last_request_at
                sleep(max(0.0, FR24_MIN_REQUEST_SPACING_S - elapsed))
            response = requests.get(
                "https://fr24api.flightradar24.com/api/historic/flight-positions/full",
                headers=headers,
                # Explorer permits up to 20 results per request; this works across tiers.
                params={"timestamp": requested_epoch, "bounds": bounds, "limit": 20},
                timeout=30,
            )
            last_request_at = monotonic()
            if response.status_code == 429:
                if attempt >= 1:
                    raise ValueError(
                        "FR24 rate-limited the request twice (HTTP 429). Wait before trying again, "
                        "or check the request-per-minute limit for your API plan."
                    )
                try:
                    retry_after_s = float(response.headers.get("Retry-After", "60"))
                except (TypeError, ValueError):
                    retry_after_s = 60.0
                sleep(max(retry_after_s, FR24_MIN_REQUEST_SPACING_S))
                continue
            if response.status_code in {500, 502, 503, 504}:
                if attempt == 2:
                    raise ValueError(
                        f"FR24 remained unavailable after 3 attempts (HTTP {response.status_code}). "
                        "Wait a few minutes and try again."
                    )
                try:
                    retry_after_s = float(response.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after_s = 2.0 ** (attempt + 1)
                sleep(max(retry_after_s, FR24_MIN_REQUEST_SPACING_S))
                continue
            break
        if response.status_code == 401:
            final_auth_header = response.request.headers.get("Authorization", "")
            if final_auth_header.startswith("Bearer ") and len(final_auth_header) > len("Bearer "):
                raise ValueError(
                    "The app sent a Bearer Authorization header, but FR24 reports that no API key arrived. "
                    "A redirect or network proxy may be stripping the header. Check the final request URL "
                    "and proxy/network settings; the token value was not logged."
                )
            raise ValueError(
                "The final HTTP request did not contain a Bearer Authorization header. "
                "Restart the current app from this project folder and retry."
            )
        if response.status_code == 403:
            raise ValueError("FR24 denied this endpoint. Check that the token has API access.")
        if response.status_code == 400:
            try:
                error_payload = response.json()
                detail = error_payload.get("details") or error_payload.get("message")
            except (ValueError, AttributeError):
                detail = None
            suffix = f" FR24 says: {detail}" if detail else " Check the timestamp, bounds, and result limit."
            raise ValueError(f"FR24 rejected the request (HTTP 400).{suffix}")
        response.raise_for_status()
        payload = response.json()
        records = payload if isinstance(payload, list) else (payload.get("data") or [])
        if not isinstance(records, list):
            raise ValueError("FR24 returned an unexpected response format.")

        for record in records:
            type_code = str(record.get("type") or record.get("aircraft_type") or "").upper()
            aircraft_class = next(
                (name for name, codes in class_type_codes.items() if type_code in codes),
                None,
            )
            if not aircraft_class:
                continue
            latitude = record.get("lat", record.get("latitude"))
            longitude = record.get("lon", record.get("longitude"))
            if latitude is None or longitude is None:
                continue
            observed_epoch = record.get("timestamp", requested_epoch)
            try:
                observed_epoch = int(observed_epoch)
            except (TypeError, ValueError):
                observed_epoch = requested_epoch
            rows.append({
                **record,
                "simulation_time_s": observed_epoch - start_epoch_s,
                "time_tag_epoch_s": observed_epoch,
                "aircraft_id": str(record.get("fr24_id") or record.get("hex") or record.get("callsign") or "Unknown"),
                "aircraft_type": aircraft_class,
                "fr24_type_code": type_code,
                "latitude_deg": float(latitude),
                "longitude_deg": float(longitude),
            })

    return pd.DataFrame(rows)


def limit_fr24_aircraft_to_configured_counts(
    tracks: pd.DataFrame, aircraft_table: pd.DataFrame
) -> pd.DataFrame:
    """Keep up to the configured number of best-sampled FR24 flights per class."""
    if tracks.empty:
        return tracks

    counts_by_class = (
        aircraft_table.groupby("aircraft_type")["count"].sum().astype(int).to_dict()
    )
    selected = []
    for aircraft_class, class_tracks in tracks.groupby("aircraft_type", sort=False):
        limit = counts_by_class.get(aircraft_class, 0)
        if limit <= 0:
            continue
        samples_per_flight = class_tracks.groupby("aircraft_id").size()
        selected_ids = (
            samples_per_flight.rename("samples")
            .reset_index()
            .sort_values(["samples", "aircraft_id"], ascending=[False, True])
            .head(limit)["aircraft_id"]
        )
        selected.append(class_tracks[class_tracks["aircraft_id"].isin(selected_ids)])

    return pd.concat(selected, ignore_index=True) if selected else tracks.iloc[0:0].copy()


def make_excel(combined: pd.DataFrame, summary: pd.DataFrame, aircraft_table: pd.DataFrame,
               settings: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Aircraft_Summary"
    for row in [list(summary.columns)] + summary.values.tolist():
        ws.append(row)

    ws_types = wb.create_sheet("Aircraft_Configurations")
    for row in [list(aircraft_table.columns)] + aircraft_table.values.tolist():
        ws_types.append([None if pd.isna(v) else v for v in row])

    ws_settings = wb.create_sheet("Simulation_Settings")
    for row in [("Parameter", "Value")] + list(settings.items()):
        ws_settings.append(list(row))

    # One synchronized master dataset, split into sheets if Excel's row limit is reached.
    total = len(combined)
    n_sheets = max(1, math.ceil(total / MAX_DATA_ROWS_PER_SHEET))
    for k in range(n_sheets):
        chunk = combined.iloc[k * MAX_DATA_ROWS_PER_SHEET:(k + 1) * MAX_DATA_ROWS_PER_SHEET]
        ws_data = wb.create_sheet(f"All_Aircraft_{k + 1}")
        ws_data.append(list(chunk.columns))
        for row in chunk.itertuples(index=False, name=None):
            ws_data.append(list(row))

    # Per-aircraft-type sheets are convenient for analysis; the master sheet remains synchronized.
    for aircraft_type in combined["aircraft_type"].drop_duplicates():
        subset = combined[combined["aircraft_type"] == aircraft_type]
        safe_name = "".join(c for c in str(aircraft_type) if c.isalnum() or c in "_-")[:24]
        sheet_name = f"Type_{safe_name}"[:31]
        if sheet_name not in wb.sheetnames:
            ws_type = wb.create_sheet(sheet_name)
            ws_type.append(list(subset.columns))
            for row in subset.itertuples(index=False, name=None):
                ws_type.append(list(row))

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
        for col in sheet.columns:
            letter = get_column_letter(col[0].column)
            samples = [str(c.value) for c in list(col)[:150] if c.value is not None]
            width = min(max(max((len(x) for x in samples), default=12) + 2, 12), 30)
            sheet.column_dimensions[letter].width = width

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


st.set_page_config(page_title="AeroGen", page_icon="🛩️", layout="wide")
st.title("AeroGen - Aircraft Trajectory Generator")
st.caption("Generate synthetic aircraft trajectories or fetch historical FR24 positions, visualize the tracks, and export the selected dataset.")

st.subheader("1. Shared simulation settings")
c1, c2, c3 = st.columns(3)
with c1:
    duration = st.number_input("Simulation duration (s)", 1.0, 3600.0, 600.0, 10.0)
with c2:
    sample_rate = st.number_input("Sampling rate (Hz)", 0.1, 100.0, 0.5, 0.5)
with c3:
    seed_input = st.text_input("Random seed (blank = random)", "42")

c4, c5, c6 = st.columns(3)
with c4:
    region_name = st.selectbox("Geographical region", list(REGIONS.keys()), index=0)
with c5:
    st.metric("Samples per aircraft", f"{int(math.floor(duration * sample_rate)) + 1:,}")
with c6:
    st.caption("Time tag format: Epoch time")
start_slot_interval_s = st.number_input(
    "Synthetic aircraft start-slot interval (s)",
    min_value=1,
    max_value=600,
    value=30,
    step=1,
    help="Aircraft of the same type get distinct randomized start slots. Each keeps the selected flight duration.",
)
region_info = REGIONS[region_name]
st.caption(
    f"Initial aircraft positions are randomized inside the approximate {region_name} region "
    f"(latitude {region_info['bounds'][0]:.2f}–{region_info['bounds'][1]:.2f}°, "
    f"longitude {region_info['bounds'][2]:.2f}–{region_info['bounds'][3]:.2f}°). "
    "This is a rectangular bounding area, not an exact national-border polygon."
)

st.subheader("2. Aircraft type configuration")
st.write(
    "Each row represents an aircraft class. Set how many individual aircraft of that type "
    "should participate in the same synchronized simulation. You can edit rows and add more."
)
aircraft_table = st.data_editor(
    PRESETS,
    num_rows="dynamic",
    use_container_width=True,
    key="aircraft_config_editor",
    column_config={
        "aircraft_type": st.column_config.TextColumn("Aircraft type"),
        "fr24_icao_types": st.column_config.TextColumn(
            "FR24 ICAO type codes (comma separated)",
            help="Map this class to FR24 aircraft codes, such as C172 or B738.",
        ),
        "count": st.column_config.NumberColumn("Number of aircraft", min_value=1, max_value=1000, step=1),
        "speed_min_mps": st.column_config.NumberColumn("Min speed (m/s)", min_value=0.1),
        "speed_max_mps": st.column_config.NumberColumn("Max speed (m/s)", min_value=0.1),
        "altitude_m": st.column_config.NumberColumn("Initial altitude (m)"),
        "heading_deg": st.column_config.NumberColumn("Initial heading (deg)", min_value=0.0, max_value=359.9),
        "profile": st.column_config.SelectboxColumn("Flight profile", options=["Straight", "Constant turn", "S-turn", "Waypoint-like"]),
        "max_turn_rate_dps": st.column_config.NumberColumn("Max turn rate (deg/s)", min_value=0.0),
        "pitch_deg": st.column_config.NumberColumn("Mean pitch (deg)", min_value=-30.0, max_value=30.0),
        "pitch_amplitude_deg": st.column_config.NumberColumn("Pitch variation (deg)", min_value=0.0, max_value=15.0),
    },
)

st.subheader("3. Output")
extract_fr24 = st.checkbox("Use historical FR24 data instead of synthetic data", value=False)
st.caption(
    "When unchecked, the app generates synthetic data. When checked, it fetches historical FR24 data only."
)
fr24_token, fr24_token_source = get_fr24_token() if extract_fr24 else (None, None)
if extract_fr24:
    if fr24_token:
        st.success(
            f"FR24 token detected in {fr24_token_source}; its value is hidden. "
            "The app will add it to the Authorization: Bearer header when you extract."
        )
    else:
        st.error(
            "FR24 token not found in this Streamlit process. Check .streamlit/secrets.toml "
            "or the FR24_API_TOKEN environment variable."
        )
    st.caption(
        "Historical positions are queried within the selected region and time window. "
        "Times below are UTC and are sent as Unix Epoch seconds. Add FR24 ICAO type codes "
        "to each class above to match aircraft."
    )
    fr24_col1, fr24_col2, fr24_col3 = st.columns(3)
    with fr24_col1:
        default_fr24_start = datetime.now(timezone.utc) - timedelta(seconds=duration)
        fr24_date = st.date_input("FR24 date (UTC)", value=default_fr24_start.date())
    with fr24_col2:
        fr24_time = st.time_input(
            "FR24 start time (UTC)",
            value=default_fr24_start.time().replace(second=0, microsecond=0),
        )
    with fr24_col3:
        fr24_interval_s = st.number_input(
            "FR24 query interval (seconds)", min_value=10, max_value=3600,
            value=60, step=10,
        )
    st.warning(
        "FR24 historical extraction makes one API request per interval and may consume API credits. "
        "The app spaces requests to stay under the Explorer plan's 10 requests/minute limit; "
        "longer extractions may take several minutes."
    )
generate = st.button(
    "Fetch historical FR24 data" if extract_fr24 else "Generate synthetic trajectories",
    type="primary",
    use_container_width=True,
)

if generate:
    synthetic_required = [
        "aircraft_type", "count", "speed_min_mps", "speed_max_mps", "altitude_m",
        "heading_deg", "profile", "max_turn_rate_dps", "pitch_deg", "pitch_amplitude_deg"
    ]
    required = (
        ["aircraft_type", "count", "fr24_icao_types"]
        if extract_fr24
        else synthetic_required
    )
    if aircraft_table.empty or any(col not in aircraft_table.columns for col in required):
        st.error("Add at least one complete aircraft configuration row.")
        st.stop()
    if aircraft_table[required].isna().any().any():
        st.error("Complete all aircraft configuration fields before generating.")
        st.stop()
    if not extract_fr24 and (aircraft_table["speed_min_mps"] > aircraft_table["speed_max_mps"]).any():
        st.error("For every aircraft type, minimum speed must be less than or equal to maximum speed.")
        st.stop()
    if (aircraft_table["count"] < 1).any():
        st.error("Each aircraft type must have at least one aircraft.")
        st.stop()

    total_aircraft = int(aircraft_table["count"].sum())
    if total_aircraft > 1000:
        st.error("This first version supports up to 1,000 aircraft per simulation.")
        st.stop()
    samples_per_aircraft = int(math.floor(duration * sample_rate)) + 1
    total_rows = total_aircraft * samples_per_aircraft
    if not extract_fr24 and total_rows > 2_000_000:
        st.error(
            f"This batch contains about {total_rows:,} rows. Reduce the duration, sampling rate, "
            "or number of aircraft for this Excel-first version."
        )
        st.stop()

    seed = int(seed_input) if not extract_fr24 and seed_input.strip() else None
    class_type_codes = {}
    fr24_start_epoch = None
    if extract_fr24:
        for row in aircraft_table.to_dict("records"):
            aircraft_class = str(row["aircraft_type"]).strip()
            codes = {
                code.strip().upper()
                for code in str(row.get("fr24_icao_types", "")).split(",")
                if code.strip()
            }
            if codes:
                class_type_codes[aircraft_class] = codes
        if not class_type_codes:
            st.error("Enter at least one FR24 ICAO type code in the aircraft configuration table.")
            st.stop()
        request_count = math.floor(duration / fr24_interval_s) + 1
        if request_count > 100:
            st.error("This FR24 request would make over 100 API calls. Increase the query interval or shorten the duration.")
            st.stop()
        if fr24_date < date(2019, 11, 18) and fr24_interval_s < 60:
            st.error("FR24 historical positions before 2019-11-18 have minute resolution. Set the query interval to at least 60 seconds.")
            st.stop()
        if not fr24_token:
            st.error(
                "FR24 API token not found. Set FR24_API_TOKEN in Streamlit secrets or the environment."
            )
            st.stop()
        fr24_start_epoch = int(
            datetime.combine(fr24_date, fr24_time, tzinfo=timezone.utc).timestamp()
        )
        if fr24_start_epoch + int(round(duration)) > int(datetime.now(timezone.utc).timestamp()):
            st.error("The selected FR24 time window ends in the future. Choose a start time far enough in the past for the selected duration.")
            st.stop()
    settings = {
        "duration_s": duration,
        "sample_rate_hz": sample_rate,
        "samples_per_aircraft": samples_per_aircraft,
        "synthetic_start_slot_interval_s": start_slot_interval_s,
        "overall_batch_duration_s": duration + max(
            max(0, int(aircraft_table.groupby("aircraft_type")["count"].sum().max()) - 1)
            * start_slot_interval_s,
            0,
        ),
        "total_aircraft": total_aircraft,
        "total_samples": total_rows,
        "geographical_region": region_name,
        "region_bounds_lat_lon": str(REGIONS[region_name]["bounds"]),
        "random_seed": seed,
        "time_base": "Shared UTC start time and synchronized elapsed time",
        "coordinate_frame": "Local NED; heading clockwise from north; pitch positive nose-up",
        "model_note": "Kinematic synthetic model; not validated aircraft performance or autopilot simulation",
    }

    combined = pd.DataFrame()
    fr24_tracks = pd.DataFrame()
    if extract_fr24:
        try:
            with st.spinner("Requesting historical positions from FR24..."):
                fr24_tracks = fetch_fr24_historical_positions(
                    fr24_start_epoch,
                    int(round(duration)),
                    int(fr24_interval_s),
                    region_name,
                    class_type_codes,
                    fr24_token,
                )
                fr24_tracks = limit_fr24_aircraft_to_configured_counts(
                    fr24_tracks, aircraft_table
                )
        except requests.RequestException as exc:
            if isinstance(exc, requests.exceptions.SSLError):
                if truststore is None:
                    st.error(
                        "FR24 TLS certificate verification failed. Install the updated project "
                        "dependencies in the same Python environment that runs Streamlit "
                        "(`python -m pip install -r requirements.txt`) so it can use the "
                        "system certificate store."
                    )
                else:
                    st.error(
                        "FR24 TLS certificate verification failed even with the system "
                        "certificate store. If your network uses HTTPS inspection, install "
                        "your organization’s trusted root CA or set REQUESTS_CA_BUNDLE to its PEM file."
                    )
            else:
                st.error(f"FR24 request failed: {exc}")
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    else:
        with st.spinner("Generating aircraft trajectories..."):
            combined, summary = generate_simultaneous_batch(
                aircraft_table, duration, sample_rate, region_name, seed,
                start_slot_interval_s,
            )
            xlsx = make_excel(combined, summary, aircraft_table, settings)

    a, b, c = st.columns(3)
    if extract_fr24:
        number_of_flights = (
            fr24_tracks["aircraft_id"].nunique() if not fr24_tracks.empty else 0
        )
        st.success(
            f"Fetched {number_of_flights:,} FR24 flights with "
            f"{len(fr24_tracks):,} historical positions."
        )
        a.metric("FR24 flights", f"{number_of_flights:,}")
        b.metric("Aircraft classes", f"{fr24_tracks['aircraft_type'].nunique() if not fr24_tracks.empty else 0:,}")
        c.metric("FR24 positions", f"{len(fr24_tracks):,}")
    else:
        st.success(
            f"Generated {total_aircraft:,} synthetic aircraft across "
            f"{len(aircraft_table):,} aircraft types, with {len(combined):,} samples."
        )
        st.caption(
            f"Aircraft of each type receive distinct randomized start slots, spaced "
            f"{start_slot_interval_s:g} seconds apart; each keeps the full {duration:g}-second duration. "
            "The master dataset and Excel sheets are ordered by timestamp."
        )
        a.metric("Aircraft", f"{total_aircraft:,}")
        b.metric("Aircraft types", f"{aircraft_table['aircraft_type'].nunique():,}")
        c.metric("Total data rows", f"{len(combined):,}")
    st.subheader("Airsky Picture")
    map_tracks = fr24_tracks if extract_fr24 else combined
    if extract_fr24:
        st.caption("Historical FR24 positions matched to your configured aircraft classes.")
        if fr24_tracks.empty:
            st.warning("FR24 returned no positions matching the selected time, region, and aircraft type codes.")
        else:
            st.success(
                f"Showing {fr24_tracks['aircraft_id'].nunique():,} FR24 flights "
                f"({len(fr24_tracks):,} positions), limited by the configured count for each class."
            )
    else:
        st.caption("Interactive geographic map. Each colored line represents one aircraft type.")
    fig = go.Figure()
    class_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    aircraft_types = map_tracks["aircraft_type"].drop_duplicates().tolist() if not map_tracks.empty else []
    class_colors = {
        aircraft_type: class_palette[index % len(class_palette)]
        for index, aircraft_type in enumerate(aircraft_types)
    }
    shown_classes = set()
    # Downsample only the visualization; the Excel export retains every generated sample.
    max_points_per_track = 300
    track_groups = map_tracks.groupby("aircraft_id", sort=True) if not map_tracks.empty else []
    for aircraft_id, track in track_groups:
        aircraft_type = track["aircraft_type"].iloc[0]
        color = class_colors[aircraft_type]
        step = max(1, int(math.ceil(len(track) / max_points_per_track)))
        display_track = track.iloc[::step]
        if display_track.index[-1] != track.index[-1]:
            display_track = pd.concat([display_track, track.tail(1)])
        fig.add_trace(go.Scattermap(
            lat=display_track["latitude_deg"],
            lon=display_track["longitude_deg"],
            mode="lines",
            name=aircraft_type,
            legendgroup=aircraft_type,
            showlegend=aircraft_type not in shown_classes,
            line={"width": 2, "color": color},
            hovertemplate=(
                f"{aircraft_id} ({aircraft_type})<br>Lat: %{{lat:.5f}}<br>"
                "Lon: %{lon:.5f}<extra></extra>"
            ),
        ))
        shown_classes.add(aircraft_type)
        # Mark the initial position of each aircraft.
        fig.add_trace(go.Scattermap(
            lat=[track["latitude_deg"].iloc[0]],
            lon=[track["longitude_deg"].iloc[0]],
            mode="markers",
            name=f"{aircraft_id} start ({aircraft_type})",
            legendgroup=aircraft_type,
            marker={"size": 7, "color": color},
            showlegend=False,
            hovertemplate=f"{aircraft_id} start ({aircraft_type})<extra></extra>",
        ))
    fig.update_layout(
        map={
            "style": "open-street-map",
            "center": {"lat": region_info["center"][0], "lon": region_info["center"][1]},
            "zoom": region_info["zoom"],
        },
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=650,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12},
    )
    if map_tracks.empty:
        st.info("No trajectories are available to plot for this selection.")
    else:
        st.plotly_chart(fig, use_container_width=True)
    if extract_fr24:
        st.subheader("FR24 dataset preview")
        st.dataframe(fr24_tracks.head(30), use_container_width=True)
        if not fr24_tracks.empty:
            st.download_button(
                "Download FR24 positions as CSV",
                data=fr24_tracks.to_csv(index=False).encode("utf-8"),
                file_name="fr24_historical_positions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.subheader("Synthetic dataset preview")
        st.dataframe(combined.head(30), use_container_width=True)
        st.download_button(
            "Download synthetic Excel workbook",
            data=xlsx,
            file_name="multi_aircraft_trajectory_batch.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
else:
    st.info(
        "Configure multiple aircraft types above. All generated aircraft will share the same "
        "simulation start time and sampling clock, while each aircraft receives its own trajectory."
    )
