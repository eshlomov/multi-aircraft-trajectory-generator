import io
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

G = 9.80665
EARTH_RADIUS_M = 6_378_137.0
EXCEL_MAX_ROWS = 1_048_576
MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS - 1

PRESETS = pd.DataFrame([
    {
        "aircraft_type": "Multirotor-small",
        "count": 3,
        "speed_min_mps": 5.0,
        "speed_max_mps": 12.0,
        "altitude_m": 100.0,
        "heading_deg": 45.0,
        "profile": "Waypoint-like",
        "max_turn_rate_dps": 12.0,
        "pitch_deg": 2.0,
        "pitch_amplitude_deg": 1.0,
    },
    {
        "aircraft_type": "Multirotor-medium",
        "count": 2,
        "speed_min_mps": 8.0,
        "speed_max_mps": 18.0,
        "altitude_m": 150.0,
        "heading_deg": 120.0,
        "profile": "S-turn",
        "max_turn_rate_dps": 8.0,
        "pitch_deg": 2.0,
        "pitch_amplitude_deg": 1.5,
    },
    {
        "aircraft_type": "Fixed-wing-small",
        "count": 2,
        "speed_min_mps": 15.0,
        "speed_max_mps": 28.0,
        "altitude_m": 250.0,
        "heading_deg": 220.0,
        "profile": "Constant turn",
        "max_turn_rate_dps": 4.0,
        "pitch_deg": 3.0,
        "pitch_amplitude_deg": 1.0,
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
    start_time_utc: datetime,
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

    timestamps = [
        (start_time_utc + pd.to_timedelta(float(sec), unit="s")).isoformat()
        for sec in t
    ]

    return pd.DataFrame({
        "simulation_time_s": t,
        "time_tag_utc": timestamps,
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
    start_lat_deg: float,
    start_lon_deg: float,
    seed: int | None,
):
    rng = np.random.default_rng(seed)
    shared_start_time = datetime.now(timezone.utc).replace(microsecond=0)
    all_tracks = []
    summary_rows = []
    aircraft_number = 0

    for row in aircraft_table.to_dict("records"):
        aircraft_type = str(row["aircraft_type"]).strip()
        count = int(row["count"])
        for _ in range(count):
            aircraft_number += 1
            aircraft_id = f"AC-{aircraft_number:04d}"
            speed = float(rng.uniform(row["speed_min_mps"], row["speed_max_mps"]))
            heading = float(row["heading_deg"]) % 360.0
            pitch = float(row["pitch_deg"])

            track = generate_aircraft_trajectory(
                aircraft_id=aircraft_id,
                aircraft_type=aircraft_type,
                duration_s=duration_s,
                sample_rate_hz=sample_rate_hz,
                start_time_utc=shared_start_time,
                start_lat_deg=start_lat_deg,
                start_lon_deg=start_lon_deg,
                altitude_m=float(row["altitude_m"]),
                speed_mps=speed,
                heading_deg=heading,
                pitch_deg=pitch,
                pitch_amplitude_deg=float(row["pitch_amplitude_deg"]),
                profile=str(row["profile"]),
                max_turn_rate_dps=float(row["max_turn_rate_dps"]),
                rng=rng,
            )
            all_tracks.append(track)
            summary_rows.append({
                "aircraft_id": aircraft_id,
                "aircraft_type": aircraft_type,
                "initial_latitude_deg": start_lat_deg,
                "initial_longitude_deg": start_lon_deg,
                "initial_altitude_m": float(row["altitude_m"]),
                "initial_speed_mps": speed,
                "initial_heading_deg": heading,
                "mean_pitch_deg": pitch,
                "profile": str(row["profile"]),
                "samples": len(track),
                "duration_s": duration_s,
                "shared_start_time_utc": shared_start_time.isoformat(),
            })

    combined = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return combined, summary


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


st.set_page_config(page_title="Multi-Aircraft Trajectory Generator", page_icon="🛩️", layout="wide")
st.title("Multi-Aircraft Synthetic Trajectory Generator")
st.caption("Generate synchronized trajectories for multiple UAV types and export the combined dataset.")

st.subheader("1. Shared simulation settings")
c1, c2, c3 = st.columns(3)
with c1:
    duration = st.number_input("Simulation duration (s)", 1.0, 3600.0, 120.0, 10.0)
with c2:
    sample_rate = st.number_input("Sampling rate (Hz)", 0.1, 100.0, 10.0, 0.5)
with c3:
    seed_input = st.text_input("Random seed (blank = random)", "42")

c4, c5, c6 = st.columns(3)
with c4:
    start_lat = st.number_input("Common origin latitude (deg)", -89.9, 89.9, 32.0853, format="%.6f")
with c5:
    start_lon = st.number_input("Common origin longitude (deg)", -180.0, 180.0, 34.7818, format="%.6f")
with c6:
    st.metric("Samples per aircraft", f"{int(math.floor(duration * sample_rate)) + 1:,}")

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
st.caption("Excel output includes a synchronized master dataset and separate sheets by aircraft type.")
generate = st.button("Generate synchronized multi-aircraft batch", type="primary", use_container_width=True)

if generate:
    required = [
        "aircraft_type", "count", "speed_min_mps", "speed_max_mps", "altitude_m",
        "heading_deg", "profile", "max_turn_rate_dps", "pitch_deg", "pitch_amplitude_deg"
    ]
    if aircraft_table.empty or any(col not in aircraft_table.columns for col in required):
        st.error("Add at least one complete aircraft configuration row.")
        st.stop()
    if aircraft_table[required].isna().any().any():
        st.error("Complete all aircraft configuration fields before generating.")
        st.stop()
    if (aircraft_table["speed_min_mps"] > aircraft_table["speed_max_mps"]).any():
        st.error("For every aircraft type, minimum speed must be less than or equal to maximum speed.")
        st.stop()
    if (aircraft_table["count"] < 1).any():
        st.error("Each aircraft type must have at least one aircraft.")
        st.stop()

    total_aircraft = int(aircraft_table["count"].sum())
    samples_per_aircraft = int(math.floor(duration * sample_rate)) + 1
    total_rows = total_aircraft * samples_per_aircraft
    if total_aircraft > 1000:
        st.error("This first version supports up to 1,000 aircraft per simulation.")
        st.stop()
    if total_rows > 2_000_000:
        st.error(
            f"This batch contains about {total_rows:,} rows. Reduce the duration, sampling rate, "
            "or number of aircraft for this Excel-first version."
        )
        st.stop()

    seed = int(seed_input) if seed_input.strip() else None
    settings = {
        "duration_s": duration,
        "sample_rate_hz": sample_rate,
        "samples_per_aircraft": samples_per_aircraft,
        "total_aircraft": total_aircraft,
        "total_samples": total_rows,
        "common_origin_latitude_deg": start_lat,
        "common_origin_longitude_deg": start_lon,
        "random_seed": seed,
        "time_base": "Shared UTC start time and synchronized elapsed time",
        "coordinate_frame": "Local NED; heading clockwise from north; pitch positive nose-up",
        "model_note": "Kinematic synthetic model; not validated aircraft performance or autopilot simulation",
    }

    with st.spinner("Generating all aircraft on a shared simulation clock..."):
        combined, summary = generate_simultaneous_batch(
            aircraft_table, duration, sample_rate, start_lat, start_lon, seed
        )
        xlsx = make_excel(combined, summary, aircraft_table, settings)

    st.success(
        f"Generated {total_aircraft:,} aircraft across {len(aircraft_table):,} aircraft types, "
        f"with {len(combined):,} synchronized samples."
    )
    a, b, c = st.columns(3)
    a.metric("Aircraft", f"{total_aircraft:,}")
    b.metric("Aircraft types", f"{aircraft_table['aircraft_type'].nunique():,}")
    c.metric("Total data rows", f"{len(combined):,}")
    st.subheader("Combined dataset preview")
    st.dataframe(combined.head(30), use_container_width=True)
    st.download_button(
        "Download multi-aircraft Excel workbook",
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
