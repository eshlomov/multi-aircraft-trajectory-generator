MULTI-AIRCRAFT UAV TRAJECTORY GENERATOR - VERSION 2

Files to upload to the ROOT of your GitHub repository:
  app.py
  requirements.txt

Streamlit deployment settings:
  Repository: your GitHub repository
  Branch: main (or your default branch)
  Main file path: app.py

Run locally:
  python -m pip install -r requirements.txt
  streamlit run app.py

Version 2 features:
- Select a geographic region (Switzerland, Israel, Austria, Germany, France,
  United Kingdom, or continental United States).
- Randomly places each aircraft start point within that region's approximate
  rectangular latitude/longitude bounds.
- Uses a shared UTC start time and exports Unix Epoch timestamps in seconds.
- Generates multiple aircraft types simultaneously.
- Displays all paths on an interactive map.
- Exports synchronized trajectory data and aircraft-type sheets to Excel.

Important limitations:
- Region bounds are rectangles, not exact country borders. Some sampled
  starting points may fall outside the actual country or over water.
- The map is a visualization of synthetic kinematic paths, not a flight
  authorization, route-safety, terrain-clearance, or airspace-compliance tool.
- The model is not validated aircraft performance or autopilot simulation.
