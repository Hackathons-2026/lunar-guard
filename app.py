"""
LUNA-GUARD Full Backend: Integrates NASA JPL Horizons + Scikit-Learn ML Models + Intelligent AI Mission Advisor
Run: python app.py
"""
import os
import re
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
try:
    from astroquery.jplhorizons import Horizons
except ImportError:
    Horizons = None

app = Flask(__name__)

HORIZON_MODEL = joblib.load("models/horizon_rf.pkl") if os.path.exists("models/horizon_rf.pkl") else None
RISK_MODEL = joblib.load("models/risk_rf.pkl") if os.path.exists("models/risk_rf.pkl") else None

CANDIDATE_SITES = [
    {"name": "Shackleton Connecting Ridge", "lat": -89.45, "lon": 136.67, "base": 76, "risk": 2.0, "base_h": 0.4},
    {"name": "de Gerlache Rim",             "lat": -88.90, "lon": -99.90, "base": 74, "risk": 2.0, "base_h": 1.1},
    {"name": "Malapert Massif",             "lat": -85.90, "lon":   0.00, "base": 80, "risk": 1.5, "base_h": 0.8},
    {"name": "Haworth Rim",                 "lat": -87.50, "lon":  -5.00, "base": 62, "risk": 3.0, "base_h": 2.4},
    {"name": "Nobile Rim",                  "lat": -85.20, "lon":  53.30, "base": 72, "risk": 2.5, "base_h": 1.4},
]

EPHEMERIS_CACHE = {}

def get_nasa_ephemeris(lat, lon, start_date_str, hours):
    cache_key = f"{round(lat,2)}_{round(lon,2)}_{start_date_str}_{hours}"
    if cache_key in EPHEMERIS_CACHE:
        return EPHEMERIS_CACHE[cache_key]

    try:
        if Horizons is None:
            raise RuntimeError('astroquery unavailable')
        start_dt = datetime.fromisoformat(start_date_str)
        stop_dt = start_dt + timedelta(hours=hours)
        location = {"lon": lon, "lat": lat, "elevation": 0.0, "body": "301"}
        epochs = {
            "start": start_dt.strftime("%Y-%m-%d %H:%M"),
            "stop": stop_dt.strftime("%Y-%m-%d %H:%M"),
            "step": "1h",
        }
        sun_eph = Horizons(id="10", location=location, epochs=epochs).ephemerides(quantities="4")
        earth_eph = Horizons(id="399", location=location, epochs=epochs).ephemerides(quantities="4")

        res = {
            "source": "NASA JPL Horizons Live API",
            "sun_az": np.array(sun_eph["AZ"][:hours], dtype=float),
            "sun_el": np.array(sun_eph["EL"][:hours], dtype=float),
            "earth_az": np.array(earth_eph["AZ"][:hours], dtype=float),
            "earth_el": np.array(earth_eph["EL"][:hours], dtype=float),
        }
        EPHEMERIS_CACHE[cache_key] = res
        return res
    except Exception:
        # Realistic Lunar South Pole analytical ephemeris:
        # Sun elevation oscillates over the 29.53-day synodic month around polar latitude offset
        t = np.arange(hours, dtype=float)
        doy = datetime.fromisoformat(start_date_str).timetuple().tm_yday
        phase_shift = (doy / 29.53) * 2 * np.pi
        polar_offset = max(0.0, 90.0 + lat) * 0.45
        sun_el = 0.8 + polar_offset * np.sin(2 * np.pi * t / 708.7 + phase_shift) + 1.35 * np.cos(2 * np.pi * t / 708.7 + phase_shift)
        sun_az = (t * 0.508 + lon) % 360.0
        earth_el = 2.5 + 4.8 * np.sin(2 * np.pi * t / 655.7 + phase_shift)
        earth_az = (60.0 + 6.5 * np.cos(2 * np.pi * t / 655.7)) % 360.0
        return {
            "source": "Analytical Lunar Ephemeris Fallback",
            "sun_az": sun_az,
            "sun_el": sun_el,
            "earth_az": earth_az,
            "earth_el": earth_el,
        }

def predict_horizon_and_risk(lat, lon, horizon_override=None):
    if horizon_override is not None and horizon_override != "":
        val = float(horizon_override)
        h24 = [round(val + 0.6 * np.sin(np.radians(d * 2)), 2) for d in range(0, 360, 15)]
    elif HORIZON_MODEL is not None:
        h24 = [round(float(x), 2) for x in HORIZON_MODEL.predict([[lat, lon]])[0]]
    else:
        polar_factor = np.clip((90.0 + lat) / 6.0, 0.0, 1.0)
        base_h = 0.6 + 1.6 * (1.0 - polar_factor)
        h24 = [round(base_h + 1.4 * np.sin(np.radians(d * 2 + lon)), 2) for d in range(0, 360, 15)]

    if RISK_MODEL is not None:
        risk = round(float(RISK_MODEL.predict([[lat, lon]])[0]), 2)
    else:
        polar_factor = np.clip((90.0 + lat) / 6.0, 0.0, 1.0)
        risk = round(1.5 + 1.5 * (1.0 - polar_factor), 2)
    return h24, risk

def simulate_battery_full(sun_el, sun_vis, array_kw, load_kw, dark_load_kw, cap_kwh, init_pct):
    eta = 0.80
    hours = len(sun_vis)
    soc = cap_kwh * (init_pct / 100.0)
    socs = []
    kw_gen = []
    brownout_hours = 0
    first_brownout = -1
    min_soc = soc

    for i in range(hours):
        lit = bool(sun_vis[i])
        g = round(float(array_kw * eta * np.cos(np.radians(min(max(sun_el[i], 0), 10)))), 2) if lit else 0.0
        demand = load_kw if lit else dark_load_kw
        net = g - demand
        if net >= 0:
            soc = min(cap_kwh, soc + net * 0.95)
        else:
            soc += net
            if soc <= 0:
                soc = 0.0
                brownout_hours += 1
                if first_brownout == -1:
                    first_brownout = i
        min_soc = min(min_soc, soc)
        socs.append(round(soc, 2))
        kw_gen.append(g)

    runs = []
    start = -1
    for i in range(hours + 1):
        is_dark = (i < hours) and (not bool(sun_vis[i]))
        if is_dark and start < 0:
            start = i
        elif (not is_dark) and start >= 0:
            soc0 = socs[start - 1] if start > 0 else cap_kwh * (init_pct / 100.0)
            soc1 = socs[i - 1]
            runs.append({"start": start, "len": i - start, "soc0": round(soc0, 1), "soc1": round(soc1, 1)})
            start = -1

    longest = max(runs, key=lambda r: r["len"]) if runs else {"start": 0, "len": 0, "soc0": round(cap_kwh * (init_pct / 100.0), 1), "soc1": round(cap_kwh * (init_pct / 100.0), 1)}
    return {
        "socs": socs,
        "kw_gen": kw_gen,
        "brown": brownout_hours,
        "first": first_brownout,
        "min": round(min_soc, 2),
        "end": round(soc, 2),
        "start": round(cap_kwh * (init_pct / 100.0), 2),
        "gen_total": round(sum(kw_gen), 1),
        "dem_total": round(sum(load_kw if bool(v) else dark_load_kw for v in sun_vis), 1),
        "runs": runs,
        "longest": longest,
        "lit_pct": round(float(np.mean(sun_vis) * 100), 1),
    }

def compute_site_score(base_score, sun_h_per_day, comm_h_per_day, risk, brownout_hours):
    raw = base_score + (sun_h_per_day - 12.0) * 1.4 + (comm_h_per_day - 8.0) * 1.1 - risk * 3.0 - (28.0 if brownout_hours > 0 else 0.0)
    return int(np.clip(round(raw), 10, 98))

def evaluate_all_sites(date_str, days, cap, load, array, init_charge, dark_load):
    m_hrs = days * 24
    ranked = []
    for idx, s in enumerate(CANDIDATE_SITES):
        sh24, srisk = predict_horizon_and_risk(s["lat"], s["lon"], None)
        seph = get_nasa_ephemeris(s["lat"], s["lon"], date_str, m_hrs)
        ssun_v = seph["sun_el"] > np.array([sh24[int(a // 15) % 24] for a in seph["sun_az"]])
        searth_v = seph["earth_el"] > np.array([sh24[int(a // 15) % 24] for a in seph["earth_az"]])
        ssim = simulate_battery_full(seph["sun_el"], ssun_v, array, load, dark_load, cap, init_charge)
        s_sun_h = round(float(np.mean(ssun_v) * 24), 1)
        s_comm_h = round(float(np.mean(searth_v) * 24), 1)
        s_score = compute_site_score(s["base"], s_sun_h, s_comm_h, srisk, ssim["brown"])
        ranked.append({
            "idx": idx,
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "daylight": s_sun_h,
            "comm": s_comm_h,
            "solar": round(ssim["gen_total"] / days, 1),
            "dark": ssim["longest"]["len"],
            "risk": srisk,
            "score": s_score,
            "verdict": "NOT SUITABLE" if ssim["brown"] > 0 else ("MARGINAL" if ssim["min"] < cap * 0.2 else "SUITABLE"),
            "lit_pct": ssim["lit_pct"],
            "backup_h": round(ssim["longest"]["soc0"] / max(dark_load, 0.1), 1),
            "brown": ssim["brown"],
            "first_brown": ssim["first"],
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    p = request.json or {}
    lat = float(p.get("lat", -89.45))
    lon = float(p.get("lon", 136.67))
    date_str = p.get("date", "2026-11-14")
    days = int(p.get("days", 14))
    cap = float(p.get("capacity", 400))
    load = float(p.get("load", 12))
    array = float(p.get("array", 30))
    init_charge = float(p.get("initCharge", 80))
    dark_load = float(p.get("darkLoad", 3))
    horizon_override = p.get("horizonOverride", None)

    h24, risk = predict_horizon_and_risk(lat, lon, horizon_override)
    long_hours = max(days * 24, 2126)
    eph = get_nasa_ephemeris(lat, lon, date_str, long_hours)

    sun_sectors = (eph["sun_az"] // 15).astype(int) % 24
    earth_sectors = (eph["earth_az"] // 15).astype(int) % 24
    sun_mask = np.array([h24[s] for s in sun_sectors])
    earth_mask = np.array([h24[s] for s in earth_sectors])

    sun_vis_all = eph["sun_el"] > sun_mask
    earth_vis_all = eph["earth_el"] > earth_mask

    m_hrs = days * 24
    sim_m = simulate_battery_full(eph["sun_el"][:m_hrs], sun_vis_all[:m_hrs], array, load, dark_load, cap, init_charge)
    sim_l = simulate_battery_full(eph["sun_el"][:2126], sun_vis_all[:2126], array, load, dark_load, cap, init_charge)

    # Exact score for the selected (lat, lon) coordinate (whether preset or clicked on 3D globe!)
    sun_h_day = round(float(np.mean(sun_vis_all[:m_hrs]) * 24), 1)
    comm_h_day = round(float(np.mean(earth_vis_all[:m_hrs]) * 24), 1)
    active_score = compute_site_score(74.0, sun_h_day, comm_h_day, risk, sim_m["brown"])

    surplus = round(array * 0.80 - load, 2)
    if sim_m["brown"] > 0:
        verdict, cls = "NOT SUITABLE", "red"
        why = f"Battery empties after {sim_m['first']} h (day {sim_m['first']//24 + 1}); {sim_m['brown']} h without power in this mission."
    elif sim_l["brown"] > 0 or surplus <= 0 or sim_l["min"] < cap * 0.2:
        verdict, cls = "MARGINAL", "amber"
        why = f"Mission window OK, but reserve drops to {round(sim_l['min']/cap*100)}% over 3 lunar cycles (longest dark {sim_l['longest']['len']} h)."
    else:
        verdict, cls = "SUITABLE", "green"
        why = f"Battery stays above 20% across 3 lunar cycles; longest darkness {sim_l['longest']['len']} h."

    ranked_sites = evaluate_all_sites(date_str, days, cap, load, array, init_charge, dark_load)

    # Real multi-date comparison for the selected (lat, lon)
    date_step = int(p.get("dateStep", 7))
    date_options = []
    base_dt = datetime.fromisoformat(date_str)
    for idx_d in range(6):
        cand_dt = base_dt + timedelta(days=idx_d * date_step)
        cand_str = cand_dt.strftime("%Y-%m-%d")
        c_eph = get_nasa_ephemeris(lat, lon, cand_str, m_hrs)
        c_sun_v = c_eph["sun_el"] > np.array([h24[int(a // 15) % 24] for a in c_eph["sun_az"]])
        c_earth_v = c_eph["earth_el"] > np.array([h24[int(a // 15) % 24] for a in c_eph["earth_az"]])
        c_sim = simulate_battery_full(c_eph["sun_el"], c_sun_v, array, load, dark_load, cap, init_charge)
        c_sun_h = round(float(np.mean(c_sun_v) * 24), 1)
        c_comm_h = round(float(np.mean(c_earth_v) * 24), 1)
        c_score = compute_site_score(74.0, c_sun_h, c_comm_h, risk, c_sim["brown"])
        c_verd = "NOT SUITABLE" if c_sim["brown"] > 0 else ("MARGINAL" if c_sim["min"] < cap * 0.2 else "SUITABLE")
        c_cls = "red" if c_verd == "NOT SUITABLE" else ("amber" if c_verd == "MARGINAL" else "green")
        date_options.append({
            "date": cand_str,
            "sun_h": c_sun_h,
            "comm_h": c_comm_h,
            "solar_kwh": round(c_sim["gen_total"] / days, 0),
            "score": c_score,
            "verdict": c_verd,
            "cls": c_cls,
        })

    return jsonify({
        "data_source": eph["source"],
        "horizon_24": h24,
        "terrain_risk": risk,
        "active_score": active_score,
        "active_sun_h": sun_h_day,
        "active_comm_h": comm_h_day,
        "sun_az": np.round(eph["sun_az"][:m_hrs], 2).tolist(),
        "sun_el": np.round(eph["sun_el"][:m_hrs], 2).tolist(),
        "sun_vis": sun_vis_all[:m_hrs].tolist(),
        "earth_az": np.round(eph["earth_az"][:m_hrs], 2).tolist(),
        "earth_el": np.round(eph["earth_el"][:m_hrs], 2).tolist(),
        "earth_vis": earth_vis_all[:m_hrs].tolist(),
        "sim_m": sim_m,
        "sim_l": {
            "min": sim_l["min"],
            "longest": sim_l["longest"],
            "brown": sim_l["brown"],
        },
        "sustain": {
            "verdict": verdict,
            "cls": cls,
            "why": why,
            "surplus": surplus,
        },
        "ranked_sites": ranked_sites,
        "date_options": date_options,
    })

@app.route("/api/advisor", methods=["POST"])
def api_advisor():
    p = request.json or {}
    question = (p.get("question") or "").strip()
    date_str = p.get("date", "2026-11-14")
    days = int(p.get("days", 14))
    cap = float(p.get("capacity", 400))
    load = float(p.get("load", 12))
    array = float(p.get("array", 30))
    init_charge = float(p.get("initCharge", 80))
    dark_load = float(p.get("darkLoad", 3))

    q_lower = question.lower()
    notes = []

    day_match = re.search(r"(\d+)\s*[- ]?day", q_lower)
    if day_match:
        parsed_days = int(day_match.group(1))
        if 1 <= parsed_days <= 60 and parsed_days != days:
            days = parsed_days
            notes.append(f"Adjusted mission duration to {days} days based on your question.")

    load_match = re.search(r"(\d+(?:\.\d+)?)\s*kw\s*(?:load|power|demand)?", q_lower)
    if load_match:
        parsed_load = float(load_match.group(1))
        if 1 <= parsed_load <= 100 and parsed_load != load:
            load = parsed_load
            notes.append(f"Evaluated with {load} kW base load from your question.")

    cap_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", q_lower)
    if cap_match:
        parsed_cap = float(cap_match.group(1))
        if 50 <= parsed_cap <= 5000:
            cap = parsed_cap
            notes.append(f"Evaluated with {cap} kWh battery capacity from your question.")

    ranked = evaluate_all_sites(date_str, days, cap, load, array, init_charge, dark_load)
    best = ranked[0]
    runner_up = ranked[1]

    lines = []
    if question:
        lines.append(f"QUESTION: {question}")
    if notes:
        lines.append("PARAMETER OVERRIDE: " + " ".join(notes))

    lines.append(f"SCENARIO EVALUATED: {days}-day mission starting {date_str} | Array: {array} kW | Day Load: {load} kW | Dark Load: {dark_load} kW | Battery: {cap} kWh")
    lines.append("")
    lines.append(f"TOP RECOMMENDED SITE: {best['name']} (Score: {best['score']}/100 · Status: {best['verdict']})")
    lines.append(f"  - Illumination: {best['daylight']} h/day ({best['lit_pct']}% sunlit) | Max Darkness: {best['dark']} h (Backup: {best['backup_h']} h)")
    lines.append(f"  - Direct-to-Earth Comms: {best['comm']} h/day | ML Terrain Risk: {best['risk']}/4.0")
    lines.append("")
    lines.append(f"RUNNER-UP OPTION: {runner_up['name']} (Score: {runner_up['score']}/100 · Status: {runner_up['verdict']})")
    lines.append(f"  - Illumination: {runner_up['daylight']} h/day | Comms: {runner_up['comm']} h/day | Max Darkness: {runner_up['dark']} h")
    lines.append("")

    ranking_str = " > ".join([f"{i+1}. {s['name']} ({s['score']}/100, {s['verdict']})" for i, s in enumerate(ranked)])
    lines.append(f"FULL CANDIDATE RANKING ({days} DAYS): {ranking_str}")

    if best["verdict"] == "NOT SUITABLE":
        needed_cap = int(np.ceil(best["dark"] * dark_load * 1.25))
        lines.append(f"ENGINEERING WARNING: Even the top site ({best['name']}) experiences a battery brownout during its {best['dark']}-hour shadow period. Increase battery capacity to at least ~{needed_cap} kWh or reduce dark-mode load below {dark_load} kW.")
    else:
        lines.append(f"RECOMMENDATION: Select {best['name']} for this {days}-day mission. It provides the strongest balance of solar power ({best['solar']} kWh/day), Earth visibility ({best['comm']} h/day), and battery survival.")

    return jsonify({"answer": "\n".join(lines), "best_site_idx": best["idx"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
