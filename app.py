import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw

# =========================
# 1) Constants / Parameters
# =========================
APP_VERSION = "v5.2"
DT_MS = 10.0
FAT_SAT_SCALE = 0.05

MODE_OPTIONS = {
    "Spin Echo (SE)": "SE",
    "Inversion Recovery (IR/FLAIR/STIR)": "IR",
    "Gradient Echo (GRE)": "GRE",
    "Diffusion Weighted (DWI)": "DWI",
}
MODE_LABELS = list(MODE_OPTIONS.keys())

TISSUE_PARAMS = {
    "White Matter": {"t1": 600, "t2": 80, "adc": 0.7},
    "Gray Matter": {"t1": 900, "t2": 100, "adc": 0.9},
    "CSF": {"t1": 4000, "t2": 2000, "adc": 3.0},
    "Infarct": {"t1": 1000, "t2": 110, "adc": 0.3},
    "Fat": {"t1": 250, "t2": 60, "adc": 0.01},
}


# =========================
# 2) Physics Engine
# =========================
def apply_pulse_x(m: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)],
        ]
    )
    return rx @ m


def apply_relaxation_and_offres(
    m: np.ndarray, e1: float, e2: float, df_hz: float, dt_ms: float
) -> np.ndarray:
    phi = 2.0 * np.pi * df_hz * (dt_ms / 1000.0)  # radians
    rz = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    return (rz @ m) * np.array([e2, e2, e1]) + np.array([0, 0, 1 - e1])


def build_sequence_events(
    mode_id: str, tr: int, te: int, ti: int, fa: int, dt_ms: float
):
    """Return RF pulses + 3-axis gradients + readout marker."""
    steps = int(tr / dt_ms)
    rf = np.zeros(steps)
    gx = np.zeros(steps)
    gy = np.zeros(steps)
    gz = np.zeros(steps)
    readout = np.zeros(steps)

    def idx(t_ms: float) -> int:
        return int(np.clip(np.round(t_ms / dt_ms), 0, steps - 1))

    def pulse(arr: np.ndarray, t_ms: float, amp: float, width_steps: int = 1):
        i0 = idx(t_ms)
        i1 = min(i0 + width_steps, steps)
        arr[i0:i1] = amp

    # --- IR ---
    if mode_id == "IR":
        rf[idx(0)] = 180
        rf[idx(ti)] = 90
        rf[idx(ti + te / 2.0)] = 180

        # Slice-select around RFs (simplified)
        pulse(gz, 0, +1.0, 2)
        pulse(gz, 2 * dt_ms, -0.5, 1)
        pulse(gz, ti, +1.0, 2)
        pulse(gz, ti + 2 * dt_ms, -0.5, 1)
        pulse(gz, ti + te / 2.0, +1.0, 2)
        pulse(gz, ti + te / 2.0 + 2 * dt_ms, -0.5, 1)

        t_ro = ti + te
        pulse(gy, t_ro - 3 * dt_ms, +0.7, 1)   # phase-encode
        pulse(gx, t_ro - 2 * dt_ms, -0.8, 1)   # prephaser
        pulse(gx, t_ro - 1 * dt_ms, +1.0, 3)   # readout plateau
        pulse(readout, t_ro, 1.0, 1)

    # --- SE / DWI ---
    elif mode_id in ("SE", "DWI"):
        rf[idx(0)] = 90
        rf[idx(te / 2.0)] = 180

        pulse(gz, 0, +1.0, 2)
        pulse(gz, 2 * dt_ms, -0.5, 1)
        pulse(gz, te / 2.0, +1.0, 2)
        pulse(gz, te / 2.0 + 2 * dt_ms, -0.5, 1)

        t_ro = te
        pulse(gy, t_ro - 3 * dt_ms, +0.7, 1)
        pulse(gx, t_ro - 2 * dt_ms, -0.8, 1)
        pulse(gx, t_ro - 1 * dt_ms, +1.0, 3)
        pulse(readout, t_ro, 1.0, 1)

        if mode_id == "DWI":
            # Simplified MPG pair on x-axis
            pulse(gx, te * 0.20, +1.2, 2)
            pulse(gx, te * 0.70, -1.2, 2)

    # --- GRE ---
    elif mode_id == "GRE":
        rf[idx(0)] = fa
        pulse(gz, 0, +1.0, 2)
        pulse(gz, 2 * dt_ms, -0.5, 1)

        t_ro = te
        pulse(gy, t_ro - 3 * dt_ms, +0.7, 1)
        pulse(gx, t_ro - 2 * dt_ms, -0.8, 1)
        pulse(gx, t_ro - 1 * dt_ms, +1.0, 3)
        pulse(readout, t_ro, 1.0, 1)

    grad = {"gx": gx, "gy": gy, "gz": gz}
    return rf, grad, readout


def simulate_mri(
    mode_id: str,
    tissue_name: str,
    t1: float,
    t2: float,
    adc: float,
    fa: int,
    tr: int,
    te: int,
    ti: int,
    b_value: int,
    fat_sat: bool = False,
    df_hz: float = 2.5,
    dt_ms: float = DT_MS,
):
    steps = int(tr / dt_ms)
    rf_events, grad_events, readout_events = build_sequence_events(mode_id, tr, te, ti, fa, dt_ms)

    mz_steady = 1.0 - np.exp(-tr / t1)
    m = np.array([0.0, 0.0, mz_steady], dtype=float)

    e1 = np.exp(-dt_ms / t1)
    e2 = np.exp(-dt_ms / t2)
    diffusion_decay = np.exp(-b_value * adc * 1e-3) if mode_id == "DWI" else 1.0

    history = []

    for k in range(steps):
        rf_angle = rf_events[k]
        if rf_angle != 0:
            m = apply_pulse_x(m, rf_angle)

        if readout_events[k] > 0:
            if mode_id == "DWI":
                m[:2] *= diffusion_decay
                if fat_sat and tissue_name == "Fat":
                    m[:2] *= FAT_SAT_SCALE
            elif mode_id == "GRE":
                mxy_mag = np.sqrt(m[0] ** 2 + m[1] ** 2)
                m[0], m[1] = mxy_mag, 0.0

        m = apply_relaxation_and_offres(m, e1, e2, df_hz=df_hz, dt_ms=dt_ms)
        history.append(m.copy())

    events = {"rf": rf_events, "grad": grad_events, "readout": readout_events}
    return np.array(history), dt_ms, events


def readout_time_ms(mode_id: str, te: int, ti: int) -> float:
    return ti + te if mode_id == "IR" else te


def get_sample_index(mode_id: str, te: int, ti: int, tr: int, dt: float) -> tuple[float, int]:
    read_time = readout_time_ms(mode_id, te=te, ti=ti)
    read_time = min(read_time, tr - dt)
    return read_time, int(read_time / dt)


# =========================
# 3) UI
# =========================
st.set_page_config(page_title="MRI Physics Lab", layout="wide")

if "ti_slider" not in st.session_state:
    st.session_state.ti_slider = 2300


def set_ti_for_null(target_t1_ms: float):
    st.session_state.ti_slider = int(target_t1_ms * 0.693)  # TI_null ~= T1*ln2


with st.sidebar:
    st.divider()
    st.header("Display Settings")

    img_gain = st.slider("Image Gain (Brightness)", 0.1, 5.0, 1.5, step=0.1)
    gamma = st.slider("Gamma Correction", 0.1, 2.0, 1.0, step=0.1)

    mode_label = st.selectbox("Sequence Mode", MODE_LABELS)
    mode_id = MODE_OPTIONS[mode_label]

    tr = st.slider("TR (ms)", 500, 10000, 4000)
    te = st.slider("TE (ms)", 20, 250, 80)

    fa, ti, b_val, fat_sat = 90, 0, 0, False

    if mode_id == "IR":
        st.write("Tissue Null Presets")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("FLAIR (CSF Null)"):
                set_ti_for_null(TISSUE_PARAMS["CSF"]["t1"])
        with c2:
            if st.button("STIR (Fat Null)"):
                set_ti_for_null(TISSUE_PARAMS["Fat"]["t1"])
        ti = st.slider("TI (ms)", 10, 5000, key="ti_slider")

    elif mode_id == "GRE":
        fa = st.slider("Flip Angle", 5, 90, 30)

    elif mode_id == "DWI":
        b_val = st.slider("b-value", 0, 2000, 1000)
        fat_sat = st.checkbox("Fat Suppression (Fat-Sat)", value=True)

    st.divider()
    t_a = st.selectbox("Tissue A (Blue Vector)", list(TISSUE_PARAMS.keys()), index=3)
    t_b = st.selectbox("Tissue B (Red Vector)", list(TISSUE_PARAMS.keys()), index=4)


# =========================
# 4) Simulation
# =========================
res = {}
for tissue_name, p in TISSUE_PARAMS.items():
    h, dt, ev = simulate_mri(
        mode_id=mode_id,
        tissue_name=tissue_name,
        t1=p["t1"],
        t2=p["t2"],
        adc=p["adc"],
        fa=fa,
        tr=tr,
        te=te,
        ti=ti,
        b_value=b_val,
        fat_sat=fat_sat,
    )
    res[tissue_name] = {
        "h": h,
        "mxy": np.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2),
        "mz": h[:, 2],
        "dt": dt,
        "ev": ev,
    }

read_time, s_idx = get_sample_index(mode_id, te, ti, tr, dt)


def gen_labeled_phantom(res_data, idx, gain=1.0, gamma_value=1.0):
    img = Image.new("RGB", (400, 400), "black")
    draw = ImageDraw.Draw(img)

    def nrm(signal):
        val = np.power(max(0, signal * gain), 1 / gamma_value) * 255
        return int(np.clip(val, 0, 255))

    sigs = {k: nrm(res_data[k]["mxy"][idx]) for k in TISSUE_PARAMS.keys()}

    draw.ellipse([20, 20, 380, 380], fill=(sigs["CSF"],) * 3, outline="gray")
    draw.ellipse([80, 80, 320, 320], fill=(sigs["Gray Matter"],) * 3, outline="gray")
    draw.ellipse([140, 140, 260, 260], fill=(sigs["White Matter"],) * 3, outline="gray")
    draw.ellipse([100, 180, 160, 240], fill=(sigs["Infarct"],) * 3, outline="red", width=3)
    draw.ellipse([240, 180, 300, 240], fill=(sigs["Fat"],) * 3, outline="yellow", width=3)

    labels = [
        (180, 35, "CSF", (255, 120, 120)),
        (140, 95, "Gray Matter", (255, 170, 170)),
        (140, 155, "White Matter", (255, 220, 220)),
        (85, 250, "Infarct", (0, 255, 255)),
        (255, 250, "Fat", (255, 255, 0)),
    ]
    for x, y, txt, col in labels:
        tw = len(txt) * 9
        draw.rectangle([x - 5, y - 2, x + tw, y + 15], fill=(0, 0, 0))
        draw.text((x, y), txt, fill=col)

    return img


# =========================
# 5) Plot
# =========================
col1, col2 = st.columns([3, 1.2])

with col1:
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "scene"}], [{"secondary_y": True}], [{"type": "xy"}]],
        vertical_spacing=0.08,
        row_heights=[0.45, 0.25, 0.3],
    )
    time_ax = np.arange(len(res[t_a]["mxy"])) * dt

    # Bottom panel: Mxy/Mz
    for n, color in [(t_a, "rgb(31, 119, 180)"), (t_b, "rgb(214, 39, 40)")]:
        fig.add_trace(
            go.Scatter(x=time_ax, y=res[n]["mxy"], name=f"{n} Mxy", line=dict(color=color, width=2.5)),
            3,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_ax,
                y=res[n]["mz"],
                name=f"{n} Mz",
                line=dict(color=color, dash="dot", width=1.5),
                showlegend=False,
            ),
            3,
            1,
        )
    fig.add_vline(x=read_time, line_width=2, line_dash="dash", line_color="red", row=3, col=1)

    # Middle panel: RF + Gx/Gy/Gz (+ readout marker)
    ev = res[t_a]["ev"]
    fig.add_trace(go.Bar(x=time_ax, y=ev["rf"], name="RF Pulse", marker_color="orange"), 2, 1, secondary_y=False)

    fig.add_trace(
        go.Scatter(x=time_ax, y=ev["grad"]["gx"], name="Gx (Readout/MPG)", line=dict(color="green", width=2)),
        2,
        1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time_ax, y=ev["grad"]["gy"], name="Gy (Phase)", line=dict(color="purple", width=2)),
        2,
        1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time_ax, y=ev["grad"]["gz"], name="Gz (Slice)", line=dict(color="blue", width=2)),
        2,
        1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=time_ax,
            y=ev["readout"],
            name="Readout",
            line=dict(color="red", width=2, dash="dot"),
        ),
        2,
        1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="RF Angle", range=[-50, 250], secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Gradient Amp", range=[-2.2, 2.2], secondary_y=True, row=2, col=1)

    # Top panel: 3D vectors
    trace_indices = []
    for n, color in [(t_a, "rgb(31, 119, 180)"), (t_b, "rgb(214, 39, 40)")]:
        h = res[n]["h"]

        fig.add_trace(
            go.Scatter3d(
                x=h[:, 0],
                y=h[:, 1],
                z=h[:, 2],
                mode="lines",
                line=dict(color=color.replace("rgb", "rgba").replace(")", ", 0.15)"), width=2),
                showlegend=False,
            ),
            1,
            1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=[0, h[0, 0]],
                y=[0, h[0, 1]],
                z=[0, h[0, 2]],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=color, width=6),
                name=f"{n} Vector",
            ),
            1,
            1,
        )
        trace_indices.append(len(fig.data) - 1)

    animation_steps = np.arange(0, len(res[t_a]["h"]), 1)
    fig.frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=[0, res[n]["h"][k, 0]],
                    y=[0, res[n]["h"][k, 1]],
                    z=[0, res[n]["h"][k, 2]],
                )
                for n in [t_a, t_b]
            ],
            traces=trace_indices,
            name=f"frame{k}",
        )
        for k in animation_steps
    ]

    fig.update_layout(
        height=980,
        scene=dict(camera=dict(eye=dict(x=1.2, y=1.2, z=0.5)), aspectmode="cube"),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="▶ Play from Start",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": False,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Clinical Recipe Guide")
    row1 = st.columns(3)
    with row1[0]:
        st.markdown(
            """
**1. FLAIR (CSF suppression)**
- Sequence Mode: `Inversion Recovery (IR/FLAIR/STIR)`
- Example: TR `9000`, TE `120`, TI `2772`
- Effect: CSF darkens, lesion contrast increases
"""
        )
    with row1[1]:
        st.markdown(
            """
**2. STIR (fat suppression)**
- Sequence Mode: `Inversion Recovery (IR/FLAIR/STIR)`
- Example: TR `4000`, TE `60`, TI `173`
- Effect: fat darkens, edema conspicuous
"""
        )
    with row1[2]:
        st.markdown(
            """
**3. DWI**
- Sequence Mode: `Diffusion Weighted (DWI)`
- Example: TR `4000`, TE `100`, b-value `1000`
- Effect: restricted diffusion remains bright
"""
        )

    row2 = st.columns(3)
    with row2[0]:
        st.markdown(
            """
**4. T1-weighted**
- Sequence Mode: `Spin Echo (SE)`
- Example: TR `500`, TE `15`
- Effect: fat bright, CSF dark
"""
        )
    with row2[1]:
        st.markdown(
            """
**5. T2*-weighted (GRE)**
- Sequence Mode: `Gradient Echo (GRE)`
- Example: TR `600`, TE `25`, FA `20`
- Effect: susceptibility-induced signal loss emphasized
"""
        )
    with row2[2]:
        st.markdown(
            """
**Contrast Tips**
- Signal Delta: larger difference => better separability
- Null Point: tune TI to suppress a specific tissue
"""
        )

with col2:
    st.header("Phantom Preview")
    phantom_img = gen_labeled_phantom(res, s_idx, gain=img_gain, gamma_value=gamma)
    st.image(phantom_img, use_container_width=True)
    st.info(f"🔵 **Tissue A**: {t_a}\n\n🔴 **Tissue B**: {t_b}")
    st.metric("Signal Delta (abs)", f"{abs(res[t_a]['mxy'][s_idx] - res[t_b]['mxy'][s_idx]):.4f}")
    st.write("---")
    st.caption(f"{APP_VERSION} - 3-axis gradient waveform view")