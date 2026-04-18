import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw

# --- 1. 組織データ定義 ---
TISSUE_PARAMS = {
    "White Matter": {"t1": 600, "t2": 80, "adc": 0.7},
    "Gray Matter": {"t1": 900, "t2": 100, "adc": 0.9},
    "CSF": {"t1": 4000, "t2": 2000, "adc": 3.0},
    "Infarct": {"t1": 1000, "t2": 110, "adc": 0.3},
    "Fat": {"t1": 250, "t2": 60, "adc": 0.01}
}

# --- 2. 物理エンジン ---
def simulate_mri(mode, t1, t2, adc, fa, tr, te, ti, b_value, fat_sat=False, df=2.5):
    """
    MRI物理シミュレーションエンジン (v5.1)
    - TR飽和効果によるT1コントラストの再現
    - DWIモードにおける拡散減衰と脂肪抑制(Fat-Sat)の実装
    """
    dt = 10.0  # シミュレーションの刻み時間 (ms)
    steps = int(tr / dt)
    
    # --- 1. 定常状態 (Steady State) の縦磁化を初期値に設定 ---
    # TRが短いと磁化が回復しきらない（T1強調の原理）
    mz_steady = 1.0 - np.exp(-tr / t1)
    M = np.array([0.0, 0.0, mz_steady])
    
    history = []
    events = {"rf": np.zeros(steps), "grad": np.zeros(steps)}
    
    # 指数減衰の定数計算
    E1 = np.exp(-dt / t1)
    E2 = np.exp(-dt / t2)
    # DWI用の拡散減衰率
    diffusion_decay = np.exp(-b_value * adc * 1e-3) if "DWI" in mode else 1.0

    def apply_pulse(m, angle):
        """RFパルスによる磁化ベクトルの回転 (X軸回転)"""
        rad = np.radians(angle)
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
        return rx @ m

    # --- 2. シーケンス開始時の初期パルス ---
    if "IR" in mode: 
        M = apply_pulse(M, 180) # 反転パルス
        events["rf"][0] = 180
    elif "SE" in mode or "DWI" in mode: 
        M = apply_pulse(M, 90)  # 励起パルス
        events["rf"][0] = 90
    elif "GRE" in mode: 
        M = apply_pulse(M, fa)  # フリップ角パルス
        events["rf"][0] = fa

    # --- 3. タイムステップ・ループ ---
    for t_step in range(steps):
        t_ms = t_step * dt
        
        # A. IR系 (FLAIR/STIR)
        if "IR" in mode:
            if abs(t_ms - ti) <= dt/2:
                M = apply_pulse(M, 90)
                events["rf"][t_step] = 90
            if abs(t_ms - (ti + te/2)) <= dt/2:
                M = apply_pulse(M, 180)
                events["rf"][t_step] = 180
            if abs(t_ms - (ti + te)) <= dt/2:
                events["grad"][t_step] = 1.0 # 信号読み出し

        # B. SE / DWI 系
        elif "SE" in mode or "DWI" in mode:
            # 再収束パルス (TEの半分で180度)
            if abs(t_ms - te/2) <= dt/2:
                M = apply_pulse(M, 180)
                events["rf"][t_step] = 180
            
            # エコー信号の読み出し (TEのタイミング)
            if abs(t_ms - te) <= dt/2:
                events["grad"][t_step] = 1.0
                if "DWI" in mode:
                    # 拡散による減衰を適用
                    M[:2] *= diffusion_decay
                    # 脂肪抑制(Fat-Sat)がオンかつ対象が脂肪の場合
                    # (FatのT1=250msで判定)
                    if fat_sat and t1 == 250:
                        M[:2] *= 0.05 # 信号を95%カット

        # C. GRE 系
        elif "GRE" in mode:
            if abs(t_ms - te) <= dt/2:
                # 理想的なリフェーズをシミュレート
                m_mag = np.sqrt(M[0]**2 + M[1]**2)
                M[0], M[1] = m_mag, 0.0
                events["grad"][t_step] = 1.0

        # --- 4. 磁化の物理緩和 (Bloch方程式の簡易解) ---
        # オフレゾナンスによるZ軸周りの回転
        rz = np.array([
            [np.cos(np.radians(df*dt)), -np.sin(np.radians(df*dt)), 0],
            [np.sin(np.radians(df*dt)), np.cos(np.radians(df*dt)), 0],
            [0, 0, 1]
        ])
        
        # 横緩和(E2)と縦緩和(E1)の適用
        # M = (緩和・回転) + (縦磁化の回復成分)
        M = (rz @ M) * np.array([E2, E2, E1]) + np.array([0, 0, 1-E1])
        
        history.append(M.copy())
        
    return np.array(history), dt, events


# --- 3. UI 設定 ---
st.set_page_config(page_title="MRI Physics Lab v4.2", layout="wide")

if 'ti_slider' not in st.session_state:
    st.session_state.ti_slider = 2300

def set_ti(target_t1):
    st.session_state.ti_slider = int(target_t1 * 0.693)

# --- サイドバーの設定 ---
with st.sidebar:
    st.divider()
    st.header("📺 Display Settings")
    # 輝度調整（ゲイン）
    img_gain = st.slider("Image Gain (Brightness)", 0.1, 5.0, 1.5, step=0.1)
    # ガンマ補正（中間階調の調整：1.0がリニア、小さいほど暗部が浮き上がる）
    gamma = st.slider("Gamma Correction", 0.1, 2.0, 1.0, step=0.1)
    mode = st.selectbox("Sequence Mode", 
                        ["Spin Echo (SE)", "Inversion Recovery (IR/FLAIR/STIR)", "Gradient Echo (GRE)", "Diffusion Weighted (DWI)"])
    
    tr = st.slider("TR (ms)", 500, 10000, 4000)
    te = st.slider("TE (ms)", 10, 250, 80)
    
    fa, ti, b_val, fat_sat = 90, 0, 0, False
    
    if "IR" in mode:
        st.write("✨ **Tissue Null Presets**")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            # FLAIRボタン（CSF抑制）
            if st.button("FLAIR (CSF Null)"):
                set_ti(TISSUE_PARAMS["CSF"]["t1"])
                # FLAIRらしいTR/TEに自動調整するのもアリ
                # st.session_state.tr_slider = 9000 
        with col_btn2:
            # STIRボタン（脂肪抑制）
            if st.button("STIR (Fat Null)"):
                set_ti(TISSUE_PARAMS["Fat"]["t1"])
        
        ti = st.slider("TI (ms)", 10, 5000, key="ti_slider")
    elif mode == "Gradient Echo (GRE)":
        fa = st.slider("Flip Angle", 5, 90, 30)
    elif mode == "Diffusion Weighted (DWI)":
        b_val = st.slider("b-value", 0, 2000, 1000)
        fat_sat = st.checkbox("Fat Suppression (Fat-Sat)", value=True)

    st.divider()
    t_a = st.selectbox("Tissue A (Blue Vector)", list(TISSUE_PARAMS.keys()), index=3)
    t_b = st.selectbox("Tissue B (Red Vector)", list(TISSUE_PARAMS.keys()), index=4)

# 計算実行
res = {}
for n in TISSUE_PARAMS.keys():
    p = TISSUE_PARAMS[n]
    h, dt, ev = simulate_mri(mode, p["t1"], p["t2"], p["adc"], fa, tr, te, ti, b_val, fat_sat=fat_sat)
    res[n] = {"h": h, "mxy": np.sqrt(h[:,0]**2 + h[:,1]**2), "mz": h[:,2], "dt": dt, "ev": ev}
if "IR" in mode:
    # IR系（FLAIR/STIR含む）は 180°(反転) -> TI待ち -> 90°(励起) -> TE待ち
    # なので、信号のピークは TI + TE の地点になります
    read_time = ti + te
else:
    # SE, GRE, DWI は 90°(励起) -> TE待ち なので、
    # 信号のピークは TE の地点になります
    read_time = te

# read_time が TR（全時間）を超えないように安全策をとり、インデックス番号に変換
read_time = min(read_time, tr - 10) # 10はdt
s_idx = int(read_time / dt)

def gen_labeled_phantom(res_data, s_idx, gain=1.0, gamma=1.0):
    img = Image.new("RGB", (400, 400), "black")
    draw = ImageDraw.Draw(img)

    def nrm(s):
        # ゲインを 1.0 前後で調整しやすくするため、基本の 255 を少し抑えるか
        # 信号強度 s (最大1.0) に基づいて計算
        # ガンマ補正をかける前にゲインを適用
        val = np.power(max(0, s * gain), 1/gamma) * 255
        return int(np.clip(val, 0, 255))
    
    sigs = {k: nrm(res_data[k]["mxy"][s_idx]) for k in TISSUE_PARAMS.keys()}
    
    # 描画（同心円状）
    draw.ellipse([20, 20, 380, 380], fill=(sigs["CSF"],)*3, outline="gray")
    draw.ellipse([80, 80, 320, 320], fill=(sigs["Gray Matter"],)*3, outline="gray")
    draw.ellipse([140, 140, 260, 260], fill=(sigs["White Matter"],)*3, outline="gray")
    draw.ellipse([100, 180, 160, 240], fill=(sigs["Infarct"],)*3, outline="red", width=3)
    draw.ellipse([240, 180, 300, 240], fill=(sigs["Fat"],)*3, outline="yellow", width=3)
    # ラベル描画
    labels = [(180, 35, "CSF", (255, 120, 120)), (140, 95, "Gray Matter", (255, 170, 170)), 
              (140, 155, "White Matter", (255, 220, 220)), (85, 250, "Infarct", (0, 255, 255)), 
              (255, 250, "Fat", (255, 255, 0))]
    for x, y, txt, col in labels:
        tw = len(txt) * 9
        draw.rectangle([x-5, y-2, x+tw, y+15], fill=(0,0,0))
        draw.text((x, y), txt, fill=col)
    return img

# --- 5. プロット表示 ---
col1, col2 = st.columns([3, 1.2])

with col1:
    fig = make_subplots(rows=3, cols=1, specs=[[{'type': 'scene'}], [{'secondary_y': True}], [{'type': 'xy'}]], 
                        vertical_spacing=0.08, row_heights=[0.45, 0.25, 0.3])
    time_ax = np.arange(len(res[t_a]["mxy"])) * dt

    # 下段：磁化曲線
    for n, color in [(t_a, "rgb(31, 119, 180)"), (t_b, "rgb(214, 39, 40)")]:
        fig.add_trace(go.Scatter(x=time_ax, y=res[n]["mxy"], name=f"{n} Mxy", line=dict(color=color, width=2.5)), 3, 1)
        fig.add_trace(go.Scatter(x=time_ax, y=res[n]["mz"], name=f"{n} Mz", line=dict(color=color, dash='dot', width=1.5), showlegend=False), 3, 1)
    fig.add_vline(x=read_time, line_width=2, line_dash="dash", line_color="red", row=3, col=1)

    # 中段：シーケンス図
    ev = res[t_a]["ev"]
    fig.add_trace(go.Bar(x=time_ax, y=ev["rf"], name="RF Pulse", marker_color="orange"), 2, 1, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_ax, y=ev["grad"], name="Grad/MPG", line=dict(color="green", width=2, shape='hv')), 2, 1, secondary_y=True)
    fig.update_yaxes(title_text="RF Angle", range=[-50, 250], secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Grad Power", range=[-3, 3], secondary_y=True, row=2, col=1)

# --- 5. プロット表示 (アニメーション最適化版) ---
    # (中略: figの作成や各グラフの追加はそのまま)

    # 3Dベクトル本体のトレース番号を保持
    trace_indices = []
    for n, color in [(t_a, "rgb(31, 119, 180)"), (t_b, "rgb(214, 39, 40)")]:
        h = res[n]['h']
        fig.add_trace(go.Scatter3d(x=h[:,0], y=h[:,1], z=h[:,2], mode='lines', 
                                   line=dict(color=color.replace("rgb", "rgba").replace(")", ", 0.15)"), width=2), 
                                   showlegend=False), 1, 1)
        fig.add_trace(go.Scatter3d(x=[0, h[0,0]], y=[0, h[0,1]], z=[0, h[0,2]], 
                                   mode='lines+markers', marker=dict(size=4), 
                                   line=dict(color=color, width=6), name=f"{n} Vector"), 1, 1)
        trace_indices.append(len(fig.data) - 1)

    # 【修正箇所1】全ステップを使用し、なめらかさを最大化
    animation_steps = np.arange(0, len(res[t_a]["h"]), 1) 

    fig.frames = [go.Frame(
        data=[go.Scatter3d(x=[0, res[n]["h"][k,0]], y=[0, res[n]["h"][k,1]], z=[0, res[n]["h"][k,2]]) for n in [t_a, t_b]],
        traces=trace_indices,
        name=f"frame{k}"
    ) for k in animation_steps]

    # 【修正箇所2】durationを30ms(スロー)に、redrawをTrueに設定
    # 【修正箇所】再生ボタンの引数に "frame: 0" を指定してリセット機能を追加
    fig.update_layout(
        height=950,
        scene=dict(camera=dict(eye=dict(x=1.2, y=1.2, z=0.5)), aspectmode='cube'),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ Play from Start", 
                     method="animate", 
                     args=[None, {
                         "frame": {"duration": 30, "redraw": True}, 
                         "fromcurrent": False,  # TrueからFalseに変更：常に最初から再生
                         "mode": "immediate",
                         "transition": {"duration": 0}
                     }]),
                dict(label="⏸ Pause", 
                     method="animate", 
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. 臨床黄金設定ガイド (グラフ直下に表示) ---
    st.divider()
    st.subheader("💡 臨床黄金設定レシピ")

    # 1段目：病変検出・特殊シーケンス
    guide_row1 = st.columns(3)
    with guide_row1[0]:
        st.markdown("""
        **1. FLAIR (水抑制)**
        * **用途**: 脳梗塞・脱髄疾患の検出
        * **設定**: Mode: `FLAIR`, TR: `9000`, TE: `120`, TI: `2772` (CSF Null)
        * **結果**: **CSF**が消失し、**Infarct**が白く浮かび上がります。
        """)

    with guide_row1[1]:
        st.markdown("""
        **2. STIR (脂肪抑制)**
        * **用途**: 骨転移・炎症・視神経の評価
        * **設定**: Mode: `IR`, TR: `4000`, TE: `60`, TI: `173` (Fat Null)
        * **結果**: **Fat**が消失し、脂肪に隠れた病変が見えやすくなります。
        """)

    with guide_row1[2]:
        st.markdown("""
        **3. DWI (急性期梗塞)**
        * **用途**: 超急性期脳梗塞の診断
        * **設定**: Mode: `DWI`, TR: `4000`, TE: `100`, b-value: `1000`
        * **結果**: 全体が暗くなる中、**Infarct**だけが明るく残ります。
        """)

    # 2段目：基本コントラスト・出血検出
    guide_row2 = st.columns(3)
    with guide_row2[0]:
        st.markdown("""
        **4. T1強調画像 (解剖)**
        * **用途**: 解剖学的構造の把握
        * **設定**: Mode: `SE`, TR: `500`, TE: `15`
        * **結果**: **Fat**が非常に明るく、**CSF**は真っ黒。解剖図に近い見え方です。
        """)

    with guide_row2[1]:
        st.markdown("""
        **5. T2*強調画像 (出血敏感)**
        * **用途**: 出血(ヘモジデリン)・石灰化の検出
        * **設定**: Mode: `GRE`, TR: `600`, TE: `25`, FA: `20°`
        * **結果**: 微小出血などが「黒い点（信号欠損）」として強調されます。
        """)

    with guide_row2[2]:
        st.markdown("""
        **📊 コントラスト評価のコツ**
        * **Signal Delta**: AとBの数値の差が大きいほど、その2つの組織の境界がはっきり見えます。
        * **Null Point**: TIを調整して特定の組織の信号が0になる瞬間を探してみましょう。
        """)
with col2:
    st.header("🖼️ Phantom Preview")
# サイドバーのスライダー(img_gain, gamma)を使って画像を作成
    phantom_img = gen_labeled_phantom(res, s_idx, gain=img_gain, gamma=gamma)
    
    # 作成した画像をStreamlitの画面に配置
    st.image(phantom_img, use_container_width=True)
    # ★★★ ここまで ★★★
    
    st.info(f"🔵 **Tissue A**: {t_a}\n\n🔴 **Tissue B**: {t_b}")
    st.metric("Signal Delta (abs)", f"{abs(res[t_a]['mxy'][s_idx] - res[t_b]['mxy'][s_idx]):.4f}")
    st.write("---")
    st.caption("v4.2 Final: 全機能復旧・視覚化最適化済み")