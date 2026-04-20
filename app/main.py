"""
main.py — Off-Road Terrain Detection + Depth & Height Estimation
================================================================
YOLOv8 Instance Segmentation  +  MiDaS Depth  +  Height Mapping

Run:
    streamlit run main.py

Install (if not done):
    pip install ultralytics streamlit opencv-python numpy pillow timm torch torchvision
"""

import io
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image as PILImage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "TerrainAI — Depth & Segmentation",
    page_icon   = "🛰️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #06090f; color: #c8d8e8; }
.main { background: #06090f; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.terrain-header {
    background: linear-gradient(135deg, #060d1a 0%, #0a1628 50%, #060d1a 100%);
    border: 1px solid rgba(0,220,180,0.4);
    border-radius: 16px;
    padding: 28px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.terrain-header::before {
    content: '';
    position: absolute; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
        rgba(0,220,180,0.015) 2px, rgba(0,220,180,0.015) 4px);
    pointer-events: none;
}
.terrain-header h1 {
    font-family: 'Orbitron', sans-serif;
    color: #00dcb4; font-size: 1.85rem; font-weight: 800; margin: 0;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,220,180,0.6), 0 0 40px rgba(0,220,180,0.3);
}
.terrain-header .sub { color: #5a9aaa; font-size: 0.88rem; margin-top: 8px; letter-spacing: 1.5px; text-transform: uppercase; }
.badge {
    display: inline-block;
    background: rgba(0,220,180,0.12); border: 1px solid rgba(0,220,180,0.3);
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; color: #00dcb4;
    margin-right: 8px; margin-top: 10px;
    font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
}

.metric-grid { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
.mtile {
    flex: 1; min-width: 120px;
    background: linear-gradient(145deg, #0a1220, #0d1a2e);
    border: 1px solid #1a3050; border-radius: 12px;
    padding: 18px 14px; text-align: center; position: relative; overflow: hidden;
}
.mtile::after {
    content: ''; position: absolute; bottom:0; left:0; right:0; height:2px;
    background: var(--accent, #00dcb4); opacity: 0.6;
}
.mtile .v {
    font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: 700;
    color: var(--accent, #00dcb4); text-shadow: 0 0 10px var(--accent, #00dcb4); line-height: 1;
}
.mtile .l { font-size: 0.66rem; color: #4a7a8a; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 6px; }
.mtile.risk { --accent: #ff4444; }
.mtile.safe { --accent: #00ff88; }
.mtile.mod  { --accent: #ffaa00; }
.mtile.uns  { --accent: #ff6622; }
.mtile.objs { --accent: #44aaff; }
.mtile.ms   { --accent: #aa88ff; }

.section-label {
    font-family: 'Orbitron', sans-serif; font-size: 0.78rem; color: #00dcb4;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; margin-top: 4px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex:1; height:1px; background: linear-gradient(90deg,rgba(0,220,180,0.3),transparent); }

.det-card {
    background: linear-gradient(135deg, #0a1220, #0d1a2e);
    border: 1px solid #1a3050; border-left: 3px solid var(--cls-color, #00dcb4);
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
}
.det-card .cls-name { font-family: 'Orbitron', sans-serif; font-size: 0.88rem; color: var(--cls-color, #00dcb4); font-weight: 600; letter-spacing: 1px; }
.det-card .det-info { font-size: 0.82rem; color: #7090a0; margin-top: 6px; display: flex; gap: 20px; flex-wrap: wrap; }
.det-card .det-info span { color: #a0c0d0; }

.styled-table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:10px; }
.styled-table thead tr { background:#0d1a2e; color:#00dcb4; font-family:'Orbitron',sans-serif; font-size:0.72rem; letter-spacing:1px; text-transform:uppercase; }
.styled-table th, .styled-table td { padding:10px 14px; text-align:left; border-bottom:1px solid #1a3050; }
.styled-table tbody tr:hover { background:rgba(0,220,180,0.05); }
.safe-cls  { color:#00ff88; font-weight:600; }
.mod-cls   { color:#ffaa00; font-weight:600; }
.unsafe-cls{ color:#ff4444; font-weight:600; }
.flat-h    { color:#00ff88; }
.raised-h  { color:#ff4444; }
.dip-h     { color:#4488ff; }

section[data-testid="stSidebar"] { background: #080e1a !important; border-right: 1px solid #1a3050; }

.stFileUploader > div { background:#0a1220 !important; border:2px dashed rgba(0,220,180,0.4) !important; border-radius:12px !important; }

.stDownloadButton > button {
    background: linear-gradient(90deg, #003a30, #005a48) !important;
    color: #00dcb4 !important; border: 1px solid rgba(0,220,180,0.4) !important;
    border-radius: 8px !important; font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important; letter-spacing: 1px !important;
}

.legend-box { background:#0a1220; border:1px solid #1a3050; border-radius:10px; padding:14px 16px; font-size:0.82rem; }
.legend-row { display:flex; align-items:center; gap:10px; margin:7px 0; color:#8aabb8; }
.legend-swatch { width:16px; height:16px; border-radius:3px; flex-shrink:0; }
.legend-section { font-family:'Orbitron',sans-serif; font-size:0.68rem; color:#00dcb4; letter-spacing:1px; margin:12px 0 4px; text-transform:uppercase; }

.profile-wrap { background:#0a1220; border:1px solid #1a3050; border-radius:12px; padding:12px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
WEIGHTS_CANDIDATES = [
    "../runs/segment/train/weights/best.pt",
    "runs/segment/train/weights/best.pt",
    "best.pt",
]
SAFE     = ["Road","road","Dirt","dirt","Path","path"]
MODERATE = ["Tree","tree","Grass","grass","Bush","bush","Plant","plant","Vegetation"]
UNSAFE   = ["Rock","rock","Rubble","rubble","Water","water","Mud","mud"]
CLS_ICON = {
    "Road":"🛣️","road":"🛣️","Dirt":"🟤","dirt":"🟤","Path":"🛤️",
    "Tree":"🌲","tree":"🌲","Bush":"🌿","bush":"🌿","Grass":"🌾","grass":"🌾",
    "Plant":"🪴","plant":"🪴","Vegetation":"🌳",
    "Rock":"🪨","rock":"🪨","Rubble":"⛏️","rubble":"⛏️","Water":"💧","Mud":"🟫",
}
CLR_SAFE     = (0, 220, 80)
CLR_MODERATE = (0, 160, 255)
CLR_UNSAFE   = (0, 40, 220)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ═════════════════════════════════════════════════════════════════════════════
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

@st.cache_resource(show_spinner=False)
def load_yolo(wp):
    return YOLO(wp)

@st.cache_resource(show_spinner=False)
def load_midas():
    model   = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    model.eval()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tforms  = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, tforms.dpt_transform, device

def find_weights():
    for p in WEIGHTS_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  DEPTH / HEIGHT
# ═════════════════════════════════════════════════════════════════════════════
def estimate_depth(bgr, midas, transform, device):
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=bgr.shape[:2],
            mode="bicubic", align_corners=False).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)

def estimate_height(depth_norm):
    blurred = cv2.GaussianBlur(depth_norm, (51,51), 0)
    hm      = depth_norm - blurred
    return (hm / (np.abs(hm).max() + 1e-8)).astype(np.float32)

def depth_colormap(d):
    return cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_TURBO)

def height_colormap(hm):
    h,w  = hm.shape
    out  = np.zeros((h,w,3), np.uint8)
    flat = np.abs(hm) < 0.18
    up   = hm >  0.18
    dn   = hm < -0.18
    out[flat] = (80, 220, 80)
    iv  = np.clip((hm - 0.18)/0.82, 0, 1)
    out[up,0] = (iv[up]*255).astype(np.uint8)
    out[up,1] = ((1-iv[up])*60).astype(np.uint8)
    out[up,2] = ((1-iv[up])*30).astype(np.uint8)
    iv2 = np.clip((-hm[dn]-0.18)/0.82, 0, 1)
    out[dn,0] = (iv2*30).astype(np.uint8)
    out[dn,1] = (iv2*100).astype(np.uint8)
    out[dn,2] = (iv2*255).astype(np.uint8)
    return out

def region_stats(mask, depth_norm, height_map):
    if mask.sum() == 0:
        return {"mean_depth":0.0,"height_class":"UNKNOWN","est_cm":0.0}
    hv = float(height_map[mask==1].mean())
    return {
        "mean_depth":   round(float(depth_norm[mask==1].mean()), 3),
        "height_class": "RAISED" if hv>0.22 else "DEPRESSION" if hv<-0.22 else "FLAT",
        "est_cm":       round(hv*85, 1),
    }

def road_fallback(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,th = cv2.threshold(blur,120,255,cv2.THRESH_BINARY_INV)
    return (cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))//255)


# ═════════════════════════════════════════════════════════════════════════════
#  PROFILE CHART  (pure OpenCV — no matplotlib, no pandas)
# ═════════════════════════════════════════════════════════════════════════════
def draw_profile(depth_col, height_col, W=1000, H=210):
    canvas = np.full((H,W,3),(10,16,26),np.uint8)
    for gx in range(0,W,W//10): cv2.line(canvas,(gx,0),(gx,H),(20,35,50),1)
    for gy in range(0,H,H//4):  cv2.line(canvas,(0,gy),(W,gy),(20,35,50),1)
    n = len(depth_col)
    def pts(vals, margin=18):
        lo,hi = vals.min(), vals.max()+1e-8
        normed = (vals-lo)/(hi-lo)
        return [(int(i/n*W), int(H-margin-normed[i]*(H-2*margin))) for i in range(n)]
    pd_ = pts(depth_col)
    ph_ = pts((height_col+1)/2)
    for i in range(1,n):
        cv2.line(canvas,pd_[i-1],pd_[i],(0,210,255),2)
        cv2.line(canvas,ph_[i-1],ph_[i],(0,140,255),1)
    mid=W//2
    cv2.line(canvas,(mid,0),(mid,H),(50,80,100),1)
    cv2.putText(canvas,"DEPTH & HEIGHT PROFILE  (centre column)",
                (14,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(60,120,140),1)
    cv2.circle(canvas,(14,40),5,(0,210,255),-1)
    cv2.putText(canvas,"Depth (red=close, blue=far)",
                (24,45),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,210,255),1)
    cv2.circle(canvas,(14,60),5,(0,140,255),-1)
    cv2.putText(canvas,"Relative Height (up=obstacle, down=dip)",
                (24,65),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,140,255),1)
    cv2.putText(canvas,"FAR  (top of image)",
                (14,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.33,(40,70,90),1)
    cv2.putText(canvas,"NEAR (bottom — vehicle position)",
                (mid+10,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.33,(40,70,90),1)
    return canvas


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(image_bgr, yolo_model, midas, transform, device):
    h,w = image_bgr.shape[:2]
    t0  = time.time()

    # YOLO
    results = yolo_model(image_bgr, conf=0.25, verbose=False)
    result  = results[0]
    seg     = image_bgr.copy()
    safe_m  = np.zeros((h,w),np.uint8)
    mod_m   = np.zeros((h,w),np.uint8)
    uns_m   = np.zeros((h,w),np.uint8)
    dets    = []

    if result.masks is not None:
        for i,mt in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[i])
            label  = yolo_model.names[cls_id]
            conf   = float(result.boxes.conf[i])
            x1,y1,x2,y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
            mn = (cv2.resize(mt.cpu().numpy(),(w,h))>0.5).astype(np.uint8)

            cat = "moderate"
            if   label in SAFE:     safe_m = np.maximum(safe_m,mn); cat="safe"
            elif label in MODERATE: mod_m  = np.maximum(mod_m,mn)
            elif label in UNSAFE:   uns_m  = np.maximum(uns_m,mn);  cat="unsafe"

            clr = CLR_SAFE if cat=="safe" else CLR_MODERATE if cat=="moderate" else CLR_UNSAFE
            cv2.rectangle(seg,(x1,y1),(x2,y2),clr,2)
            txt = f"{label} {conf:.2f}"
            (tw,th2),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(seg,(x1,y1-28),(x1+tw+6,y1),(15,22,32),-1)
            cv2.rectangle(seg,(x1,y1-28),(x1+tw+6,y1),clr,1)
            cv2.putText(seg,txt,(x1+3,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(220,220,220),2)
            dets.append({"label":label,"conf":round(conf,2),"bbox":[x1,y1,x2,y2],"mask":mn,"cat":cat})

    if safe_m.sum()==0: safe_m = road_fallback(image_bgr)

    def ov(mask,clr):
        o=np.zeros_like(image_bgr); o[mask==1]=clr; return o

    seg = cv2.addWeighted(seg,0.62,ov(safe_m,CLR_SAFE),    0.58,0)
    seg = cv2.addWeighted(seg,0.72,ov(mod_m, CLR_MODERATE),0.45,0)
    seg = cv2.addWeighted(seg,0.68,ov(uns_m, CLR_UNSAFE),  0.55,0)

    # Drivable contour
    contours,_ = cv2.findContours(safe_m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lg = max(contours,key=cv2.contourArea)
        cv2.drawContours(seg,[lg],-1,(0,255,0),5)
        M = cv2.moments(lg)
        if M["m00"]!=0:
            cx=int(M["m10"]/M["m00"]); cy=int(M["m01"]/M["m00"])
            cv2.arrowedLine(seg,(w//2,h-5),(cx,cy),(0,255,0),5,tipLength=0.03)
            cv2.putText(seg,"DRIVABLE PATH",(cx-88,cy-18),cv2.FONT_HERSHEY_SIMPLEX,0.88,(0,0,0),3)
            cv2.putText(seg,"DRIVABLE PATH",(cx-88,cy-18),cv2.FONT_HERSHEY_SIMPLEX,0.88,(0,255,100),2)

    risk = (uns_m.sum()+0.5*mod_m.sum())/(h*w)
    risk_pct = round(risk*100,2)
    cv2.rectangle(seg,(8,8),(278,58),(10,18,28),-1)
    cv2.rectangle(seg,(8,8),(278,58),(0,220,160),1)
    cv2.putText(seg,f"Terrain Risk: {risk_pct}%",(16,42),cv2.FONT_HERSHEY_SIMPLEX,0.78,(0,220,160),2)

    # MiDaS
    depth_norm = estimate_depth(image_bgr, midas, transform, device)
    height_map = estimate_height(depth_norm)
    depth_vis  = depth_colormap(depth_norm)
    height_vis = height_colormap(height_map)

    depth_ann  = depth_vis.copy()
    for det in dets:
        s = region_stats(det["mask"],depth_norm,height_map)
        det["stats"] = s
        x1,y1,x2,y2 = det["bbox"]
        cx=(x1+x2)//2; cy=(y1+y2)//2
        cv2.putText(depth_ann,det["label"],(cx-30,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
        cv2.putText(depth_ann,f"d={s['mean_depth']:.2f}",(cx-30,cy+12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,255,200),1)
        cv2.putText(seg,f"D:{s['mean_depth']:.2f} H:{s['est_cm']:+.0f}cm",
                    (x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)

    return {
        "seg":        seg,
        "depth_ann":  depth_ann,
        "height_vis": height_vis,
        "depth_norm": depth_norm,
        "height_map": height_map,
        "centre_d":   depth_norm[:,w//2],
        "centre_h":   height_map[:,w//2],
        "risk_pct":   risk_pct,
        "safe_pct":   round(100*safe_m.sum()/(h*w),1),
        "mod_pct":    round(100*mod_m.sum()/(h*w),1),
        "uns_pct":    round(100*uns_m.sum()/(h*w),1),
        "dets":       dets,
        "ms":         int((time.time()-t0)*1000),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═════════════════════════════════════════════════════════════════════════════
def to_bytes_bgr(bgr):
    buf=io.BytesIO(); PILImage.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(buf,"PNG"); buf.seek(0); return buf

def to_bytes_rgb(rgb):
    buf=io.BytesIO(); PILImage.fromarray(rgb).save(buf,"PNG"); buf.seek(0); return buf


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="terrain-header">
  <h1>◈ TERRAIN AI — DETECTION + DEPTH + HEIGHT</h1>
  <div class="sub">Off-Road Autonomous Vehicle Perception System</div>
  <span class="badge">YOLOv8-seg</span>
  <span class="badge">MiDaS Depth</span>
  <span class="badge">Height Mapping</span>
  <span class="badge">Traversability</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Display Options</div>', unsafe_allow_html=True)
    show_depth   = st.toggle("Depth Map",          value=True)
    show_height  = st.toggle("Height Map",         value=True)
    show_profile = st.toggle("Depth Profile Chart",value=True)
    show_table   = st.toggle("Detection Table",    value=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎨 Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-box">
      <div class="legend-section">Segmentation</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#00ff88"></div>Safe / Road / Dirt</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#ffaa00"></div>Moderate — Tree / Bush</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#ff4444"></div>Unsafe — Rock / Rubble</div>
      <div class="legend-section">Depth Map (MiDaS)</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#ff2200"></div>Close to camera</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#00ff88"></div>Mid distance</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#000090"></div>Far from camera</div>
      <div class="legend-section">Height Map</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#50dc50"></div>Flat terrain</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#ff2200"></div>Raised obstacle</div>
      <div class="legend-row"><div class="legend-swatch" style="background:#0044ff"></div>Depression / dip</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📡 Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-box" style="font-family:monospace;font-size:0.78rem;color:#5a9aaa">
      Segmentation : YOLOv8n-seg<br>
      Depth        : MiDaS DPT-Hybrid<br>
      Height       : Local depth contrast<br>
      Framework    : PyTorch + OpenCV<br>
    </div>
    """, unsafe_allow_html=True)

# Load models
if not HAS_YOLO:
    st.error("Install ultralytics:  pip install ultralytics"); st.stop()

weights = find_weights()
if weights is None:
    st.error("No trained weights found. Run python train.py first.")
    st.info("Expected: runs/segment/train/weights/best.pt"); st.stop()

status = st.empty()
with st.spinner("Loading YOLOv8..."):
    yolo_model = load_yolo(weights)
with st.spinner("Loading MiDaS (first run ~100 MB)..."):
    try:
        midas, midas_t, device = load_midas()
        status.success(f"✅  YOLOv8: `{weights}`  |  MiDaS: DPT-Hybrid  |  Device: `{device}`")
    except Exception as e:
        st.error(f"MiDaS failed: {e}  →  pip install timm"); st.stop()

st.markdown("---")

# Upload
st.markdown('<div class="section-label">📤 Upload Terrain Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop JPG / PNG here", type=["jpg","jpeg","png","bmp","webp"],
                             label_visibility="collapsed")

if uploaded:
    raw   = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot read image."); st.stop()

    prog = st.progress(0, "Running pipeline...")
    with st.spinner("🔍 Segmentation + Depth + Height..."):
        prog.progress(20, "YOLO segmenting...")
        out = run_pipeline(img, yolo_model, midas, midas_t, device)
        prog.progress(100, "Done!")
    prog.empty()

    # Metrics
    st.markdown(f"""
    <div class="metric-grid">
      <div class="mtile objs"><div class="v">{len(out['dets'])}</div><div class="l">Objects</div></div>
      <div class="mtile risk"><div class="v">{out['risk_pct']}%</div><div class="l">Terrain Risk</div></div>
      <div class="mtile safe"><div class="v">{out['safe_pct']}%</div><div class="l">Safe Zone</div></div>
      <div class="mtile mod"><div class="v">{out['mod_pct']}%</div><div class="l">Moderate Zone</div></div>
      <div class="mtile uns"><div class="v">{out['uns_pct']}%</div><div class="l">Unsafe Zone</div></div>
      <div class="mtile ms"><div class="v">{out['ms']} ms</div><div class="l">Inference</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 — Input + Seg
    st.markdown('<div class="section-label">📷 Input Image &nbsp;·&nbsp; 🗺️ Segmentation + Traversability</div>',
                unsafe_allow_html=True)
    c1,c2 = st.columns(2, gap="medium")
    with c1:
        st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"{uploaded.name}  ·  {img.shape[1]}×{img.shape[0]} px")
    with c2:
        st.image(cv2.cvtColor(out["seg"],cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"Risk {out['risk_pct']}%  ·  Safe {out['safe_pct']}%  ·  {len(out['dets'])} objects  ·  Depth+Height labels shown")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Depth + Height
    if show_depth or show_height:
        st.markdown('<div class="section-label">📏 Depth Map &nbsp;·&nbsp; ⛰️ Height Map</div>',
                    unsafe_allow_html=True)
        c3,c4 = st.columns(2, gap="medium")
        if show_depth:
            with c3:
                st.image(cv2.cvtColor(out["depth_ann"],cv2.COLOR_BGR2RGB), use_container_width=True,
                         caption="MiDaS Depth — RED=closest · BLUE=farthest · Each region labelled with depth score")
        if show_height:
            with c4:
                st.image(out["height_vis"], use_container_width=True,
                         caption="Height Map — RED=raised obstacle · GREEN=flat · BLUE=depression/dip")
        st.markdown("<br>", unsafe_allow_html=True)

    # Row 3 — Profile
    if show_profile:
        st.markdown('<div class="section-label">📉 Depth & Height Profile — Centre Column</div>',
                    unsafe_allow_html=True)
        chart = draw_profile(out["centre_d"], out["centre_h"], W=1100, H=210)
        st.markdown('<div class="profile-wrap">', unsafe_allow_html=True)
        st.image(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Cyan = depth curve (spike = sky/tree boundary). Orange = relative height. Left = far away (sky), Right = close (near vehicle).")
        st.markdown("<br>", unsafe_allow_html=True)

    # Detection details
    if show_table and out["dets"]:
        st.markdown("---")
        st.markdown('<div class="section-label">📊 Detection Details — Depth & Height per Object</div>',
                    unsafe_allow_html=True)

        for i, det in enumerate(out["dets"]):
            s  = det.get("stats",{})
            hc = s.get("height_class","—")
            color = "#00ff88" if det["cat"]=="safe" else "#ffaa00" if det["cat"]=="moderate" else "#ff4444"
            hcolor= "#00ff88" if hc=="FLAT" else "#ff4444" if hc=="RAISED" else "#4488ff"
            icon  = CLS_ICON.get(det["label"],"🔲")
            st.markdown(f"""
            <div class="det-card" style="--cls-color:{color}">
              <div class="cls-name">{icon}&nbsp; {det['label'].upper()} — Detection #{i+1}</div>
              <div class="det-info">
                <div>Confidence: <span>{det['conf']:.0%}</span></div>
                <div>Depth Score: <span>{s.get('mean_depth',0):.3f} &nbsp;(1=closest)</span></div>
                <div>Est. Height: <span>{s.get('est_cm',0):+.0f} cm</span></div>
                <div>Height Class: <span style="color:{hcolor};font-weight:600">{hc}</span></div>
                <div>BBox: <span style="font-size:0.78rem">[{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}]</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Pure HTML table — NO PANDAS NEEDED
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Summary Table</div>', unsafe_allow_html=True)
        rows = ""
        for i,det in enumerate(out["dets"]):
            s   = det.get("stats",{})
            hc  = s.get("height_class","—")
            cc  = "safe-cls" if det["cat"]=="safe" else "mod-cls" if det["cat"]=="moderate" else "unsafe-cls"
            hcc = "flat-h" if hc=="FLAT" else "raised-h" if hc=="RAISED" else "dip-h"
            rows += f"""<tr>
              <td>{i+1}</td>
              <td class="{cc}">{det['label']}</td>
              <td>{det['conf']:.0%}</td>
              <td>{s.get('mean_depth',0):.3f}</td>
              <td class="{hcc}">{s.get('est_cm',0):+.0f} cm</td>
              <td class="{hcc}">{hc}</td>
              <td style="font-size:0.75rem;color:#3a6a7a">[{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}]</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>#</th><th>Class</th><th>Confidence</th>
            <th>Depth Score</th><th>Est. Height</th><th>Height Class</th><th>Bounding Box</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    # Downloads
    st.markdown("---")
    st.markdown('<div class="section-label">💾 Download Results</div>', unsafe_allow_html=True)
    stem = os.path.splitext(uploaded.name)[0]
    d1,d2,d3,d4 = st.columns(4)
    with d1: st.download_button("⬇ Segmentation", to_bytes_bgr(out["seg"]),   f"{stem}_seg.png",    "image/png")
    with d2:
        if show_depth:  st.download_button("⬇ Depth Map",    to_bytes_bgr(out["depth_ann"]), f"{stem}_depth.png",  "image/png")
    with d3:
        if show_height: st.download_button("⬇ Height Map",   to_bytes_rgb(out["height_vis"]),f"{stem}_height.png", "image/png")
    with d4:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        report = {"file":uploaded.name,"risk_pct":float(out["risk_pct"]),"safe_pct":float(out["safe_pct"]),
                  "detections":[{"label":d["label"],"confidence":float(d["conf"]),"category":d["cat"],
                    "depth_score":float(d.get("stats",{}).get("mean_depth",0)),
                    "est_height_cm":float(d.get("stats",{}).get("est_cm",0)),
                    "height_class":d.get("stats",{}).get("height_class",""),
                    "bbox":[int(v) for v in d["bbox"]]} for d in out["dets"]]}
        st.download_button("⬇ JSON Report", json.dumps(report,indent=2,cls=NpEncoder), f"{stem}_report.json","application/json")

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px">
      <div style="font-size:5rem;margin-bottom:20px;filter:drop-shadow(0 0 20px rgba(0,220,180,0.4))">🛰️</div>
      <h3 style="font-family:'Orbitron',sans-serif;color:#00dcb4;letter-spacing:3px;text-shadow:0 0 16px rgba(0,220,180,0.5)">
        AWAITING TERRAIN IMAGE
      </h3>
      <p style="color:#3a6a7a;max-width:500px;margin:16px auto;font-size:0.95rem;line-height:1.8">
        Upload any off-road terrain photo to get<br>
        <span style="color:#00ff88">■ Segmentation</span> &nbsp;
        <span style="color:#ffaa00">■ Depth Map</span> &nbsp;
        <span style="color:#ff6644">■ Height Map</span> &nbsp;
        <span style="color:#aa88ff">■ Profile Chart</span>
      </p>
      <p style="color:#2a4a5a;font-size:0.8rem;font-family:monospace">JPG · PNG · BMP · WEBP</p>
    </div>
    """, unsafe_allow_html=True)
