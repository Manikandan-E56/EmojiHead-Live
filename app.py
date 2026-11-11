import math
import os
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ==========================================================
# SETTINGS (tweak these)
# ==========================================================
DOCK_POSITIONS = ["left", "right", "top", "bottom"]
dock_pos = "left"     # press D to move the tray
SIZE = 1.40           # ball size vs face height (bigger = covers more hair/neck)
LIFT = 0.10           # move ball up to cover hair
SMOOTH = 0.5          # 0..1 (higher = follows faster, lower = smoother)
YAW_RANGE = 75        # degrees the ball can turn left/right
PITCH_RANGE = 50      # degrees the ball can tilt up/down
YAW_GAIN = 3.0        # how strongly head turn rotates the ball
PITCH_GAIN = 3.0      # how strongly head nod rotates the ball
PITCH_NEUTRAL = 0.62  # nose position when looking straight (tune if face looks up/down)
FLIP_YAW = False      # set True if ball turns the wrong way left/right
FLIP_PITCH = False    # set True if ball nods the wrong way up/down
RENDER_RES = 480      # sphere render quality (lower = faster)
HOLD_FRAMES = 6       # keep the ball for a few frames if face is lost

HAIR = 0.50        # how far above the forehead point to cover (hair). Bigger = covers more top
CHIN = 0.10        # extra cover below the chin
EAR = 0.16         # extra cover beyond the cheeks (ears), as a fraction of face width
FIT = 1.06         # final safety margin (1.0 = tight fit, 1.15 = loose)

# ==========================================================
# 1. Model + detector
# ==========================================================
MODEL_PATH = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading Face Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
    )
)

# ==========================================================
# 2. Emoji face textures (flat faces, wrapped on the sphere later)
# ==========================================================
SS = 2
TEX_RES = 512
YELLOW = (255, 205, 35)
RED = (245, 75, 45)


def W(S, px):
    return max(1, int(px * S / 450))


def add_layer(im, draw_fn):
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    return Image.alpha_composite(im, layer)


def new_tex(S, fill=YELLOW):
    return Image.new("RGBA", (S, S), fill + (255,))


def tex_cry(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    brown, dark, blue = (70, 40, 20), (50, 25, 10), (30, 160, 255)
    d.line([(S * .22, S * .28), (S * .42, S * .36)], fill=brown, width=W(S, 9))
    d.line([(S * .78, S * .28), (S * .58, S * .36)], fill=brown, width=W(S, 9))
    d.ellipse([S * .28, S * .38, S * .40, S * .50], fill=dark)
    d.ellipse([S * .60, S * .38, S * .72, S * .50], fill=dark)
    d.arc([S * .32, S * .60, S * .68, S * .78], 195, 345, fill=(60, 30, 15), width=W(S, 10))
    d.polygon([(S * .34, S * .46), (S * .27, S * .68), (S * .41, S * .68)], fill=blue)
    d.ellipse([S * .26, S * .58, S * .42, S * .74], fill=blue)
    return add_layer(im, lambda dl: (
        dl.ellipse([S * .30, S * .60, S * .34, S * .66], fill=(255, 255, 255, 170)),
        dl.ellipse([S * .30, S * .40, S * .33, S * .43], fill=(255, 255, 255, 190)),
        dl.ellipse([S * .62, S * .40, S * .65, S * .43], fill=(255, 255, 255, 190))))


def tex_joy(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    brown = (60, 30, 15)
    d.arc([S * .22, S * .30, S * .42, S * .44], 190, 350, fill=brown, width=W(S, 10))
    d.arc([S * .58, S * .30, S * .78, S * .44], 190, 350, fill=brown, width=W(S, 10))
    d.chord([S * .26, S * .36, S * .74, S * .80], 0, 180, fill=(180, 20, 30),
            outline=(50, 20, 10), width=W(S, 6))
    d.rectangle([S * .30, S * .58, S * .70, S * .64], fill=(255, 255, 255))
    d.ellipse([S * .38, S * .70, S * .62, S * .79], fill=(255, 100, 120))
    return im


def tex_heart(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    red = (225, 20, 45)
    glints = []
    for cx in (int(S * .31), int(S * .69)):
        cy, r = int(S * .40), int(S * .12)
        d.polygon([(cx - int(r * 1.4), cy), (cx + int(r * 1.4), cy), (cx, cy + int(r * 1.7))], fill=red)
        d.ellipse([cx - int(r * 1.4), cy - r, cx, cy + int(r * .4)], fill=red)
        d.ellipse([cx, cy - r, cx + int(r * 1.4), cy + int(r * .4)], fill=red)
        glints.append([cx - r * 1.0, cy - r * .8, cx - r * .25, cy - r * .25])
    d.chord([S * .26, S * .50, S * .74, S * .80], 0, 180, fill=(120, 10, 20),
            outline=(60, 15, 10), width=W(S, 5))
    d.rectangle([S * .30, S * .59, S * .70, S * .64], fill=(255, 255, 255))
    d.ellipse([S * .38, S * .70, S * .62, S * .79], fill=(255, 100, 110))
    return add_layer(im, lambda dl: [dl.ellipse(g, fill=(255, 255, 255, 170)) for g in glints])


def tex_cool(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (20, 20, 25)
    d.rectangle([S * .18, S * .36, S * .82, S * .44], fill=dark)
    d.polygon([(S * .18, S * .42), (S * .46, S * .42), (S * .42, S * .56), (S * .22, S * .56)], fill=dark)
    d.polygon([(S * .54, S * .42), (S * .82, S * .42), (S * .78, S * .56), (S * .58, S * .56)], fill=dark)
    d.arc([S * .36, S * .60, S * .68, S * .74], 15, 165, fill=(50, 25, 15), width=W(S, 8))
    return add_layer(im, lambda dl: (
        dl.line([(S * .24, S * .44), (S * .34, S * .54)], fill=(255, 255, 255, 120), width=W(S, 4)),
        dl.line([(S * .60, S * .44), (S * .70, S * .54)], fill=(255, 255, 255, 120), width=W(S, 4))))


def tex_angry(S):
    im = new_tex(S, RED)
    d = ImageDraw.Draw(im)
    dark = (40, 10, 10)
    d.line([(S * .20, S * .32), (S * .45, S * .42)], fill=dark, width=W(S, 11))
    d.line([(S * .80, S * .32), (S * .55, S * .42)], fill=dark, width=W(S, 11))
    d.ellipse([S * .28, S * .43, S * .40, S * .53], fill=(30, 10, 10))
    d.ellipse([S * .60, S * .43, S * .72, S * .53], fill=(30, 10, 10))
    d.arc([S * .30, S * .65, S * .70, S * .80], 195, 345, fill=dark, width=W(S, 10))
    return im


def tex_surprise(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    brown = (60, 30, 15)
    d.arc([S * .22, S * .22, S * .42, S * .34], 190, 350, fill=brown, width=W(S, 8))
    d.arc([S * .58, S * .22, S * .78, S * .34], 190, 350, fill=brown, width=W(S, 8))
    for x0 in (.24, .56):
        d.ellipse([S * x0, S * .36, S * (x0 + .20), S * .54], fill=(255, 255, 255),
                  outline=(40, 20, 10), width=W(S, 5))
    d.ellipse([S * .31, S * .42, S * .37, S * .48], fill=(30, 20, 10))
    d.ellipse([S * .63, S * .42, S * .69, S * .48], fill=(30, 20, 10))
    d.ellipse([S * .40, S * .62, S * .60, S * .82], fill=(40, 20, 15),
              outline=(30, 15, 10), width=W(S, 6))
    return im

def star_pts(cx, cy, r):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def tex_smile(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (50, 25, 10)
    d.ellipse([S * .30, S * .34, S * .40, S * .52], fill=dark)
    d.ellipse([S * .60, S * .34, S * .70, S * .52], fill=dark)
    d.arc([S * .26, S * .42, S * .74, S * .78], 20, 160, fill=(60, 30, 15), width=W(S, 10))
    return im


def tex_wink(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (50, 25, 10)
    d.ellipse([S * .28, S * .34, S * .40, S * .52], fill=dark)
    d.arc([S * .58, S * .38, S * .74, S * .54], 200, 340, fill=dark, width=W(S, 10))
    d.arc([S * .26, S * .42, S * .74, S * .78], 20, 160, fill=(60, 30, 15), width=W(S, 10))
    return im


def tex_tongue(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (50, 25, 10)
    d.ellipse([S * .28, S * .32, S * .40, S * .50], fill=dark)
    d.ellipse([S * .60, S * .32, S * .72, S * .50], fill=dark)
    d.chord([S * .28, S * .50, S * .72, S * .80], 0, 180, fill=(150, 20, 30),
            outline=(50, 20, 10), width=W(S, 6))
    d.ellipse([S * .40, S * .66, S * .60, S * .90], fill=(255, 100, 120),
              outline=(200, 60, 80), width=W(S, 4))
    d.line([(S * .50, S * .70), (S * .50, S * .82)], fill=(200, 60, 80), width=W(S, 4))
    return im


def tex_kiss(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (50, 25, 10)
    d.ellipse([S * .30, S * .34, S * .40, S * .52], fill=dark)
    d.arc([S * .58, S * .38, S * .74, S * .54], 200, 340, fill=dark, width=W(S, 10))
    d.ellipse([S * .44, S * .60, S * .56, S * .76], fill=(210, 40, 60))
    # small heart
    red = (235, 30, 60)
    cx, cy, r = int(S * .76), int(S * .66), int(S * .05)
    d.polygon([(cx - int(r * 1.4), cy), (cx + int(r * 1.4), cy), (cx, cy + int(r * 1.7))], fill=red)
    d.ellipse([cx - int(r * 1.4), cy - r, cx, cy + int(r * .4)], fill=red)
    d.ellipse([cx, cy - r, cx + int(r * 1.4), cy + int(r * .4)], fill=red)
    return im


def tex_sleepy(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    dark = (60, 30, 15)
    d.arc([S * .24, S * .36, S * .42, S * .52], 20, 160, fill=dark, width=W(S, 10))
    d.arc([S * .58, S * .36, S * .76, S * .52], 20, 160, fill=dark, width=W(S, 10))
    d.ellipse([S * .44, S * .62, S * .56, S * .74], fill=(120, 40, 30))
    zc = (70, 120, 220)
    d.line([(S * .70, S * .16), (S * .82, S * .16), (S * .70, S * .28), (S * .82, S * .28)],
           fill=zc, width=W(S, 7))
    return im


def tex_star(S):
    im = new_tex(S)
    d = ImageDraw.Draw(im)
    gold = (255, 150, 0)
    d.polygon(star_pts(S * .32, S * .42, S * .14), fill=gold, outline=(200, 90, 0))
    d.polygon(star_pts(S * .68, S * .42, S * .14), fill=gold, outline=(200, 90, 0))
    d.chord([S * .28, S * .50, S * .72, S * .80], 0, 180, fill=(150, 20, 30),
            outline=(50, 20, 10), width=W(S, 6))
    d.rectangle([S * .32, S * .59, S * .68, S * .64], fill=(255, 255, 255))
    return im

# ==========================================================
# 3. 3D sphere renderer (shading + wrapping the face on a ball)
# ==========================================================
class Sphere:
    def __init__(self, N):
        self.N = N
        ys, xs = np.mgrid[0:N, 0:N].astype(np.float32)
        x = (xs + 0.5) / N * 2 - 1
        y = (ys + 0.5) / N * 2 - 1
        r2 = x * x + y * y
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        self.P = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)

        self.alpha = np.clip((1 - np.sqrt(r2)) * N / 2, 0, 1).astype(np.float32)

        L = np.array([-0.45, -0.55, 0.70], np.float32)
        L /= np.linalg.norm(L)
        n = self.P
        diff = np.clip(n @ L, 0, 1)
        Hv = L + np.array([0, 0, 1], np.float32)
        Hv /= np.linalg.norm(Hv)
        spec = np.clip(n @ Hv, 0, 1) ** 55
        rim = (1 - n[:, 2]) ** 2
        bounce = np.clip(n[:, 1], 0, 1) * 0.10
        self.light = (0.40 + 0.70 * diff - 0.30 * rim + bounce).reshape(N, N, 1).astype(np.float32)
        self.spec = (spec * 0.70).reshape(N, N, 1).astype(np.float32)

    def render(self, tex_bgr, R):
        N = self.N
        T = tex_bgr.shape[0]
        q = self.P @ R.astype(np.float32)          # un-rotate each point into face space
        mx = ((q[:, 0] + 1) / 2 * (T - 1)).reshape(N, N)
        my = ((q[:, 1] + 1) / 2 * (T - 1)).reshape(N, N)
        back = (q[:, 2] <= 0).reshape(N, N)        # far side of the ball = plain colour
        mx[back] = 0
        my[back] = 0
        col = cv2.remap(tex_bgr, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        col = col * self.light + self.spec * 255.0
        col = np.clip(col, 0, 255)
        out = np.dstack([col, self.alpha * 255.0])
        return out.astype(np.uint8)                # BGRA


def rotation(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


sphere = Sphere(RENDER_RES)
thumb_sphere = Sphere(200)


def build_catalog():
    makers = [("Crying", tex_cry), ("Joy", tex_joy), ("Heart Eyes", tex_heart),
              ("Cool", tex_cool), ("Angry", tex_angry), ("Surprise", tex_surprise),
              ("Smile", tex_smile), ("Wink", tex_wink), ("Tongue", tex_tongue),
              ("Kiss", tex_kiss), ("Sleepy", tex_sleepy), ("Star Eyes", tex_star)]
    cat = []
    for name, fn in makers:
        im = fn(TEX_RES * SS).resize((TEX_RES, TEX_RES), Image.LANCZOS).convert("RGB")
        tex = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        front = thumb_sphere.render(tex, np.eye(3))
        cat.append((name, tex, front))
    return cat


catalog = build_catalog()
selected_idx = 0

# ==========================================================
# 4. Head pose from landmarks (position, size, turn, nod, tilt)
# ==========================================================
def estimate_pose(lm, w, h):
    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)

    top, chin, left, right, nose = pt(10), pt(152), pt(234), pt(454), pt(1)
    face_h = np.linalg.norm(chin - top)
    face_w = np.linalg.norm(right - left)

    roll = math.atan2(right[1] - left[1], right[0] - left[0])

    across = right - left
    ratio_x = np.dot(nose - left, across) / (np.dot(across, across) + 1e-6)
    yaw = float(np.clip((ratio_x - 0.5) * YAW_GAIN, -1, 1)) * math.radians(YAW_RANGE)

    down = (chin - top) / (face_h + 1e-6)
    ratio_y = np.dot(nose - top, down) / (face_h + 1e-6)
    pitch = float(np.clip((ratio_y - PITCH_NEUTRAL) * PITCH_GAIN, -1, 1)) * math.radians(PITCH_RANGE)

    if FLIP_YAW:
        yaw = -yaw
    if FLIP_PITCH:
        pitch = -pitch

    # --- smallest circle that covers hair, ears, jaw and chin ---
    side = across / (face_w + 1e-6)          # unit vector along the head width
    pts = [
        top - down * face_h * HAIR,          # top of hair
        chin + down * face_h * CHIN,         # below chin
        left - side * face_w * EAR,          # left ear
        right + side * face_w * EAR,         # right ear
        pt(172) - side * face_w * 0.05 + down * face_h * 0.04,   # jaw left
        pt(397) + side * face_w * 0.05 + down * face_h * 0.04,   # jaw right
        pt(127) - side * face_w * EAR,       # side of head left
        pt(356) + side * face_w * EAR,       # side of head right
        pt(54) - down * face_h * HAIR * 0.7 - side * face_w * 0.1,   # upper hair left
        pt(284) - down * face_h * HAIR * 0.7 + side * face_w * 0.1,  # upper hair right
    ]
    (ccx, ccy), radius = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
    return np.array([ccx, ccy, 2.0 * radius * FIT, yaw, pitch, roll])
    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)

    top, chin, left, right, nose = pt(10), pt(152), pt(234), pt(454), pt(1)
    face_h = np.linalg.norm(chin - top)
    face_w = np.linalg.norm(right - left)

    roll = math.atan2(right[1] - left[1], right[0] - left[0])

    up = (top - chin) / (face_h + 1e-6)
    center = (top + chin) / 2 + up * face_h * LIFT

    across = right - left
    ratio_x = np.dot(nose - left, across) / (np.dot(across, across) + 1e-6)
    yaw = float(np.clip((ratio_x - 0.5) * YAW_GAIN, -1, 1)) * math.radians(YAW_RANGE)

    down = (chin - top) / (face_h + 1e-6)
    ratio_y = np.dot(nose - top, down) / (face_h + 1e-6)
    pitch = float(np.clip((ratio_y - PITCH_NEUTRAL) * PITCH_GAIN, -1, 1)) * math.radians(PITCH_RANGE)

    if FLIP_YAW:
        yaw = -yaw
    if FLIP_PITCH:
        pitch = -pitch

    d = max(face_h, face_w * 1.3) * SIZE
    return np.array([center[0], center[1], d, yaw, pitch, roll])


# ==========================================================
# 5. Compositing
# ==========================================================
def blend(frame, rgb, a, x, y):
    H, Wf = frame.shape[:2]
    h, w = a.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, Wf), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    c = rgb[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(c * sa + roi * (1 - sa), 0, 255).astype(np.uint8)


def draw_ball(frame, tex, cx, cy, d, yaw, pitch, roll):
    d = int(np.clip(d, 40, 1400))
    sprite = sphere.render(tex, rotation(yaw, pitch, roll))
    sprite = cv2.resize(sprite, (d, d), interpolation=cv2.INTER_AREA)
    alpha = sprite[..., 3].astype(np.float32) / 255.0
    x, y = int(cx - d / 2), int(cy - d / 2)

    # soft shadow under the ball
    sh = cv2.GaussianBlur(alpha, (0, 0), max(1.0, d * 0.05)) * 0.35
    blend(frame, np.zeros((d, d, 3), np.float32), sh, x, y + int(d * 0.05))

    blend(frame, sprite[..., :3], alpha, x, y)


# ==========================================================
# 6. Scrollable emoji tray (left / right / top / bottom)
# ==========================================================
ICON = 64            # fixed icon size in the tray (change this to resize icons)
GAP = 16             # space between icons
TRAY = 96            # tray thickness

scroll = 0.0
max_scroll = 0.0
dock_hits = []
dock_rect = (0, 0, 0, 0)
drag = {"down": False, "start": 0, "scroll0": 0.0, "moved": False}


def dock_geometry(fw, fh):
    horiz = dock_pos in ("top", "bottom")
    if horiz:
        rect = (0, 0 if dock_pos == "top" else fh - TRAY, fw, TRAY)
        view = fw
    else:
        rect = (0 if dock_pos == "left" else fw - TRAY, 0, TRAY, fh)
        view = fh
    content = len(catalog) * (ICON + GAP) + GAP
    return rect, horiz, view, content, max(0, content - view)


def clamp_scroll():
    global scroll
    scroll = float(np.clip(scroll, 0, max_scroll))


def ensure_visible(i):
    """Scroll so emoji number i is inside the tray."""
    global scroll
    fh, fw = last_frame_size
    rect, horiz, view, content, ms = dock_geometry(fw, fh)
    start = GAP + i * (ICON + GAP)
    if start < scroll:
        scroll = start - GAP
    elif start + ICON > scroll + view:
        scroll = start + ICON - view + GAP
    clamp_scroll()


def wheel_delta(flags):
    try:
        return cv2.getMouseWheelDelta(flags)
    except Exception:
        d = flags >> 16
        return d - 65536 if d > 32767 else d


last_frame_size = (480, 640)


def draw_dock(frame):
    global dock_hits, dock_rect, max_scroll, last_frame_size
    fh, fw = frame.shape[:2]
    last_frame_size = (fh, fw)
    rect, horiz, view, content, max_scroll = dock_geometry(fw, fh)
    clamp_scroll()
    dock_rect = rect
    rx, ry, rw, rh = rect

    roi = frame[ry:ry + rh, rx:rx + rw]
    dark = np.zeros_like(roi)
    dark[:] = (22, 22, 24)
    frame[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(roi, 0.25, dark, 0.75, 0)

    dock_hits = []
    for i, (name, tex, front) in enumerate(catalog):
        along = GAP + ICON / 2 + i * (ICON + GAP) - scroll
        if along < -ICON or along > view + ICON:
            dock_hits.append(None)
            continue
        cx, cy = (int(along), ry + rh // 2) if horiz else (rx + rw // 2, int(along))
        dock_hits.append((cx, cy, ICON // 2 + 6))
        if i == selected_idx:
            cv2.circle(frame, (cx, cy), ICON // 2 + 5, (0, 230, 255), 3, cv2.LINE_AA)
        thumb = cv2.resize(front, (ICON, ICON), interpolation=cv2.INTER_AREA)
        blend(frame, thumb[..., :3], thumb[..., 3].astype(np.float32) / 255.0,
              cx - ICON // 2, cy - ICON // 2)

    # scroll bar + arrows
    if max_scroll > 0:
        bar_len = max(30, int(view * view / content))
        bar_pos = int((view - bar_len) * scroll / max_scroll)
        if horiz:
            y = ry + rh - 5
            cv2.line(frame, (bar_pos, y), (bar_pos + bar_len, y), (0, 230, 255), 3, cv2.LINE_AA)
        else:
            x = rx + (4 if dock_pos == "right" else rw - 5)
            cv2.line(frame, (x, bar_pos), (x, bar_pos + bar_len), (0, 230, 255), 3, cv2.LINE_AA)

    tx = rx + rw + 14 if dock_pos == "left" else 14
    ty = ry + rh + 28 if dock_pos == "top" else 30
    text = "Scroll / drag tray | , . keys | 1-9 pick | D = move tray | Q = quit"
    cv2.putText(frame, text, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param):
    global selected_idx, scroll
    rx, ry, rw, rh = dock_rect
    inside = rx <= x < rx + rw and ry <= y < ry + rh
    horiz = dock_pos in ("top", "bottom")
    pos = x if horiz else y

    if event == cv2.EVENT_MOUSEWHEEL and inside:
        scroll -= (1 if wheel_delta(flags) > 0 else -1) * (ICON + GAP)
        clamp_scroll()

    elif event == cv2.EVENT_LBUTTONDOWN and inside:
        drag.update(down=True, start=pos, scroll0=scroll, moved=False)

    elif event == cv2.EVENT_MOUSEMOVE and drag["down"]:
        delta = pos - drag["start"]
        if abs(delta) > 6:
            drag["moved"] = True
        if drag["moved"]:
            scroll = drag["scroll0"] - delta
            clamp_scroll()

    elif event == cv2.EVENT_LBUTTONUP and drag["down"]:
        drag["down"] = False
        if not drag["moved"]:              # plain click = select emoji
            for i, hit in enumerate(dock_hits):
                if hit and (x - hit[0]) ** 2 + (y - hit[1]) ** 2 <= hit[2] ** 2:
                    selected_idx = i
                    break
# ==========================================================
# 7. Main loop
# ==========================================================
WINDOW = "3D Emoji Head"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not found. Try cv2.VideoCapture(1) or close other apps using the camera.")

cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)

state, missed = None, 0
t0, last_ts = time.time(), -1

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = max(int((time.time() - t0) * 1000), last_ts + 1)
    last_ts = ts
    result = detector.detect_for_video(mp_image, ts)

    if result.face_landmarks:
        new = estimate_pose(result.face_landmarks[0], w, h)
        state = new if state is None else state + (new - state) * SMOOTH
        missed = 0
    else:
        missed += 1
        if missed > HOLD_FRAMES:
            state = None

    if state is not None:
        cx, cy, d, yaw, pitch, roll = state
        draw_ball(frame, catalog[selected_idx][1], cx, cy, d, yaw, pitch, roll)

    draw_dock(frame)
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('d'):
        dock_pos = DOCK_POSITIONS[(DOCK_POSITIONS.index(dock_pos) + 1) % len(DOCK_POSITIONS)]
    elif ord('1') <= key <= ord('9') and (key - ord('1')) < len(catalog):
        selected_idx = key - ord('1')
    elif key == ord('d'):
        dock_pos = DOCK_POSITIONS[(DOCK_POSITIONS.index(dock_pos) + 1) % len(DOCK_POSITIONS)]
        scroll = 0.0
    elif key == ord(','):
        scroll -= ICON + GAP
        clamp_scroll()
    elif key == ord('.'):
        scroll += ICON + GAP
        clamp_scroll()
    elif ord('1') <= key <= ord('9') and (key - ord('1')) < len(catalog):
        selected_idx = key - ord('1')
        ensure_visible(selected_idx)


cap.release()
cv2.destroyAllWindows()
detector.close()