import cv2
import numpy as np

# ---------- Utility ----------
def ensure_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

def overlay_rgba(base_bgr, overlay_rgba, x, y):
    if overlay_rgba is None:
        return base_bgr
    H, W = base_bgr.shape[:2]
    h, w = overlay_rgba.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return base_bgr
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = base_bgr[y1:y2, x1:x2].astype(np.float32)
    ov  = overlay_rgba[oy1:oy2, ox1:ox2].astype(np.float32)
    alpha = ov[..., 3:4] / 255.0
    out = ov[..., :3] * alpha + roi * (1 - alpha)
    base_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return base_bgr

# ---------- Face Detector ----------
class FaceDetector:
    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=(80,80)):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                             "haarcascade_frontalface_default.xml")
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        self.show_boxes = True

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors, minSize=self.minSize
        )
        return faces

    @staticmethod
    def draw_boxes(img, faces, color=(0,220,255), thickness=2):
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)
        return img

# ---------- Base Filter ----------
class Filter:
    NAME = "Base"
    KEY  = None
    def apply(self, frame_bgr, context): return frame_bgr

# ---------- BEAUTIFY: base ----------
def beautify(frame):
    smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    sharp  = cv2.addWeighted(smooth, 1.15, cv2.GaussianBlur(smooth, (0,0), 3), -0.15, 0)
    return sharp

# ---------- Panel: 1 (Beautiful) ----------
class BeautyPanel:
    name = "Beautify Controls"
    shown = False
    @staticmethod
    def create():
        if BeautyPanel.shown: return
        cv2.namedWindow(BeautyPanel.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(BeautyPanel.name, 420, 220)
        cv2.createTrackbar("Smooth(+)",  BeautyPanel.name, 0, 100, lambda v: None)
        cv2.createTrackbar("Sharp(+)",   BeautyPanel.name, 0, 100, lambda v: None)
        cv2.createTrackbar("Saturation", BeautyPanel.name, 10, 100, lambda v: None)
        cv2.createTrackbar("Whiten",     BeautyPanel.name, 50, 100, lambda v: None)
        BeautyPanel.shown = True
    @staticmethod
    def destroy():
        if BeautyPanel.shown:
            cv2.destroyWindow(BeautyPanel.name)
            BeautyPanel.shown = False
    @staticmethod
    def read():
        s_smooth = cv2.getTrackbarPos("Smooth(+)",  BeautyPanel.name)
        s_sharp  = cv2.getTrackbarPos("Sharp(+)",   BeautyPanel.name)
        s_sat    = cv2.getTrackbarPos("Saturation", BeautyPanel.name)
        s_white  = cv2.getTrackbarPos("Whiten",     BeautyPanel.name)
        sigma_add = np.interp(s_smooth, [0,100], [0,120])
        d_add     = int(np.interp(s_smooth, [0,100], [5,11])) | 1
        sharp_amt = np.interp(s_sharp,  [0,100], [0.0, 1.0])
        sat_gain  = np.interp(s_sat,    [0,50,100], [0.6,1.2,1.8])
        gamma     = np.interp(s_white,  [0,50,100], [0.90,1.00,1.30])
        return {"sigma_add": sigma_add, "d_add": d_add,
                "sharp_amt": sharp_amt, "sat_gain": sat_gain,
                "gamma": float(gamma)}

# ---------- Panel: 2 (Black&White) ----------
class BWPanel:
    name = "B&W Controls"
    shown = False
    @staticmethod
    def create():
        if BWPanel.shown: return
        cv2.namedWindow(BWPanel.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(BWPanel.name, 360, 140)
        cv2.createTrackbar("Blacks", BWPanel.name, 0,   100, lambda v: None)
        cv2.createTrackbar("Whites", BWPanel.name, 100, 100, lambda v: None)
        BWPanel.shown = True
    @staticmethod
    def destroy():
        if BWPanel.shown:
            cv2.destroyWindow(BWPanel.name)
            BWPanel.shown = False
    @staticmethod
    def read():
        s_b = cv2.getTrackbarPos("Blacks", BWPanel.name)
        s_w = cv2.getTrackbarPos("Whites", BWPanel.name)
        black_point = int(np.interp(s_b, [0,100], [0,120]))
        white_point = int(np.interp(s_w, [0,100], [155,255]))
        if white_point <= black_point + 1:
            white_point = black_point + 2
        return {"bp": black_point, "wp": white_point}

# ---------- Panel: 3 (Blur) ----------
class BlurPanel:
    name = "Blur Controls"
    shown = False
    @staticmethod
    def create():
        if BlurPanel.shown: return
        cv2.namedWindow(BlurPanel.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(BlurPanel.name, 320, 100)
        cv2.createTrackbar("Blur", BlurPanel.name, 40, 100, lambda v: None)
        BlurPanel.shown = True
    @staticmethod
    def destroy():
        if BlurPanel.shown:
            cv2.destroyWindow(BlurPanel.name)
            BlurPanel.shown = False
    @staticmethod
    def read():
        s = cv2.getTrackbarPos("Blur", BlurPanel.name)
        ksize = int(np.interp(s, [0,100], [1,61]))
        if ksize % 2 == 0: ksize += 1
        sigma = float(np.interp(s, [0,100], [0.0, 20.0]))
        return {"ksize": ksize, "sigma": sigma}

# ---------- Panel: 4 (Vintage) ----------
class VintagePanel:
    name = "Vintage Controls"
    shown = False
    @staticmethod
    def create():
        if VintagePanel.shown: return
        cv2.namedWindow(VintagePanel.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VintagePanel.name, 420, 180)
        cv2.createTrackbar("Sepia Mix",  VintagePanel.name, 60, 100, lambda v: None)
        cv2.createTrackbar("Warmth",     VintagePanel.name, 50, 100, lambda v: None)
        cv2.createTrackbar("Fade",       VintagePanel.name, 20, 100, lambda v: None)
        cv2.createTrackbar("Grain",      VintagePanel.name, 15, 100, lambda v: None)
        cv2.createTrackbar("Vig.Str",    VintagePanel.name, 35, 100, lambda v: None)
        cv2.createTrackbar("Vig.Size",   VintagePanel.name, 50, 100, lambda v: None)
        VintagePanel.shown = True
    @staticmethod
    def destroy():
        if VintagePanel.shown:
            cv2.destroyWindow(VintagePanel.name)
            VintagePanel.shown = False
    @staticmethod
    def read():
        s_sepia = cv2.getTrackbarPos("Sepia Mix", VintagePanel.name)
        s_warm  = cv2.getTrackbarPos("Warmth",   VintagePanel.name)
        s_fade  = cv2.getTrackbarPos("Fade",     VintagePanel.name)
        s_grain = cv2.getTrackbarPos("Grain",    VintagePanel.name)
        s_vs    = cv2.getTrackbarPos("Vig.Str",  VintagePanel.name)
        s_vz    = cv2.getTrackbarPos("Vig.Size", VintagePanel.name)
        sepia_mix = float(s_sepia) / 100.0
        warmth    = np.interp(s_warm, [0,50,100], [-20, 0, +20])
        fade_amt  = np.interp(s_fade, [0,100], [0.0, 0.35])
        grain_std = np.interp(s_grain,[0,100], [0.0, 20.0])
        vig_str   = np.interp(s_vs,   [0,100], [0.0, 0.6])
        size_k    = np.interp(s_vz,   [0,100], [1.0, 2.0])
        return {"sepia": sepia_mix, "warmth": warmth, "fade": fade_amt,
                "grain": grain_std, "vig_str": vig_str, "vig_size": size_k}

# ---------- Panel: 5 (Sticker: Size + Offsets) ----------
class StickerPanel:
    name = "Sticker Controls"
    shown = False
    @staticmethod
    def create(initial_percent=115):
        if StickerPanel.shown:
            return
        cv2.namedWindow(StickerPanel.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(StickerPanel.name, 360, 150)
        # Size %: 60..200 (100 = เท่าความกว้างหน้า)
        init = max(60, min(200, int(initial_percent)))
        cv2.createTrackbar("Size %", StickerPanel.name, init, 200, lambda v: None)
        # OffX / OffY: 0..200 → map เป็น -0.5..+0.5 ของ w/h
        cv2.createTrackbar("OffX", StickerPanel.name, 100, 200, lambda v: None)  # ซ้าย/ขวา
        cv2.createTrackbar("OffY", StickerPanel.name, 100, 200, lambda v: None)  # ขึ้น/ลง
        StickerPanel.shown = True
    @staticmethod
    def destroy():
        if StickerPanel.shown:
            cv2.destroyWindow(StickerPanel.name)
            StickerPanel.shown = False
    @staticmethod
    def read():
        size_percent = max(60, cv2.getTrackbarPos("Size %", StickerPanel.name))
        # map 0..200 → -1..+1 แล้วคูณ 0.5 = -0.5..+0.5 (สัดส่วนของ w/h)
        offx = (cv2.getTrackbarPos("OffX", StickerPanel.name) - 100) / 100.0 * 0.5
        offy = (cv2.getTrackbarPos("OffY", StickerPanel.name) - 100) / 100.0 * 0.5
        return {"size_percent": size_percent, "offx": offx, "offy": offy}

# ---------- Filters ----------
class Beautiful(Filter):
    NAME, KEY = "Beautiful", '1'
    @staticmethod
    def _gamma(img, g=1.0):
        g = max(g, 1e-6)
        inv = 1.0 / g
        table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    @staticmethod
    def _saturate(img, gain=1.0):
        if abs(gain-1.0) < 1e-3: return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * gain, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    def apply(self, frame_bgr, context):
        base = beautify(frame_bgr)
        if not BeautyPanel.shown:
            return base
        p = BeautyPanel.read()
        out = base
        if p["sigma_add"] > 0:
            out = cv2.bilateralFilter(out, d=p["d_add"],
                                      sigmaColor=p["sigma_add"], sigmaSpace=p["sigma_add"])
        if abs(p["gamma"]-1.0) > 1e-3:
            out = self._gamma(out, g=p["gamma"])
        out = self._saturate(out, gain=p["sat_gain"])
        if p["sharp_amt"] > 0:
            blur = cv2.GaussianBlur(out, (0,0), 1.6)
            out  = cv2.addWeighted(out, 1.0 + p["sharp_amt"], blur, -p["sharp_amt"], 0)
        return out

class BlackWhite(Filter):
    NAME, KEY = "Black&White", '2'
    lut = (1/(1+np.exp(-10*(np.linspace(0,1,256,dtype=np.float32)-0.5))))
    lut = ((lut - lut.min())/(lut.max()-lut.min())*255).astype(np.uint8)
    def _apply_bw_points(self, g, bp, wp):
        g = g.astype(np.float32)
        g = (g - bp) * (255.0 / max(1.0, (wp - bp)))
        return np.clip(g, 0, 255).astype(np.uint8)
    def apply(self, frame_bgr, context):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if BWPanel.shown:
            p = BWPanel.read()
            g = self._apply_bw_points(g, p["bp"], p["wp"])
        g = cv2.LUT(g, self.lut)
        noise = np.random.normal(0, 8, g.shape).astype(np.int16)
        g = np.clip(g.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        H, W = g.shape
        y, x = np.ogrid[:H, :W]
        cy, cx = H/2, W/2
        mask = ((x-cx)**2/(W/1.5)**2 + (y-cy)**2/(H/1.5)**2)
        mask = np.clip(1 - 0.4*mask, 0, 1)
        mask = cv2.GaussianBlur(mask, (0,0), max(H,W)/20)
        g = (g.astype(np.float32)*mask).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

class BlurAesthetic(Filter):
    NAME, KEY = "Blur", '3'
    def apply(self, frame_bgr, context):
        if BlurPanel.shown:
            p = BlurPanel.read()
            k = max(1, p["ksize"])
            if k % 2 == 0: k += 1
            if k <= 1:
                return frame_bgr
            return cv2.GaussianBlur(frame_bgr, (k, k), p["sigma"])
        else:
            return cv2.GaussianBlur(frame_bgr, (21, 21), 8.0)

class Vintage(Filter):
    NAME, KEY = "Vintage", '4'
    @staticmethod
    def _sepia(img):
        k = np.array([[0.272,0.534,0.131],
                      [0.349,0.686,0.168],
                      [0.393,0.769,0.189]])
        out = cv2.transform(img, k)
        return np.clip(out, 0, 255).astype(np.uint8)
    @staticmethod
    def _warm(img, warmth_delta=0):
        if abs(warmth_delta) < 1e-6: return img
        b,g,r = cv2.split(img.astype(np.int16))
        r = np.clip(r + warmth_delta, 0, 255)
        b = np.clip(b - warmth_delta, 0, 255)
        return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
    @staticmethod
    def _fade(img, amt=0.0):
        if amt <= 0: return img
        return cv2.addWeighted(img, 1.0-amt, np.full_like(img, 127), amt, 0)
    @staticmethod
    def _grain(img, std=0.0):
        if std <= 0: return img
        noise = np.random.normal(0, std, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    @staticmethod
    def _vignette(img, strength=0.0, size_k=1.6):
        if strength <= 0: return img
        H, W = img.shape[:2]
        y, x = np.ogrid[:H, :W]
        cy, cx = H/2, W/2
        rx, ry = (W/size_k), (H/size_k)
        mask = ((x-cx)**2/(rx**2) + (y-cy)**2/(ry**2))
        mask = np.clip(1 - strength*mask, 0, 1)
        mask = cv2.GaussianBlur(mask, (0,0), max(H,W)/20)
        return np.clip(img.astype(np.float32) * mask[...,None], 0, 255).astype(np.uint8)
    def apply(self, frame_bgr, context):
        if VintagePanel.shown:
            p = VintagePanel.read()
            sep = self._sepia(frame_bgr)
            out = cv2.addWeighted(frame_bgr, 1.0 - p["sepia"], sep, p["sepia"], 0)
            out = self._warm(out, p["warmth"])
            out = self._fade(out, p["fade"])
            out = self._grain(out, p["grain"])
            out = self._vignette(out, p["vig_str"], p["vig_size"])
            return out
        else:
            out = self._sepia(frame_bgr)
            out = self._warm(out, warmth_delta=+10)
            out = self._fade(out, amt=0.15)
            out = self._grain(out, std=5.0)
            out = self._vignette(out, strength=0.35, size_k=1.3)
            return out

class StickerCat(Filter):
    NAME, KEY = "Sticker", '5'
    def __init__(self,path="C:\Dog.png"):
        self.png = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        if self.png is not None and (self.png.shape[2]!=4):
            self.png=None
    def apply(self, frame_bgr, context):
        if self.png is None: return frame_bgr
        faces = context.get("faces",[])
        if len(faces)==0: return frame_bgr
        
        x,y,w,h = max(faces,key=lambda b:b[2]*b[3])

        # อ่านค่า Size/Offset จากแผง (ถ้าอยู่โหมด 5)
        if StickerPanel.shown:
            p = StickerPanel.read()
            size_percent = p["size_percent"]      # 60..200 (% ของ w)
            offx_ratio   = p["offx"]              # -0.5..+0.5 (ของ w)
            offy_ratio   = p["offy"]              # -0.5..+0.5 (ของ h)
        else:
            size_percent = 115
            offx_ratio = 0.0
            offy_ratio = -0.05  # offset เดิมเล็กน้อยขึ้นด้านบน

        # คำนวณขนาดสติกเกอร์
        target_w = int((size_percent/100.0) * w)
        target_w = max(10, target_w)
        scale    = target_w / self.png.shape[1]
        target_h = max(1, int(self.png.shape[0]*scale))
        sticker  = cv2.resize(self.png, (target_w,target_h), interpolation=cv2.INTER_AREA)

        # ตำแหน่งกึ่งกลางหน้า + ออฟเซ็ต
        tx = x + (w - target_w)//2 + int(offx_ratio * w)
        ty = y + (h - target_h)//2 + int(offy_ratio * h)
        return overlay_rgba(frame_bgr,sticker,tx,ty)

# ---------- Register ----------
REGISTERED = [Beautiful(), BlackWhite(), BlurAesthetic(), Vintage(), StickerCat()]

def build_hotkey_map(filters): return {f.KEY:f.NAME for f in filters if f.KEY}
def build_name_map(filters): return {f.NAME:f for f in filters}
def help_line(filters):
    items=[f"[{f.KEY}]{f.NAME}" for f in filters if f.KEY]
    return " ".join(items)+" [0]None [F]FaceBox [S]Save [Q]Quit"

# ---------- Main ----------
def main():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    face=FaceDetector()
    hotkeys=build_hotkey_map(REGISTERED)
    name_map=build_name_map(REGISTERED)

    active="None"; save_count=0; help1=help_line(REGISTERED)

    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        faces=face.detect(frame)
        out=frame.copy()

        if active!="None":
            filt=name_map.get(active)
            if filt: out=filt.apply(out,{"faces":faces})

        canvas=ensure_bgr(out)
        if face.show_boxes and len(faces)>0: canvas=FaceDetector.draw_boxes(canvas,faces)
        cv2.putText(canvas,help1,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(canvas,f"Active: {active} FaceBox:{'ON' if face.show_boxes else 'OFF'}",
                    (10,56),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Video Filters + FaceBox",canvas)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break
        elif key==ord('0'):
            # ออกจากทุกโหมด + ปิดทุกแผง
            active="None"
            BeautyPanel.destroy(); BWPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()
        elif key in (ord('f'),ord('F')):
            face.show_boxes=not face.show_boxes
        elif key in (ord('s'),ord('S')):
            fn=f"frame_{save_count:03d}.png"; cv2.imwrite(fn,canvas)
            print("Saved:",fn); save_count+=1
        else:
            ch=chr(key) if 32<=key<=126 else None
            if ch and ch in hotkeys:
                new_active = hotkeys[ch]
                # เปิด/ปิด panels ตามโหมด
                if new_active == "Beautiful":
                    BeautyPanel.create(); BWPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()
                elif new_active == "Black&White":
                    BWPanel.create(); BeautyPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()
                elif new_active == "Blur":
                    BlurPanel.create(); BeautyPanel.destroy(); BWPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()
                elif new_active == "Vintage":
                    VintagePanel.create(); BeautyPanel.destroy(); BWPanel.destroy(); BlurPanel.destroy(); StickerPanel.destroy()
                elif new_active == "Sticker":
                    StickerPanel.create(initial_percent=115)
                    BeautyPanel.destroy(); BWPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy()
                else:
                    BeautyPanel.destroy(); BWPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()
                active = new_active

    cap.release(); cv2.destroyAllWindows()
    BeautyPanel.destroy(); BWPanel.destroy(); BlurPanel.destroy(); VintagePanel.destroy(); StickerPanel.destroy()  # กันลืม

if __name__=="__main__":
    main()
