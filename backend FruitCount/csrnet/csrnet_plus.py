# CSRNet++ : CSRNet cai tien cho dem qua (MinneApple) - model + dataset + train + suy luan
import os, sys, json, glob, math, time, random, copy, warnings
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

STRIDE = 4                                                   # do phan giai density map = 1/4 anh
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255     # RGB
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
MEAN_BGR = tuple(float(x) for x in MEAN[::-1])

# ------------------------------------------------------------------ MODEL
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 8)), nn.ReLU(True), nn.Linear(max(c // r, 8), c), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x.mean((2, 3)))
        return x * w[:, :, None, None]

class IBN(nn.Module):
    # IBN-a: mot nua kenh dung InstanceNorm (bat bien anh sang/mau), mot nua giu BatchNorm (khoi tao tu BN pretrained)
    def __init__(self, bn):
        super().__init__()
        c = bn.num_features; self.half = c // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(c - self.half)
        with torch.no_grad():
            self.IN.weight.copy_(bn.weight[:self.half]); self.IN.bias.copy_(bn.bias[:self.half])
            self.BN.weight.copy_(bn.weight[self.half:]); self.BN.bias.copy_(bn.bias[self.half:])
            self.BN.running_mean.copy_(bn.running_mean[self.half:]); self.BN.running_var.copy_(bn.running_var[self.half:])
    def forward(self, x):
        a, b = torch.split(x, [self.half, x.shape[1] - self.half], 1)
        return torch.cat([self.IN(a), self.BN(b.contiguous())], 1)

def cbr(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class ASPP(nn.Module):
    # cac nhanh dilation song song (khong dung image-pooling de on dinh giua crop va anh day du)
    def __init__(self, i, o, rates=(1, 2, 4, 6)):
        super().__init__()
        self.br = nn.ModuleList([cbr(i, o // len(rates), 1 if r == 1 else 3, r) for r in rates])
        self.proj = cbr((o // len(rates)) * len(rates), o, 1)
    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.br], 1))

class ImprovedCSRNet(nn.Module):
    # frontend: VGG16-BN (conv1_1 .. conv4_3), IBN o block 1-2, SE
    # backend : ASPP -> 3 conv dilation=2 (kieu CSRNet) -> SE -> upsample x2 + skip conv3_3 -> density map (stride 4)
    def __init__(self, pretrained=True, use_ibn=True, use_se=True, aspp_rates=(1, 2, 4, 6)):
        super().__init__()
        w = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        feats = list(torchvision.models.vgg16_bn(weights=w).features.children())
        if use_ibn:
            for i in (1, 4, 8, 11): feats[i] = IBN(feats[i])
        self.stage3 = nn.Sequential(*feats[:23])      # -> conv3_3, stride 4, 256 kenh
        self.stage4 = nn.Sequential(*feats[23:33])    # -> conv4_3, stride 8, 512 kenh
        self.se3 = SE(256) if use_se else nn.Identity()
        self.se4 = SE(512) if use_se else nn.Identity()
        self.aspp = ASPP(512, 256, aspp_rates)
        self.backend = nn.Sequential(cbr(256, 256, 3, 2), cbr(256, 128, 3, 2), cbr(128, 128, 3, 2))
        self.se_b = SE(128) if use_se else nn.Identity()
        self.lateral = cbr(256, 128, 1)
        self.fuse = nn.Sequential(cbr(256, 128, 3), cbr(128, 64, 3))
        self.head = nn.Conv2d(64, 1, 1)
        for m in [self.aspp, self.backend, self.lateral, self.fuse, self.head]:
            for l in m.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu")
                    if l.bias is not None: nn.init.zeros_(l.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def frontend_params(self):
        return list(self.stage3.parameters()) + list(self.stage4.parameters())
    def head_params(self):
        fid = {id(p) for p in self.frontend_params()}
        return [p for p in self.parameters() if id(p) not in fid]

    def forward(self, x):
        c3 = self.se3(self.stage3(x))
        c4 = self.se4(self.stage4(c3))
        y = self.se_b(self.backend(self.aspp(c4)))
        y = F.interpolate(y, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([y, self.lateral(c3)], 1))
        return F.relu(self.head(y))

def load_model(path, dev):
    ck = torch.load(path, map_location="cpu")
    m = ImprovedCSRNet(pretrained=False, **ck["kw"]); m.load_state_dict(ck["state"])
    return m.to(dev).eval()

# ------------------------------------------------------------------ DU LIEU: nhan YOLO (bbox) -> density map
def lab_path(img):
    s = str(img); i = s.rfind("/images/")
    return (Path(s[:i] + "/labels/" + s[i + 8:]) if i >= 0 else Path(s)).with_suffix(".txt")

def load_labels(img_path):
    lp = lab_path(img_path)
    if not lp.exists() or lp.stat().st_size == 0: return np.zeros((0, 4), np.float32)
    return np.loadtxt(lp, ndmin=2)[:, 1:5].astype(np.float32)      # xc, yc, w, h (chuan hoa)

def make_density(h, w, pts, sizes, k=0.25, smin=4.0, smax=16.0, stride=STRIDE):
    # moi qua = 1 Gaussian tong = 1 (giu nguyen so luong); sigma ti le voi kich thuoc bbox
    hs, ws = math.ceil(h / stride), math.ceil(w / stride)
    d = np.zeros((hs, ws), np.float32)
    for (x, y), s in zip(pts, sizes):
        cx, cy = x / stride, y / stride
        if not (0 <= cx < ws and 0 <= cy < hs): continue
        sg = max(float(np.clip(k * s, smin, smax)) / stride, 0.8); r = int(math.ceil(3 * sg)) + 1
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, ws); y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, hs)
        g = np.exp(-(((np.arange(y0, y1) + .5)[:, None] - cy) ** 2 + ((np.arange(x0, x1) + .5)[None, :] - cx) ** 2) / (2 * sg * sg))
        d[y0:y1, x0:x1] += g / max(g.sum(), 1e-9)
    return d

def hsv_jitter(img, h=0.015, s=0.6, v=0.4):
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)); x = np.arange(256, dtype=r.dtype)
    out = cv2.merge((cv2.LUT(hue, ((x * r[0]) % 180).astype(np.uint8)), cv2.LUT(sat, np.clip(x * r[1], 0, 255).astype(np.uint8)),
                     cv2.LUT(val, np.clip(x * r[2], 0, 255).astype(np.uint8))))
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

def to_tensor(img_bgr):
    return torch.from_numpy(((img_bgr[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())

class CountDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, c):
        self.imgs = [str(p) for p in imgs]; self.lab = [load_labels(p) for p in imgs]; self.c = c
    def __len__(self): return self.c.get("samples_per_epoch") or len(self.imgs) * self.c["crops_per_img"]
    def __getitem__(self, i):
        c = self.c
        i = random.randrange(len(self.imgs)) if c.get("samples_per_epoch") else i % len(self.imgs)
        img = cv2.imread(self.imgs[i]); H, W = img.shape[:2]; L = self.lab[i]
        pts = np.stack([L[:, 0] * W, L[:, 1] * H], 1).astype(np.float32); sz = np.sqrt(L[:, 2] * W * L[:, 3] * H).astype(np.float32)
        s = random.uniform(*c["scale_range"])
        if abs(s - 1) > 0.02:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR); pts = pts * s; sz = sz * s; H, W = img.shape[:2]
        if random.random() < c["fliplr"]: img = img[:, ::-1]; pts[:, 0] = W - pts[:, 0]
        if random.random() < c["flipud"]: img = img[::-1]; pts[:, 1] = H - pts[:, 1]
        img = np.ascontiguousarray(img); C = c["crop"]
        ph, pw = max(C - H, 0), max(C - W, 0)
        if ph or pw: img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=MEAN_BGR); H, W = img.shape[:2]
        y0, x0 = random.randint(0, H - C), random.randint(0, W - C)
        img = img[y0:y0 + C, x0:x0 + C]; pts = pts - np.array([x0, y0], np.float32)
        k = (pts[:, 0] >= 0) & (pts[:, 0] < C) & (pts[:, 1] >= 0) & (pts[:, 1] < C)
        den = make_density(C, C, pts[k], sz[k], c["sigma_k"], c["sigma_min"], c["sigma_max"])
        if random.random() < 0.8: img = hsv_jitter(img)
        return to_tensor(img), torch.from_numpy(den)[None]

# ------------------------------------------------------------------ SUY LUAN (full / tta / tiled / tiled_full)
def _pad(img, mult=8, min_size=0):
    H, W = img.shape[:2]; Hp = max(math.ceil(H / mult) * mult, min_size); Wp = max(math.ceil(W / mult) * mult, min_size)
    return cv2.copyMakeBorder(img, 0, Hp - H, 0, Wp - W, cv2.BORDER_CONSTANT, value=MEAN_BGR) if (Hp, Wp) != (H, W) else img

def _fwd(model, x, dev):
    with torch.autocast("cuda", enabled=(dev.type == "cuda")):
        return model(x.to(dev)).float()

@torch.no_grad()
def density_full(model, img, dev):
    H, W = img.shape[:2]
    y = _fwd(model, to_tensor(_pad(img))[None], dev)[0, 0].cpu().numpy()
    return y[:math.ceil(H / STRIDE), :math.ceil(W / STRIDE)]

@torch.no_grad()
def density_tiled(model, img, dev, tile=512, overlap=0.25, bs=8):
    H, W = img.shape[:2]; p = _pad(img, 8, tile); Hp, Wp = p.shape[:2]
    step = max(int(tile * (1 - overlap)) // 8 * 8, 8)
    xs = list(range(0, Wp - tile + 1, step)); ys = list(range(0, Hp - tile + 1, step))
    if xs[-1] + tile < Wp: xs.append(Wp - tile)
    if ys[-1] + tile < Hp: ys.append(Hp - tile)
    offs = [(x, y) for y in ys for x in xs]; t = tile // STRIDE
    hw = np.hanning(t + 2)[1:-1].astype(np.float32); win = 0.1 + 0.9 * np.outer(hw, hw)
    acc = np.zeros((Hp // STRIDE, Wp // STRIDE), np.float32); wsum = np.zeros_like(acc)
    for i in range(0, len(offs), bs):
        part = offs[i:i + bs]
        out = _fwd(model, torch.stack([to_tensor(p[y:y + tile, x:x + tile]) for x, y in part]), dev)[:, 0].cpu().numpy()
        for (x, y), o in zip(part, out):
            gx, gy = x // STRIDE, y // STRIDE; acc[gy:gy + t, gx:gx + t] += o * win; wsum[gy:gy + t, gx:gx + t] += win
    return (acc / np.maximum(wsum, 1e-6))[:math.ceil(H / STRIDE), :math.ceil(W / STRIDE)]

@torch.no_grad()
def predict_density(model, img, mode, dev, tile=512, overlap=0.25):
    if mode == "full": return density_full(model, img, dev)
    if mode == "tta":
        a = density_full(model, img, dev)
        b = density_full(model, np.ascontiguousarray(img[:, ::-1]), dev)[:, ::-1]
        c = density_full(model, np.ascontiguousarray(img[::-1]), dev)[::-1]
        # anh lat co the lech 1 o do pad -> cat ve cung kich thuoc
        h, w = a.shape; return (a + b[:h, :w] + c[:h, :w]) / 3
    if mode == "tiled": return density_tiled(model, img, dev, tile, overlap)
    if mode == "tiled_full": return 0.5 * (density_tiled(model, img, dev, tile, overlap) + density_full(model, img, dev))
    raise ValueError(mode)

def count_metrics(p, g):
    p = np.asarray(p, float); g = np.asarray(g, float); e = p - g; nz = g > 0
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float((np.abs(e[nz]) / g[nz]).mean() * 100) if nz.any() else float("nan"),
                R2=float(1 - (e ** 2).sum() / max(((g - g.mean()) ** 2).sum(), 1e-9)))

# ------------------------------------------------------------------ TRAIN (chay trong tien trinh rieng, 1 GPU)
class ModelEMA:
    def __init__(self, model, decay=0.998, tau=500):
        self.ema = copy.deepcopy(model).eval(); self.decay, self.tau, self.n = decay, tau, 0
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = self.decay * (1 - math.exp(-self.n / self.tau)); msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point: v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else: v.copy_(msd[k])

def _load(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")

def _save_atomic(obj, path):
    tmp = str(path) + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def train_one(cfg_path):
    import pandas as pd
    c = json.load(open(cfg_path)); key = c["key"]
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rd = lambda f: [Path(l) for l in Path(f).read_text().split("\n") if l.strip()]
    tr_imgs, va_imgs = rd(c["train_list"]), rd(c["val_list"])
    va_data = [cv2.imread(str(p)) for p in va_imgs]; va_gt = np.array([len(load_labels(p)) for p in va_imgs])
    kw = dict(use_ibn=c["use_ibn"], use_se=c["use_se"], aspp_rates=tuple(c["aspp_rates"]))
    model = ImprovedCSRNet(pretrained=c["pretrained"], **kw).to(dev); ema = ModelEMA(model)
    ds = CountDataset(tr_imgs, c)
    dl = torch.utils.data.DataLoader(ds, batch_size=c["batch"], shuffle=True, num_workers=c["workers"], drop_last=True,
                                     pin_memory=(dev.type == "cuda"), persistent_workers=c["workers"] > 0,
                                     worker_init_fn=lambda w: (random.seed(c["seed"] + w + int(time.time()) % 1000), np.random.seed((c["seed"] + w) % 2 ** 31)))
    out = Path(c["out"]) / "runs" / key; out.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda")); S, LAM = c["density_scale"], c["count_loss_w"]
    rp, lp = out / "resume.pth", out / "last.pth"
    hist, best, gep, opt_state, ck_phase, rewarm = [], 1e9, 0, None, None, False

    # ---- RESUME: (1) resume.pth = day du (model+EMA+optimizer) -> tiep tuc chinh xac
    #              (2) chi co last.pth (ban cu, chi co trong so EMA) -> tiep tuc gan dung, optimizer khoi tao lai + warmup 1 epoch
    try:
        if rp.exists():
            r = _load(rp); model.load_state_dict(r["model"]); ema.ema.load_state_dict(r["ema"]); ema.n = r["ema_n"]
            scaler.load_state_dict(r["scaler"]); hist, best, gep = r["hist"], r["best"], int(r["gep"]); opt_state, ck_phase = r["opt"], r["phase"]
            print(f"[{key}] RESUME day du tu epoch {gep}", flush=True)
        elif lp.exists():
            r = _load(lp); model.load_state_dict(r["state"]); ema.ema.load_state_dict(r["state"]); ema.n = 2000
            gep = int(r["epoch"]); hp = out / "history.csv"
            hist = [h for h in pd.read_csv(hp).to_dict("records") if h["epoch"] <= gep] if hp.exists() else []
            best = min([float(r.get("val_MAE", 1e9))] + [h["val_MAE"] for h in hist]); rewarm = True
            print(f"[{key}] RESUME gan dung tu last.pth (epoch {gep}, optimizer khoi tao lai)", flush=True)
    except Exception as e:
        print(f"[{key}] khong nap duoc checkpoint ({e}) -> train lai tu dau", flush=True)
        model = ImprovedCSRNet(pretrained=c["pretrained"], **kw).to(dev); ema = ModelEMA(model)
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda")); hist, best, gep, opt_state, rewarm = [], 1e9, 0, None, False

    offset = 0
    for phase, epochs, lr, lrf in (("A", c["epochs_a"], c["lr_a"], 0.1), ("B", c["epochs_b"], c["lr_b"], 0.02)):
        if epochs <= 0: continue
        if gep >= offset + epochs: offset += epochs; continue          # pha nay da xong
        frozen = phase == "A"
        for p in model.frontend_params(): p.requires_grad_(not frozen)
        opt = torch.optim.AdamW([dict(params=model.head_params(), lr=lr),
                                 dict(params=model.frontend_params(), lr=lr * c["backbone_mult"])], weight_decay=c["weight_decay"])
        base = [g["lr"] for g in opt.param_groups]                       # lay base lr TRUOC khi nap optimizer state
        if opt_state is not None and ck_phase == phase:
            try: opt.load_state_dict(opt_state)
            except Exception as e: print(f"[{key}] khong nap duoc optimizer ({e})", flush=True); rewarm = True
        opt_state = None
        nb = len(dl); total = epochs * nb; start_ep = gep - offset; it = start_ep * nb; it0 = it
        for ep in range(start_ep, epochs):
            model.train()
            if frozen: model.stage3.eval(); model.stage4.eval()      # pha A: dong bang backbone (ca BN)
            tl, tn, t0 = 0.0, 0, time.time()
            for x, d in dl:
                f = (it + 1) / nb if it < nb else lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * (it - nb) / max(total - nb, 1)))
                if rewarm: f *= min(1.0, (it - it0 + 1) / nb)
                for g, b in zip(opt.param_groups, base): g["lr"] = b * f
                x, d = x.to(dev, non_blocking=True), d.to(dev, non_blocking=True)
                with torch.autocast("cuda", enabled=(dev.type == "cuda")): pred = model(x)
                pred = pred.float()
                loss = F.mse_loss(pred * S, d * S) + LAM * (pred.sum((1, 2, 3)) - d.sum((1, 2, 3))).abs().mean()
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 10.0); scaler.step(opt); scaler.update(); ema.update(model)
                tl += float(loss); tn += 1; it += 1
            rewarm = False; gep += 1; ema.ema.eval()
            pc = [float(density_full(ema.ema, im, dev).sum()) for im in va_data]; m = count_metrics(pc, va_gt)
            hist.append(dict(epoch=gep, phase=phase, train_loss=tl / max(tn, 1), val_MAE=m["MAE"], val_RMSE=m["RMSE"], val_R2=m["R2"], lr=opt.param_groups[0]["lr"]))
            pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
            ck = dict(state=ema.ema.state_dict(), kw=kw, epoch=gep, val_MAE=m["MAE"])
            if m["MAE"] < best: best = m["MAE"]; _save_atomic(ck, out / "best.pth")
            _save_atomic(ck, lp)
            _save_atomic(dict(model=model.state_dict(), ema=ema.ema.state_dict(), ema_n=ema.n, opt=opt.state_dict(), scaler=scaler.state_dict(),
                              phase=phase, gep=gep, best=best, hist=hist), rp)
            print(f"[{key}] {phase} ep{gep:02d} loss={tl / max(tn, 1):.4f} val MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f} best={best:.2f} ({time.time() - t0:.0f}s)", flush=True)
        offset += epochs
    Path(c["done"]).write_text("ok")

if __name__ == "__main__":
    train_one(sys.argv[1])
