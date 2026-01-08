import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional (for 3rd model if you really have it)
try:
    import joblib
except Exception:
    joblib = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None


def sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def soft_vote(probs, weights=None) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    if weights is None:
        weights = np.ones_like(probs, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
    s = float(weights.sum())
    weights = weights / (s + 1e-9)
    return float((probs * weights).sum())


def is_prob_like(y: float) -> bool:
    return 0.0 <= y <= 1.0


def model_predict_prob_keras(model, x: np.ndarray) -> float:
    """
    model(x) -> scalar prob.
    Accepts prob output or logit output.
    """
    y = model.predict(x, verbose=0)
    y = float(np.asarray(y).reshape(-1)[0])
    return y if is_prob_like(y) else sigmoid(y)


def resample_seq(arr: np.ndarray, L: int) -> np.ndarray:
    """
    arr: (T, ...) -> resampled to (L, ...) by picking indices.
    """
    T = arr.shape[0]
    if T == L:
        return arr
    if T <= 1:
        return np.repeat(arr, L, axis=0)
    idx = np.linspace(0, T - 1, L).round().astype(int)
    return arr[idx]


# -----------------------------
# Pose → heatmap
# -----------------------------
def draw_pose_heatmap(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray,
                      out_h: int, out_w: int,
                      conf_thr: float = 0.3,
                      radius: int = 3) -> np.ndarray:
    """
    keypoints_xy: (K,2) in pixel coords of original frame
    keypoints_conf: (K,)
    Return: heatmap (out_h, out_w, 1) float32 in [0,1]
    """
    hm = np.zeros((out_h, out_w), dtype=np.uint8)

    # Normalize from original coords to heatmap coords handled outside (we will scale by bbox)
    # Here keypoints_xy are already in heatmap coords
    for (x, y), c in zip(keypoints_xy, keypoints_conf):
        if c < conf_thr:
            continue
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < out_w and 0 <= yi < out_h:
            cv2.circle(hm, (xi, yi), radius, 255, -1)

    hm = hm.astype(np.float32) / 255.0
    return hm[..., None]


def pick_person(result, target_tid=None):
    """
    Pick which person to use.
    - if tracking enabled and target_tid exists, pick that
    - else pick largest bbox
    Return: person index, chosen_tid (or -1)
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, -1

    tids = None
    if getattr(boxes, "id", None) is not None:
        try:
            tids = boxes.id.cpu().numpy().astype(int)
        except Exception:
            tids = None

    if target_tid is not None and tids is not None:
        idxs = np.where(tids == target_tid)[0]
        if len(idxs):
            i = int(idxs[0])
            return i, int(tids[i])

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(areas))
    tid = int(tids[i]) if tids is not None else -1
    return i, tid


def get_keypoints(result, person_i):
    """
    Return:
      - xy: (K,2) pixel coords in original frame
      - conf: (K,)
    """
    if result.keypoints is None:
        return None, None
    kps = result.keypoints
    xy = kps.xy
    if hasattr(xy, "cpu"):
        xy = xy.cpu().numpy()
    else:
        xy = np.asarray(xy)

    conf = getattr(kps, "conf", None)
    if conf is not None:
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()
        else:
            conf = np.asarray(conf)
    else:
        conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    return xy[person_i], conf[person_i]


def bbox_xyxy(result, person_i):
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()[person_i]
    return xyxy  # x1,y1,x2,y2


# -----------------------------
# Optional 3rd model loader
# -----------------------------
def load_third_model(path: str):
    """
    Best-effort:
      - joblib for sklearn
      - LightGBM Booster model file
    """
    if path is None:
        return None, None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Try joblib (sklearn RF, etc.)
    if joblib is not None:
        try:
            return joblib.load(path), "joblib"
        except Exception:
            pass

    # Try LightGBM Booster native model
    if lgb is not None:
        try:
            return lgb.Booster(model_file=path), "lgb_booster"
        except Exception:
            pass

    raise RuntimeError("Third model format not supported. Provide joblib/pkl or LightGBM native model file.")


def predict_third_model_prob(model, kind: str, X_2d: np.ndarray) -> float:
    if model is None:
        return float("nan")
    if kind == "joblib":
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_2d)
            return float(np.asarray(p)[0, 1])
        y = float(np.asarray(model.predict(X_2d)).reshape(-1)[0])
        return y if is_prob_like(y) else sigmoid(y)
    if kind == "lgb_booster":
        p = model.predict(X_2d)
        return float(np.asarray(p).reshape(-1)[0])
    y = float(np.asarray(model.predict(X_2d)).reshape(-1)[0])
    return y if is_prob_like(y) else sigmoid(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input video")
    ap.add_argument("--outdir", default=str(Path.home() / "Downloads" / "fall_out"), help="output dir")
    ap.add_argument("--pose-weights", default="yolo11n-pose.pt", help="YOLO pose weights")
    ap.add_argument("--device", default=None, help="0 or cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--track", action="store_true", help="enable tracking")

    ap.add_argument("--model1", required=True, help="Keras model (.keras) #1")
    ap.add_argument("--model2", required=True, help="Keras model (.keras) #2")

    ap.add_argument("--model3", default=None, help="optional 3rd model file (joblib/pkl or LightGBM native model)")
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--w3", type=float, default=1.0)

    ap.add_argument("--unit-sec", type=float, default=1.0, help="output probability per unit time (sec)")
    ap.add_argument("--kp-conf-thr", type=float, default=0.3)
    ap.add_argument("--hm-size", type=int, default=128, help="pose heatmap size (square)")
    ap.add_argument("--radius", type=int, default=3, help="heatmap keypoint radius")

    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load models
    m1 = load_model(args.model1, compile=False)
    m2 = load_model(args.model2, compile=False)

    third_model, third_kind = load_third_model(args.model3) if args.model3 else (None, None)

    # Inspect expected input shapes (we support seq-image models: (None, T, H, W, C))
    in1 = m1.input_shape
    in2 = m2.input_shape

    def parse_seq_image_shape(inp):
        # Expect 5D: (None, T, H, W, C)
        if inp is None:
            return None
        if isinstance(inp, list):
            inp = inp[0]
        if len(inp) == 5:
            _, T, H, W, C = inp
            return int(T), int(H), int(W), int(C)
        return None

    s1 = parse_seq_image_shape(in1)
    s2 = parse_seq_image_shape(in2)

    if s1 is None or s2 is None:
        raise RuntimeError(
            "This script currently assumes your .keras models take sequence images: (batch, T, H, W, C).\n"
            f"Got model1 input_shape={in1}, model2 input_shape={in2}.\n"
            "If your models take numeric features instead, tell me the input_shape and I will adapt the feeder."
        )

    T1, H1, W1, C1 = s1
    T2, H2, W2, C2 = s2

    # YOLO pose
    yolo = YOLO(args.pose_weights)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1 else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    unit_frames = max(1, int(round(args.unit_sec * fps)))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outdir / "annotated.mp4"), fourcc, fps, (W, H))
        if not writer.isOpened():
            writer = None
            print("WARN: cannot open writer, skip saving video.")

    # buffers for the current unit window
    hm_buf = []
    target_tid = None

    rows = []
    frame_idx = -1
    last_p_final = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        # inference
        if args.track:
            res = yolo.track(frame, persist=True, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        else:
            res = yolo.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]

        person_i, tid = pick_person(res, target_tid=target_tid)
        if target_tid is None and args.track and tid != -1:
            target_tid = tid

        out_frame = res.plot() if res is not None else frame

        if person_i is not None:
            xy, cf = get_keypoints(res, person_i)
            if xy is not None:
                x1,y1,x2,y2 = bbox_xyxy(res, person_i)
                x1 = max(0, min(W-1, int(x1)))
                y1 = max(0, min(H-1, int(y1)))
                x2 = max(0, min(W-1, int(x2)))
                y2 = max(0, min(H-1, int(y2)))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)

                # map keypoints into bbox-normalized heatmap coordinates
                hmN = args.hm_size
                xy_hm = xy.copy().astype(np.float32)
                xy_hm[:, 0] = (xy_hm[:, 0] - x1) / bw * (hmN - 1)
                xy_hm[:, 1] = (xy_hm[:, 1] - y1) / bh * (hmN - 1)

                hm = draw_pose_heatmap(xy_hm, cf, hmN, hmN, conf_thr=args.kp_conf_thr, radius=args.radius)
                hm_buf.append(hm)

        # once per unit
        if (frame_idx > 0) and (frame_idx % unit_frames == 0) and len(hm_buf) > 0:
            # build inputs for model1 and model2 by resampling to required T,H,W,C
            seq = np.stack(hm_buf, axis=0)  # (Traw, hmN, hmN, 1)

            def make_input(seq_raw, T, Hm, Wm, C):
                seq_rs = resample_seq(seq_raw, T)  # (T, hmN, hmN, 1)
                if (seq_rs.shape[1] != Hm) or (seq_rs.shape[2] != Wm):
                    # resize each frame
                    tmp = []
                    for f in seq_rs:
                        img = f[..., 0]
                        img = cv2.resize(img, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
                        tmp.append(img[..., None])
                    seq_rs = np.stack(tmp, axis=0)
                if C == 1:
                    pass
                else:
                    # replicate channel if model expects >1 channels
                    seq_rs = np.repeat(seq_rs, C, axis=-1)
                x = seq_rs[None, ...].astype(np.float32)  # (1,T,H,W,C)
                return x

            x1_in = make_input(seq, T1, H1, W1, C1)
            x2_in = make_input(seq, T2, H2, W2, C2)

            p1 = model_predict_prob_keras(m1, x1_in)
            p2 = model_predict_prob_keras(m2, x2_in)

            probs = [p1, p2]
            weights = [args.w1, args.w2]

            # optional third model: feed numeric features derived from the same window if you have it
            # Here we provide a simple numeric vector: [mean tilt proxy, mean motion proxy, ...]
            # If your RF/LGBM were trained on a specific feature schema, we must match it.
            p3 = float("nan")
            if third_model is not None:
                # simple baseline numeric features from heatmap energy (placeholders)
                energy = float(seq.mean())
                motion = float(np.abs(np.diff(seq, axis=0)).mean()) if seq.shape[0] > 1 else 0.0
                X3 = np.array([[energy, motion]], dtype=np.float32)  # <-- replace with your trained feature schema
                p3 = predict_third_model_prob(third_model, third_kind, X3)
                probs.append(p3)
                weights.append(args.w3)

            p_final = soft_vote(probs, weights)
            last_p_final = p_final

            rows.append({
                "sec": int(round(t_sec)),
                "t_sec": float(t_sec),
                "frame_idx": int(frame_idx),
                "p_model1": float(p1),
                "p_model2": float(p2),
                "p_model3": float(p3) if not math.isnan(p3) else None,
                "p_final": float(p_final),
            })

            # reset window buffer for next unit
            hm_buf = []

        # overlay
        if last_p_final is not None:
            cv2.putText(out_frame, f"Fall prob (per {args.unit_sec:.1f}s): {last_p_final:.3f}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(out_frame)

        if args.show:
            cv2.imshow("Fall Probability (q to quit)", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    out_csv = outdir / "prob_per_unit.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("DONE")
    print("CSV:", out_csv)
    if args.save_video:
        print("Video:", outdir / "annotated.mp4")


if __name__ == "__main__":
    main()
