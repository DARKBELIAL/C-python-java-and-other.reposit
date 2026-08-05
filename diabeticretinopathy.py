# ============================================================================
# HYBRID 3-LAYER DIABETIC RETINOPATHY MODEL + LIVE GRADIO INTERFACE
# Paste this whole cell into Google Colab and run it. It will:
#   1) install what's needed
#   2) scan /content for your fundus images and report how many were found
#   3) train Layer 1 (tabular ensemble) on /content/diabetes.csv
#   4) build Layer 2 (CNN seed field -> Turing reaction-diffusion progression)
#   5) build Layer 3 (SHAP + reaction-diffusion visual explanation)
#   6) LAUNCH a Gradio form that takes real patient data + a real image
#      (uploaded, or picked from the images already in /content) and returns
#      a live prediction + explanation plots
# ============================================================================

!pip -q install xgboost lightgbm shap opencv-python-headless torch torchvision gradio --upgrade

import os, glob, warnings, tempfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ----------------------------------------------------------------------------
# CONFIG -- matches your actual Colab /content layout:
#   /content/diabetes.csv
#   /content/IDRiD_001.jpg ... IDRiD_413.jpg  (images sit directly in /content)
# Edit these if your paths differ.
# ----------------------------------------------------------------------------
CONFIG = {
    "csv_path": "/content/diabetes_dataset00.csv",
    "target_col": "Target",           # your CSV's real label column (multi-class disease type)
    "id_col": None,
    "test_size": 0.2,
    "random_state": 42,
    "stage1_prob_threshold": 0.5,

    "image_dir": "/content",           # images live directly in /content, not /content/images
    "image_glob_pattern": "IDRiD_*.jpg",

    "img_size": 256,
    "rd_steps": 150,
    "rd_Du": 0.16,
    "rd_Dv": 0.08,
    "rd_feed": 0.035,
    "rd_kill": 0.060,
    "progression_prob_threshold": 0.5,
}

# ----------------------------------------------------------------------------
# Locate + count the fundus images that are actually sitting in Colab
# ----------------------------------------------------------------------------
def scan_available_images(cfg):
    pattern = os.path.join(cfg["image_dir"], cfg["image_glob_pattern"])
    paths = sorted(glob.glob(pattern))
    if paths:
        print(f"Found {len(paths)} fundus images in '{cfg['image_dir']}' "
              f"matching '{cfg['image_glob_pattern']}' "
              f"(first: {os.path.basename(paths[0])}, last: {os.path.basename(paths[-1])}).")
    else:
        print(f"WARNING: no images found in '{cfg['image_dir']}' matching "
              f"'{cfg['image_glob_pattern']}'. Check CONFIG['image_dir'] / "
              f"CONFIG['image_glob_pattern'] against your Colab Files panel.")
    return paths


AVAILABLE_IMAGES = scan_available_images(CONFIG)

# Human-readable stage labels, matching the actual DR pathophysiology:
#   Stage 1: "Diabetes without retinopathy" -- fundus looks NORMAL to the eye, but
#            high blood sugar is already causing microscopic pericyte damage. This
#            is exactly why Layer 1 (tabular clinical risk model) exists: there is
#            nothing visible in the image yet for Layer 2 to work with.
#   Stage 2: Non-proliferative (pre-proliferative) retinopathy -- pericyte loss ->
#            weakened capillary walls -> microaneurysms (red dots) + leaky vessels
#            -> hard exudates (yellow-white flecks). Now VISIBLE in the image, so a
#            doctor (or Layer 2 directly) can skip Layer 1 entirely.
#   Stage 3: Proliferative retinopathy (PDR) -- prolonged ischemia -> VEGF-driven
#            growth of fragile new vessels -> risk of vitreous hemorrhage / retinal
#            detachment. This is the clinical end-stage; "progressing past Stage 3"
#            means worsening PDR complications, not a numbered Stage 4.
STAGE_NAMES = {
    1: "Stage 1 - Diabetes without retinopathy (normal-looking fundus; microscopic pericyte damage only)",
    2: "Stage 2 - Non-proliferative retinopathy (microaneurysms, hard exudates)",
    3: "Stage 3 - Proliferative retinopathy (VEGF-driven neovascularization)",
}

def stage_label(n):
    """Stage label for any n, including beyond the terminal Stage 3."""
    if n in STAGE_NAMES:
        return STAGE_NAMES[n]
    return ("Beyond Stage 3 - worsening proliferative retinopathy "
            "(risk of vitreous hemorrhage / retinal detachment)")

# EDITABLE placeholder: added to the reaction-diffusion image score (in logit space)
# before the final sigmoid, so later stages start from a slightly higher baseline
# progression risk than earlier ones (Stage 2 -> 3, i.e. NPDR -> PDR, is a clinically
# well-known higher-risk transition than Stage 1 -> 2). Replace with real
# transition-rate statistics once you have longitudinal follow-up data.
STAGE_RISK_PRIOR = {1: -0.3, 2: 0.0, 3: 0.4}

# ----------------------------------------------------------------------------
# Load real CSV if present, else build a small synthetic demo dataset so the
# whole pipeline (and the UI form) always has something to train/run on.
# ----------------------------------------------------------------------------
def load_or_make_tabular_data(cfg):
    if os.path.exists(cfg["csv_path"]):
        df = pd.read_csv(cfg["csv_path"])
        if cfg["target_col"] not in df.columns:
            print(f"WARNING: '{cfg['target_col']}' not found in CSV columns {list(df.columns)}. "
                  f"Using last column '{df.columns[-1]}' as target instead.")
            cfg["target_col"] = df.columns[-1]
        print(f"Loaded real dataset: {cfg['csv_path']}  shape={df.shape}")
        print(f"Target column '{cfg['target_col']}' class counts:")
        print(df[cfg["target_col"]].value_counts())
        return df

    print("No CSV found at CONFIG['csv_path'] -> generating a synthetic demo dataset "
          "so the app runs end-to-end. Fix CONFIG['csv_path'] to point at your real diabetes.csv.")
    rng = np.random.default_rng(cfg["random_state"])
    n = 800
    age = rng.integers(30, 80, n)
    duration_diabetes_years = rng.integers(0, 25, n)
    hba1c = rng.normal(7.5, 1.3, n).clip(4, 14)
    systolic_bp = rng.normal(135, 15, n).clip(90, 200)
    diastolic_bp = rng.normal(85, 10, n).clip(60, 120)
    bmi = rng.normal(27, 4, n).clip(15, 45)
    ldl = rng.normal(120, 30, n).clip(50, 250)

    risk = (
        0.03 * (hba1c - 7) + 0.02 * (duration_diabetes_years) +
        0.01 * (systolic_bp - 130) + 0.02 * (bmi - 25) + rng.normal(0, 1, n)
    )
    label = (risk > np.median(risk)).astype(int)

    df = pd.DataFrame({
        "age": age, "duration_diabetes_years": duration_diabetes_years,
        "hba1c": hba1c, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "bmi": bmi, "ldl": ldl, cfg["target_col"]: label,
    })
    return df


# ----------------------------------------------------------------------------
# LAYER 1 -- Tabular ensemble (Stage-1 DR / diabetes-risk screening)
#
# Your real dataset's "Target" column is MULTI-CLASS (Type 1 Diabetes, Type 2
# Diabetes, Prediabetic, LADA, Wolfram Syndrome, Steroid-Induced Diabetes,
# Neonatal Diabetes Mellitus, ...), not a ready-made binary Stage-1 flag, and
# most of its other columns (Genetic Markers, Family History, Smoking Status,
# etc.) are categorical text, not numbers. This class now handles both:
#   - one-hot encodes every text/categorical column instead of silently
#     dropping it (the previous version only kept numeric columns, which
#     would have thrown away almost all of your real features)
#   - auto-binarizes the multi-class target: any class whose name contains
#     "healthy" / "normal" / "no diabetes" / "non-diabetic" / "negative" is
#     treated as the negative (0) class; everything else becomes positive
#     (1) = "flag for Stage-1 screening". If no such class exists, the most
#     frequent class is used as the negative/baseline class instead, and a
#     NOTE is printed telling you exactly what was chosen so you can correct
#     it in _encode_target() if it's not what you want.
# ----------------------------------------------------------------------------
class TabularStage1Model:
    def __init__(self, config):
        self.cfg = config
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.negative_classes = None
        self.model = None

    def _encode_target(self, y_raw):
        if pd.api.types.is_numeric_dtype(y_raw):
            uniques = set(pd.unique(y_raw.dropna()))
            if uniques <= {0, 1}:
                self.negative_classes = [0]
                return y_raw.fillna(0).astype(int).values

        y_str = y_raw.astype(str)
        classes = y_str.unique().tolist()
        negative_keywords = ["healthy", "normal", "no diabetes", "non-diabetic",
                              "nondiabetic", "negative", "none", "no ", "control"]
        neg_classes = [c for c in classes if any(k in c.lower() for k in negative_keywords)]

        if neg_classes:
            self.negative_classes = neg_classes
        else:
            most_common = y_str.value_counts().idxmax()
            self.negative_classes = [most_common]
            other_classes = [c for c in classes if c != most_common]
            print(f"NOTE: no obvious 'healthy / no-diabetes' class found in "
                  f"'{self.cfg['target_col']}'. Treating the most frequent class "
                  f"'{most_common}' as the negative/baseline class, and all other "
                  f"classes {other_classes} as Stage-1-positive. Edit "
                  f"TabularStage1Model._encode_target() if you want a different split.")

        y_bin = (~y_str.isin(self.negative_classes)).astype(int)
        return y_bin.values

    def _prep_xy(self, df):
        cfg = self.cfg
        drop_cols = [c for c in [cfg["id_col"], cfg["target_col"]] if c]
        X_raw = df.drop(columns=drop_cols, errors="ignore").copy()

        self.numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        X = pd.get_dummies(X_raw, columns=self.categorical_cols, drop_first=False)

        y = None
        if cfg["target_col"] in df.columns:
            y = self._encode_target(df[cfg["target_col"]])
        return X, y

    def fit(self, df):
        X, y = self._prep_xy(df)
        self.feature_names = X.columns.tolist()
        print(f"Layer 1: {len(self.numeric_cols)} numeric + {len(self.categorical_cols)} "
              f"one-hot-encoded categorical columns -> {len(self.feature_names)} total features.")
        print(f"Layer 1 negative/baseline class(es): {self.negative_classes}  "
              f"(positive rate: {y.mean():.1%})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg["test_size"], random_state=self.cfg["random_state"], stratify=y
        )
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                     random_state=self.cfg["random_state"])
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=self.cfg["random_state"])
        lgbm = LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
                               class_weight="balanced", random_state=self.cfg["random_state"],
                               verbose=-1)

        self.model = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
            voting="soft", weights=[1, 1.2, 1.2],
        )
        self.model.fit(X_train, y_train)

        proba = self.model.predict_proba(X_test)[:, 1]
        preds = (proba >= self.cfg["stage1_prob_threshold"]).astype(int)
        print("Layer 1 AUC:", round(roc_auc_score(y_test, proba), 4))
        print(classification_report(y_test, preds))
        return self

    def predict_proba_row(self, row_dict):
        X_raw = pd.DataFrame([row_dict])
        cat_cols_present = [c for c in self.categorical_cols if c in X_raw.columns]
        X = pd.get_dummies(X_raw, columns=cat_cols_present, drop_first=False)

        dummy_cols = set(self.feature_names) - set(self.numeric_cols)
        for col in self.feature_names:
            if col not in X.columns:
                # a one-hot column that just never appeared for this single row -> 0
                # a genuinely missing numeric column -> NaN, let the imputer fill it
                X[col] = 0 if col in dummy_cols else np.nan
        X = X[self.feature_names]

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return float(self.model.predict_proba(X)[0, 1])


# ----------------------------------------------------------------------------
# LAYER 2a -- CNN seed-field extractor
# ----------------------------------------------------------------------------
class LesionActivationExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print("Could not download pretrained weights (no internet?) -- using random init:", e)
            backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.channel_reduce = nn.Conv2d(512, 1, kernel_size=1)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)
        act = torch.sigmoid(self.channel_reduce(feat))
        return act, feat


IMG_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_fundus_image_array(img_rgb, img_size=256):
    img = cv2.resize(img_rgb, (img_size, img_size))
    return img


def load_fundus_image_from_disk(path, img_size=256):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return load_fundus_image_array(img_rgb, img_size)


def get_seed_field(extractor, img_rgb, out_size=128):
    tensor = IMG_TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)
    extractor.to(DEVICE)
    act_map, feat = extractor(tensor)
    act_map = act_map.squeeze().detach().cpu().numpy()
    act_map = cv2.resize(act_map, (out_size, out_size))
    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
    return act_map


# ----------------------------------------------------------------------------
# LAYER 2b -- Turing / Gray-Scott reaction-diffusion simulator
# ----------------------------------------------------------------------------
def laplacian(Z):
    return (
        -Z
        + 0.20 * (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
                  + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1))
        + 0.05 * (np.roll(np.roll(Z, 1, axis=0), 1, axis=1)
                  + np.roll(np.roll(Z, 1, axis=0), -1, axis=1)
                  + np.roll(np.roll(Z, -1, axis=0), 1, axis=1)
                  + np.roll(np.roll(Z, -1, axis=0), -1, axis=1))
    )


class ReactionDiffusionSimulator:
    def __init__(self, cfg):
        self.Du, self.Dv = cfg["rd_Du"], cfg["rd_Dv"]
        self.F, self.k = cfg["rd_feed"], cfg["rd_kill"]
        self.steps = cfg["rd_steps"]

    def run(self, seed_map, dt=1.0):
        # Gray-Scott only forms lasting, image-specific patterns when v is seeded as
        # small LOCALIZED hotspots (its classic regime) -- a full-field seed collapses
        # to the same homogeneous steady state for every image within ~20 steps.
        u = np.ones_like(seed_map)
        v = np.zeros_like(seed_map)

        # Fixed absolute threshold (NOT a percentile) so images with more/stronger CNN
        # activation genuinely get a larger seeded area than calmer images.
        hotspot_threshold = 0.65
        hotspots = seed_map >= hotspot_threshold
        v[hotspots] = 0.25 + 0.25 * seed_map[hotspots]
        u[hotspots] = 0.50

        rng = np.random.default_rng(0)
        v = np.clip(v + 0.01 * rng.standard_normal(v.shape) * hotspots, 0, 1)
        u = np.clip(u, 0, 1)

        history = []
        for step in range(self.steps):
            Lu, Lv = laplacian(u), laplacian(v)
            uvv = u * v * v
            u += dt * (self.Du * Lu - uvv + self.F * (1 - u))
            v += dt * (self.Dv * Lv + uvv - (self.F + self.k) * v)
            u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
            history.append({
                "step": step, "mean_u": float(u.mean()), "mean_v": float(v.mean()),
                "var_v": float(v.var()), "max_v": float(v.max()),
                "frac_high_v": float((v > 0.15).mean()),
            })
        return pd.DataFrame(history), u, v


# ----------------------------------------------------------------------------
# LAYER 2c -- time-series progression head (train on real longitudinal labels
# when you have them; see train_progression_lstm() below)
# ----------------------------------------------------------------------------
class ProgressionLSTM(nn.Module):
    def __init__(self, n_features=5, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return torch.sigmoid(self.head(h[-1])).squeeze(-1)


def ts_to_tensor(ts_df, cols=("mean_u", "mean_v", "var_v", "max_v", "frac_high_v")):
    arr = ts_df[list(cols)].values.astype(np.float32)
    return torch.tensor(arr).unsqueeze(0)


def train_progression_lstm(model, sequences, labels, epochs=30, lr=1e-3):
    """sequences: list of (T, F) numpy arrays. labels: list of 0/1 (progressed to
    Stage-2 or not). Plug in your real longitudinal cohort here, then set
    hybrid.lstm_is_trained = True to switch Layer 2 over to using this model."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    X = [torch.tensor(s, dtype=torch.float32) for s in sequences]
    y = torch.tensor(labels, dtype=torch.float32)
    for epoch in range(epochs):
        opt.zero_grad()
        preds = torch.stack([model(x.unsqueeze(0).to(DEVICE)) for x in X]).squeeze(-1)
        loss = loss_fn(preds, y.to(DEVICE))
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
    return model


# ----------------------------------------------------------------------------
# LAYER 3 -- XAI (SHAP for Layer 1, visual explanation for Layer 2)
# ----------------------------------------------------------------------------
class XAILayer:
    def __init__(self, layer1_model, cnn_extractor):
        self.layer1_model = layer1_model
        self.cnn_extractor = cnn_extractor
        self._shap_explainer = None

    def explain_layer1(self, row_dict):
        if self.layer1_model is None:
            return None

        try:
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self.layer1_model.model.estimators_[0])

            X_raw = pd.DataFrame([row_dict])
            cat_cols_present = [c for c in self.layer1_model.categorical_cols if c in X_raw.columns]
            X = pd.get_dummies(X_raw, columns=cat_cols_present, drop_first=False)
            dummy_cols = set(self.layer1_model.feature_names) - set(self.layer1_model.numeric_cols)
            for col in self.layer1_model.feature_names:
                if col not in X.columns:
                    X[col] = 0 if col in dummy_cols else np.nan
            X = X[self.layer1_model.feature_names]

            X_t = self.layer1_model.imputer.transform(X)
            X_t = self.layer1_model.scaler.transform(X_t)
            shap_values = self._shap_explainer.shap_values(X_t)

            # Different shap/sklearn version combos return this differently for binary
            # classification: a list of two (n_samples, n_features) arrays [class0, class1],
            # OR a single (n_samples, n_features, n_classes) array, OR just (n_samples,
            # n_features) already. Normalize all of these to the positive-class 2D array
            # so summary_plot always gets the shape it expects (this mismatch was the
            # silent failure that made the SHAP plot not appear in some cases).
            if isinstance(shap_values, list):
                shap_vals_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_vals_pos = shap_values[:, :, 1]
            else:
                shap_vals_pos = shap_values

            plt.figure()
            # bar plot is the right choice for a single-instance explanation (a "dot"
            # summary plot is degenerate with only one sample)
            shap.summary_plot(shap_vals_pos, X_t, feature_names=self.layer1_model.feature_names,
                               plot_type="bar", show=False)
            fig = plt.gcf()
            plt.close(fig)
            return fig
        except Exception as e:
            print(f"WARNING: SHAP explanation failed ({e}); showing placeholder instead.")
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"SHAP explanation unavailable for this input:\n{e}",
                    ha="center", va="center", wrap=True)
            ax.axis("off")
            plt.close(fig)
            return fig

    def explain_layer2(self, img_rgb, seed_map, ts_df, current_stage=None, next_stage=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        title = "Input fundus image"
        if current_stage is not None:
            title += f"\n(Stage {current_stage} -> Stage {next_stage})"
        axes[0].imshow(img_rgb); axes[0].set_title(title); axes[0].axis("off")
        axes[1].imshow(seed_map, cmap="inferno")
        axes[1].set_title("CNN activation seed field\n(reaction-diffusion initial condition)")
        axes[1].axis("off")
        axes[2].plot(ts_df["step"], ts_df["mean_v"], label="mean inhibitor (v)")
        axes[2].plot(ts_df["step"], ts_df["frac_high_v"], label="fraction high-v area")
        axes[2].set_ylim(0, 1)  # FIX: fixed shared scale, was auto-scaling per plot and
                                 # making every case's curve look like the same "full" bump
                                 # regardless of its true (often much smaller) magnitude
        axes[2].set_xlabel("simulated time step")
        axes[2].set_title("Reaction-diffusion time series")
        axes[2].legend()
        plt.tight_layout()
        plt.close(fig)
        return fig


# ----------------------------------------------------------------------------
# ORCHESTRATION -- HybridDRModel (Layer1 -> Layer2 -> Layer3, with the
# "skip Layer 1 if image is already Stage-2" rule)
# ----------------------------------------------------------------------------
class HybridDRModel:
    def __init__(self, config):
        self.cfg = config
        self.layer1 = None
        self.cnn_extractor = LesionActivationExtractor()
        self.rd_sim = ReactionDiffusionSimulator(config)
        self.lstm = ProgressionLSTM()
        self.lstm_is_trained = False  # flip to True after calling train_progression_lstm() on real data
        self.xai = XAILayer(None, self.cnn_extractor)

    def set_layer1(self, layer1_model):
        self.layer1 = layer1_model
        self.xai.layer1_model = layer1_model
        return self

    def _run_layer2(self, img_rgb, current_stage):
        img_rgb = load_fundus_image_array(img_rgb, self.cfg["img_size"])
        seed_map = get_seed_field(self.cnn_extractor, img_rgb)
        ts_df, u_final, v_final = self.rd_sim.run(seed_map)

        if self.lstm_is_trained:
            seq_tensor = ts_to_tensor(ts_df).to(DEVICE)
            self.lstm.to(DEVICE).eval()
            with torch.no_grad():
                progression_prob = float(self.lstm(seq_tensor).cpu().item())
        else:
            # ProgressionLSTM starts with random, UNTRAINED weights, so its raw output
            # barely reacts to the input image until trained on real longitudinal
            # progression labels. Until then, use this deterministic heuristic derived
            # from the reaction-diffusion simulation instead.
            progression_prob = self._heuristic_progression_prob(seed_map, ts_df, current_stage)

        return {
            "image": img_rgb, "seed_map": seed_map, "timeseries": ts_df,
            "current_stage": current_stage, "next_stage": current_stage + 1,
            "progression_probability": progression_prob,
            "progressing_to_next_stage": progression_prob >= self.cfg["progression_prob_threshold"],
        }

    @staticmethod
    def _heuristic_progression_prob(seed_map, ts_df, current_stage):
        # IMPORTANT FIX: the previous version scored this from `frac_high_v` AFTER
        # running the Gray-Scott simulation. A controlled test showed that statistic
        # explodes to ~1.0 for almost ANY nonzero seed within a handful of steps --
        # Gray-Scott is extremely explosive once seeded at all -- which is exactly why
        # every image was coming back "True" regardless of content.
        #
        # Instead, score primarily from the seed map itself (bounded, monotonic,
        # directly reflects how much of the retina the CNN flagged as activated),
        # and use the reaction-diffusion trajectory only as a smaller secondary term
        # so Layer 2 still genuinely uses its simulation output, just not as the sole
        # saturating signal.
        hotspot_frac = float((seed_map >= 0.65).mean())
        peak_var = float(ts_df["var_v"].max())

        score = 6.0 * (hotspot_frac - 0.15) + 5.0 * (peak_var - 0.06)
        score += STAGE_RISK_PRIOR.get(current_stage, STAGE_RISK_PRIOR[max(STAGE_RISK_PRIOR)])
        score = float(np.clip(score, -6.0, 6.0))  # safety net against outliers
        return float(1.0 / (1.0 + np.exp(-score)))

    def predict(self, patient_row=None, img_rgb=None, current_stage=None):
        """current_stage:
             None        -> run Layer 1 on patient_row to decide Stage-1 status (the
                             fundus looks normal, so a tabular clinical model is needed);
                             if positive, Layer 2 estimates P(Stage 1 -> Stage 2).
             1, 2, 3      -> skip Layer 1 entirely (the image already visibly shows this
                             stage), feed the image straight into Layer 2, which estimates
                             P(current_stage -> next stage). E.g. current_stage=2 ->
                             probability of progressing from Stage 2 (NPDR) to Stage 3 (PDR).
        """
        result = {"layer1_ran": False, "layer2_ran": False, "layer1_fig": None, "layer2_fig": None}

        if current_stage is not None:
            assert img_rgb is not None, "An image is required when a current stage is specified."
            assert current_stage >= 1, "current_stage must be >= 1 (Stage 0 = no DR has nothing to progress from yet)."
            layer2_out = self._run_layer2(img_rgb, current_stage)
            fig2 = self.xai.explain_layer2(layer2_out["image"], layer2_out["seed_map"], layer2_out["timeseries"],
                                            current_stage=layer2_out["current_stage"], next_stage=layer2_out["next_stage"])
            fig1_placeholder, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No SHAP plot here: Layer 1 (tabular model) was\nintentionally "
                              "skipped because a current stage was given directly.",
                    ha="center", va="center", wrap=True)
            ax.axis("off")
            plt.close(fig1_placeholder)
            result.update(layer2_ran=True, layer2=layer2_out, layer2_fig=fig2, layer1_fig=fig1_placeholder,
                          note=f"{stage_label(current_stage)} image supplied directly; Layer 1 bypassed.")
            return result

        assert patient_row is not None and self.layer1 is not None, \
            "Patient data and a trained Layer 1 model are required when current_stage is not given."
        stage1_prob = self.layer1.predict_proba_row(patient_row)
        stage1_positive = stage1_prob >= self.cfg["stage1_prob_threshold"]
        result.update(layer1_ran=True, stage1_probability=stage1_prob, stage1_positive=stage1_positive)
        result["layer1_fig"] = self.xai.explain_layer1(patient_row)

        if not stage1_positive:
            result["conclusion"] = "No DR / not Stage-1 -> pipeline stops, no image analysis needed."
            return result

        assert img_rgb is not None, "Layer 1 flagged Stage-1 -> please also provide a fundus image."
        layer2_out = self._run_layer2(img_rgb, current_stage=1)
        fig2 = self.xai.explain_layer2(layer2_out["image"], layer2_out["seed_map"], layer2_out["timeseries"],
                                            current_stage=layer2_out["current_stage"], next_stage=layer2_out["next_stage"])
        result.update(layer2_ran=True, layer2=layer2_out, layer2_fig=fig2,
                      conclusion="Stage 1 approved by Layer 1 -> Layer 2 estimated progression risk to Stage 2.")
        return result


# ----------------------------------------------------------------------------
# TRAIN LAYER 1 NOW
# ----------------------------------------------------------------------------
_df = load_or_make_tabular_data(CONFIG)
layer1 = TabularStage1Model(CONFIG).fit(_df)

hybrid = HybridDRModel(CONFIG)
hybrid.set_layer1(layer1)

# ----------------------------------------------------------------------------
# BUILD + LAUNCH THE GRADIO UI (this is the part that actually takes user input)
# ----------------------------------------------------------------------------
FEATURE_NAMES = layer1.feature_names
# Sensible defaults for the form: numeric columns default to their dataset mean;
# one-hot categorical columns default to 0 (they're not meant to be typed in directly
# -- pick the category with the Dropdowns generated below instead).
_numeric_defaults = {c: round(float(_df[c].mean()), 2) for c in layer1.numeric_cols if c in _df.columns}


def _format_report(result):
    lines = []
    if result.get("layer1_ran"):
        lines.append(
            f"### Layer 1 - Tabular Screening\n"
            f"- Stage-1 probability: **{result['stage1_probability']:.3f}**\n"
            f"- Stage-1 positive: **{result['stage1_positive']}**"
        )
    if result.get("layer2_ran"):
        l2 = result["layer2"]
        cur, nxt = l2["current_stage"], l2["next_stage"]
        lines.append(
            f"### Layer 2 - Reaction-Diffusion Progression\n"
            f"- From: {stage_label(cur)}\n"
            f"- To: {stage_label(nxt)}\n"
            f"- Progression probability: **{l2['progression_probability']:.3f}**\n"
            f"- Predicted to progress: **{l2['progressing_to_next_stage']}**"
        )
    conclusion = result.get("conclusion") or result.get("note")
    if conclusion:
        lines.append(f"### Conclusion\n{conclusion}")
    return "\n\n".join(lines) if lines else "No result."


def _resolve_image(uploaded_image, dataset_image_choice):
    if uploaded_image is not None:
        return uploaded_image
    if dataset_image_choice:
        return load_fundus_image_from_disk(
            os.path.join(CONFIG["image_dir"], dataset_image_choice), CONFIG["img_size"]
        )
    return None


STAGE_CHOICE_AUTO = "Auto (use Layer 1 tabular screening for Stage 1)"
STAGE_CHOICES = [STAGE_CHOICE_AUTO] + [f"{n} -> currently at Stage {s}" for s, n in STAGE_NAMES.items()]


def predict_fn(uploaded_image, dataset_image_choice, stage_choice, *feature_values):
    img = _resolve_image(uploaded_image, dataset_image_choice)

    if stage_choice == STAGE_CHOICE_AUTO:
        current_stage = None
        patient_row = dict(zip(FEATURE_NAMES_FOR_FORM, feature_values))
    else:
        current_stage = int(stage_choice.split("Stage ")[-1])
        patient_row = None

    try:
        result = hybrid.predict(patient_row=patient_row, img_rgb=img, current_stage=current_stage)
    except (AssertionError, FileNotFoundError) as e:
        return f"**Input error:** {e}", None, None
    return _format_report(result), result.get("layer1_fig"), result.get("layer2_fig")


# ---- build one input widget per ORIGINAL column (not per one-hot column) ----
# Numeric columns -> gr.Number. Categorical columns -> gr.Dropdown with the
# actual categories seen in your CSV, so the form matches your real data.
FEATURE_NAMES_FOR_FORM = layer1.numeric_cols + layer1.categorical_cols
form_widgets = []
for col in layer1.numeric_cols:
    form_widgets.append(gr.Number(label=col, value=_numeric_defaults.get(col, 0)))
for col in layer1.categorical_cols:
    choices = sorted(_df[col].dropna().astype(str).unique().tolist())
    form_widgets.append(gr.Dropdown(choices=choices, value=choices[0] if choices else None, label=col))

inputs = [
    gr.Image(type="numpy", label="Upload a fundus image"),
    gr.Dropdown(choices=[os.path.basename(p) for p in AVAILABLE_IMAGES],
                label=f"...or pick one of the {len(AVAILABLE_IMAGES)} images already in {CONFIG['image_dir']}"),
    gr.Dropdown(choices=STAGE_CHOICES, value=STAGE_CHOICE_AUTO,
                label="What stage is this image already at? (Auto uses Layer 1 + patient data for Stage 1; "
                      "picking a stage skips Layer 1 and asks 'what's the probability of reaching the next stage?')"),
] + form_widgets

demo = gr.Interface(
    fn=predict_fn,
    inputs=inputs,
    outputs=[
        gr.Markdown(label="Diagnosis report"),
        gr.Plot(label="Layer 1 - SHAP explanation"),
        gr.Plot(label="Layer 2 - reaction-diffusion explanation"),
    ],
    title="Hybrid 3-Layer Diabetic Retinopathy Diagnosis",
    description=(
        "Layer 1 (tabular ensemble) -> Layer 2 (Turing reaction-diffusion progression) -> Layer 3 (XAI). "
        "Provide EITHER an uploaded image OR pick one from the dataset dropdown. Leave the stage dropdown "
        "on 'Auto' to screen for Stage 1 from patient data first; pick 'currently at Stage 2/3/4' to skip "
        "Layer 1 and directly ask 'what's the probability this progresses to the NEXT stage?' -- e.g. "
        "picking Stage 3 answers 'probability of reaching Stage 4.'"
    ),
)

demo.launch(share=True, debug=True)
