import io
import json
import math
import os
import random
import re
import threading
import time
import tkinter as tk
import webbrowser
from io import BytesIO
from tkinter import filedialog, messagebox, ttk

import requests
from google.cloud import storage
from PIL import Image, ImageChops, ImageDraw, ImageTk

# ======================================================
# CONFIGURATION
# ======================================================

# Credenciales: solo se fija la variable si el fichero existe (evita romper la app
# en equipos donde no esté esa ruta). Si ya hay ADC o la variable definida, se respeta.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CRED = os.path.join(BASE_DIR, "gcp-service-account.json")  # NO se sube a GitHub (.gitignore)
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and os.path.exists(_DEFAULT_CRED):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _DEFAULT_CRED

# URL de TU servicio Cloud Run (ver docs/06_despliegue_google_cloud.md). También se puede
# fijar con la variable de entorno SCANIA_API_URL o desde Settings dentro de la app.
DEFAULT_CLOUD_RUN_URL = os.environ.get("SCANIA_API_URL", "https://TU-SERVICIO.run.app/predict")
BUCKET_NAME = os.environ.get("SCANIA_BUCKET_2D", "mri-bucket-heatmap")  # For images (Grad-CAM bucket)
BUCKET_NIFTI = os.environ.get("SCANIA_BUCKET_3D", "mri-bucket-3d")      # For NIfTI
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

GRADCAM_DIR = "gradcam_images"
os.makedirs(GRADCAM_DIR, exist_ok=True)

CREDENTIALS_FILE = "credentials.json"
CONFIG_FILE = "config.json"

DISCLAIMER_TEXT = (
    "ScanIA es una herramienta experimental de apoyo al análisis de imágenes. "
    "El resultado no constituye un diagnóstico médico y debe ser interpretado por "
    "un profesional sanitario cualificado."
)

# ======================================================
# APP CONFIG (persistente, no sensible)
# ======================================================

DEFAULT_CONFIG = {
    "api_url": DEFAULT_CLOUD_RUN_URL,
    "timeout_image": 600,
    "timeout_nifti": 900,
    "visual_threshold": 0.5,
    "auto_rotate_3d": True,
    "confirm_on_close": True,
    "model_type": "binary",
}

# Modelos de IA disponibles (valor guardado en config.json -> nombre visible)
AI_MODEL_OPTIONS = [
    ("binary", "Binary Tumor Detection"),
    ("cheng", "Cheng Brain Tumor Classification"),
]
AI_MODEL_LABELS = dict(AI_MODEL_OPTIONS)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


APP_CONFIG = load_config()


def cloud_run_url():
    return APP_CONFIG.get("api_url", DEFAULT_CLOUD_RUN_URL)


def health_url():
    url = cloud_run_url()
    if url.endswith("/predict"):
        return url[:-len("/predict")] + "/health"
    return url.rstrip("/") + "/health"


# ======================================================
# COLOR PALETTE — macOS dark / "Apple-like" theme
# ======================================================

COLOR_BG = "#1C1C1E"
COLOR_BG_ELEVATED = "#161618"
COLOR_CARD = "#2C2C2E"
COLOR_CARD_ALT = "#3A3A3C"
COLOR_SEPARATOR = "#38383A"
COLOR_TEXT = "#F5F5F7"
COLOR_TEXT_SECONDARY = "#98989D"
COLOR_TEXT_TERTIARY = "#6E6E73"

COLOR_ACCENT = "#0A84FF"
COLOR_ACCENT_HOVER = "#409CFF"
COLOR_ACCENT_SOFT = "#152C4A"

COLOR_SUCCESS = "#30D158"
COLOR_SUCCESS_SOFT = "#122E1B"
COLOR_DANGER = "#FF453A"
COLOR_DANGER_SOFT = "#3A1815"
COLOR_WARNING = "#FF9F0A"
COLOR_WARNING_SOFT = "#3A2A10"

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_DISPLAY = "Segoe UI Semibold"

# Estado de sesión
SESSION = {"user": None}

# ======================================================
# UTILITIES
# ======================================================

def clean_filename(filename):
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename


def upload_to_gcs(bucket_name, source_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_path)
    try:
        blob.make_public()
        return blob.public_url
    except Exception:
        return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"


def convert_public_url_to_gs(public_url):
    if not public_url.startswith("https://storage.googleapis.com/"):
        raise ValueError("Invalid public URL")
    path = public_url.replace("https://storage.googleapis.com/", "")
    return "gs://" + path


def predict_file(gs_url, model_type=None):
    headers = {"Content-Type": "application/json"}
    # El pipeline NIfTI del backend SIEMPRE usa el modelo binario para el
    # escaneo volumétrico (por diseño). Enviamos igualmente "model" para que,
    # si es "cheng", el backend clasifique ADEMÁS el tipo de tumor más
    # probable sobre los cortes ya detectados por el binario.
    model_type = model_type or APP_CONFIG.get("model_type", "binary")
    body = {"file_path": gs_url, "model": model_type}  # NIfTI uses file_path
    response = requests.post(cloud_run_url(), json=body, headers=headers,
                             timeout=APP_CONFIG.get("timeout_nifti", 900))
    if response.status_code == 200:
        return response.json()
    raise Exception(f"API Error: {response.status_code} — {response.text}")


def predict_image(gs_url, model_type=None):
    headers = {"Content-Type": "application/json"}
    model_type = model_type or APP_CONFIG.get("model_type", "binary")
    body = {"image_path": gs_url, "model": model_type}  # Images use image_path
    response = requests.post(cloud_run_url(), json=body, headers=headers,
                             timeout=APP_CONFIG.get("timeout_image", 600))
    if response.status_code == 200:
        return response.json()
    raise Exception(f"API Error: {response.status_code} — {response.text}")


def check_cloud_health(timeout=5):
    """Devuelve (ok, info_dict|None)."""
    try:
        r = requests.get(health_url(), timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None


# ======================================================
# CLOUD HEALTH (estado por modelo, compartido por la UI)
# ======================================================

AI_MODEL_SHORT = {"binary": "Binary", "cheng": "Cheng"}

# Último /health conocido. Se rellena en segundo plano y la UI se repinta
# a través de los listeners registrados (topbar, tarjeta AI MODEL, ...).
CLOUD_HEALTH = {"checked": False, "ok": False, "info": None}
_cloud_listeners = []

# estado -> (texto, color)
MODEL_STATUS_STYLE = {
    "checking": ("Comprobando…", COLOR_TEXT_TERTIARY),
    "offline": ("Sin conexión", COLOR_DANGER),
    "ready": ("Listo", COLOR_SUCCESS),
    "idle": ("No cargado", COLOR_WARNING),
    "unknown": ("Sin datos", COLOR_TEXT_TERTIARY),
}


def model_health(model_type=None):
    """Estado del modelo concreto según el último /health: (estado, versión).
    Lee models.<modelo>.{loaded,version}; los campos genéricos
    model_loaded/model_version solo describen el binario."""
    model_type = model_type or APP_CONFIG.get("model_type", "binary")
    if not CLOUD_HEALTH["checked"]:
        return "checking", ""
    if not CLOUD_HEALTH["ok"]:
        return "offline", ""
    info = CLOUD_HEALTH["info"] or {}
    per_model = (info.get("models") or {}).get(model_type)
    if per_model is None:
        # Backend antiguo sin detalle por modelo
        if model_type != "binary":
            return "unknown", ""
        per_model = {"loaded": info.get("model_loaded", False), "version": info.get("model_version", "")}
    return ("ready" if per_model.get("loaded") else "idle"), per_model.get("version", "")


def notify_cloud_listeners():
    for listener in list(_cloud_listeners):
        try:
            listener()
        except tk.TclError:
            # widget destruido (p. ej. tras cerrar sesión)
            _cloud_listeners.remove(listener)


def refresh_cloud_health():
    """Consulta /health en segundo plano y repinta los indicadores al terminar."""
    def _do():
        ok, info = check_cloud_health()

        def upd():
            CLOUD_HEALTH.update(checked=True, ok=ok, info=info)
            notify_cloud_listeners()
        try:
            root.after(0, upd)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
    with open(CREDENTIALS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_credentials(creds):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f)


# ======================================================
# ROUNDED-UI TOOLKIT (Canvas based, Apple-style widgets)
# ======================================================

def _round_rect_points(x1, y1, x2, y2, radius):
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]


def draw_round_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    return canvas.create_polygon(_round_rect_points(x1, y1, x2, y2, radius), smooth=True, **kwargs)


def center_window(win, root_win, width, height):
    win.update_idletasks()
    screen_w = root_win.winfo_screenwidth()
    screen_h = root_win.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")


class RoundedButton(tk.Canvas):
    """A pill / rounded-rect button drawn on a Canvas, Apple-style."""

    def __init__(self, parent, text, command=None, width=240, height=46, radius=None,
                 bg=COLOR_ACCENT, hover_bg=COLOR_ACCENT_HOVER, fg="white",
                 font=(FONT_FAMILY, 12, "bold"), outline="", **kwargs):
        parent_bg = kwargs.pop("parent_bg", None) or _safe_bg(parent)
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, bd=0, **kwargs)
        self.command = command
        self.bg_color = bg
        self.hover_color = hover_bg
        self.fg_color = fg
        self.enabled = True
        self.radius = radius if radius is not None else height / 2
        self.shape = draw_round_rect(self, 1, 1, width - 1, height - 1, self.radius,
                                     fill=bg, outline=outline)
        self.label = self.create_text(width / 2, height / 2, text=text, fill=fg, font=font)
        self.configure(cursor="hand2")
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _e):
        if self.enabled:
            self.itemconfig(self.shape, fill=self.hover_color)

    def _on_leave(self, _e):
        if self.enabled:
            self.itemconfig(self.shape, fill=self.bg_color)

    def _on_click(self, _e):
        if self.enabled and self.command:
            self.command()

    def set_text(self, text):
        self.itemconfig(self.label, text=text)

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.itemconfig(self.shape, fill=self.bg_color if enabled else COLOR_CARD_ALT)
        self.itemconfig(self.label, fill=self.fg_color if enabled else COLOR_TEXT_TERTIARY)
        self.configure(cursor="hand2" if enabled else "arrow")


class RoundedCard(tk.Canvas):
    """A rounded rectangle panel that hosts a normal tk.Frame for content."""

    def __init__(self, parent, width, height, radius=22, bg=COLOR_CARD,
                 border=COLOR_SEPARATOR, border_width=1, **kwargs):
        parent_bg = kwargs.pop("parent_bg", None) or _safe_bg(parent)
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, bd=0, **kwargs)
        draw_round_rect(self, 1, 1, width - 1, height - 1, radius,
                        fill=bg, outline=border, width=border_width)
        self.inner = tk.Frame(self, bg=bg)
        self.create_window(width / 2, height / 2, window=self.inner,
                           width=width - 2 * border_width - 2, height=height - 2 * border_width - 2)


class RoundedEntry(tk.Canvas):
    """A pill-shaped input field."""

    def __init__(self, parent, width=300, height=46, radius=None, bg=COLOR_CARD_ALT,
                 fg=COLOR_TEXT, show=None, font=(FONT_FAMILY, 12), **kwargs):
        parent_bg = kwargs.pop("parent_bg", None) or _safe_bg(parent)
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, bd=0, **kwargs)
        radius = radius if radius is not None else height / 2.6
        draw_round_rect(self, 1, 1, width - 1, height - 1, radius, fill=bg, outline="")
        self.entry = tk.Entry(self, bg=bg, fg=fg, insertbackground=fg, relief="flat",
                              show=show, font=font, bd=0, justify="left")
        self.create_window(width / 2, height / 2, window=self.entry, width=width - 34, height=height - 12)

    def get(self):
        return self.entry.get()

    def focus_set(self):
        self.entry.focus_set()

    def bind_key(self, sequence, func):
        self.entry.bind(sequence, func)

    def configure_readonly(self):
        self.entry.configure(state="readonly", readonlybackground=self.entry["bg"], fg=COLOR_TEXT_SECONDARY)

    def insert_text(self, text):
        self.entry.insert(0, text)

    def set_text(self, text):
        state = self.entry["state"]
        self.entry.configure(state="normal")
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.configure(state=state)


def _safe_bg(widget):
    try:
        return widget["bg"]
    except Exception:
        try:
            return widget.cget("background")
        except Exception:
            return COLOR_BG


def pill_badge(parent, text, fg, bg, width, height=48, font=(FONT_FAMILY, 15, "bold")):
    badge = RoundedCard(parent, width=width, height=height, radius=height / 2, bg=bg, border=bg, border_width=0)
    tk.Label(badge.inner, text=text, bg=bg, fg=fg, font=font).pack(expand=True)
    return badge


def stat_pill(parent, width, height, label_text, value_text, value_color=COLOR_TEXT):
    card = RoundedCard(parent, width=width, height=height, radius=18, bg=COLOR_CARD_ALT, border=COLOR_SEPARATOR)
    tk.Label(card.inner, text=label_text, bg=COLOR_CARD_ALT, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
    tk.Label(card.inner, text=value_text, bg=COLOR_CARD_ALT, fg=value_color,
             font=(FONT_FAMILY, 17, "bold")).pack(anchor="w", padx=18)
    return card


def draw_sparkline(canvas, values, width, height, color=COLOR_ACCENT, threshold=None):
    """Dibuja una línea de probabilidad por corte en un tk.Canvas (sin dependencias)."""
    canvas.delete("all")
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        canvas.create_text(width / 2, height / 2, text="Sin datos de cortes",
                           fill=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 9))
        return
    pad = 6
    n = len(values)
    xr = max(1, n - 1)
    # rejilla base
    draw_round_rect(canvas, 0, 0, width, height, 10, fill=COLOR_CARD_ALT, outline="")
    # línea de umbral
    if threshold is not None:
        ty = height - pad - threshold * (height - 2 * pad)
        canvas.create_line(pad, ty, width - pad, ty, fill=COLOR_TEXT_TERTIARY, dash=(3, 3))
    pts = []
    for i, v in enumerate(values):
        x = pad + (i / xr) * (width - 2 * pad)
        vv = v if isinstance(v, (int, float)) else 0.0
        y = height - pad - vv * (height - 2 * pad)
        pts.extend([x, y])
    if len(pts) >= 4:
        canvas.create_line(*pts, fill=color, width=2, smooth=True)


# ======================================================
# GLOBAL STYLE (ttk)
# ======================================================

def style_app(window):
    window.title("ScanIA · Clinical Brain Tumor Detection System")
    window.configure(bg=COLOR_BG)

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background=COLOR_BG)
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=(FONT_FAMILY, 11))
    style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 11))
    style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=(FONT_FAMILY, 28, "bold"))
    style.configure("Subtitle.TLabel", background=COLOR_BG, foreground=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 12))

    style.configure(
        "Loading.Horizontal.TProgressbar",
        troughcolor=COLOR_CARD_ALT, background=COLOR_ACCENT, bordercolor=COLOR_CARD,
        lightcolor=COLOR_ACCENT, darkcolor=COLOR_ACCENT, thickness=4,
    )
    style.configure(
        "Vertical.TScrollbar",
        background=COLOR_CARD, troughcolor=COLOR_BG, bordercolor=COLOR_BG,
        arrowcolor=COLOR_TEXT_SECONDARY, relief="flat",
    )


# ======================================================
# LOGIN & REGISTER
# ======================================================

def fit_window(win, root_win, width):
    """Dimensiona la ventana a lo que pide su contenido (fuentes y escalado de
    Windows incluidos) y la muestra centrada. Evita alturas fijas que recortan."""
    win.update_idletasks()
    center_window(win, root_win, max(width, win.winfo_reqwidth()), win.winfo_reqheight())
    win.deiconify()


def _auth_header(parent, title, subtitle):
    """Cabecera común de login/registro: icono en círculo suave + título."""
    badge = tk.Canvas(parent, width=60, height=60, bg=COLOR_BG, highlightthickness=0)
    badge.pack()
    badge.create_oval(2, 2, 58, 58, fill=COLOR_ACCENT_SOFT, outline="")
    badge.create_text(30, 31, text="🧠", fill=COLOR_ACCENT, font=(FONT_FAMILY, 20))
    tk.Label(parent, text=title, bg=COLOR_BG, fg=COLOR_TEXT, font=(FONT_FAMILY, 19, "bold")).pack(pady=(18, 2))
    tk.Label(parent, text=subtitle, bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 10)).pack(pady=(0, 26))


def _auth_field(parent, label, show=None):
    tk.Label(parent, text=label, bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(0, 6))
    entry = RoundedEntry(parent, width=300, height=44, show=show)
    entry.pack(pady=(0, 14))
    return entry


def _auth_error_label(parent):
    # Texto vacío = una línea reservada: el error no desplaza los botones
    label = tk.Label(parent, text="", bg=COLOR_BG, fg=COLOR_DANGER, font=(FONT_FAMILY, 9),
                     wraplength=300, justify="left")
    label.pack(anchor="w", pady=(0, 12))
    return label


def show_login(root):
    login_window = tk.Toplevel(root)
    login_window.withdraw()  # se muestra ya dimensionada (fit_window)
    login_window.title("Login · ScanIA")
    login_window.configure(bg=COLOR_BG)
    login_window.resizable(False, False)

    body = tk.Frame(login_window, bg=COLOR_BG)
    body.pack(padx=50, pady=(40, 34))

    _auth_header(body, "Welcome back", "Sign in to ScanIA")

    username_entry = _auth_field(body, "Username")
    password_entry = _auth_field(body, "Password", show="*")
    error_label = _auth_error_label(body)

    result = {'action': None}

    def login_action(event=None):
        username = username_entry.get().strip()
        password = password_entry.get().strip()
        if not username or not password:
            error_label.config(text="Please enter your username and password.")
            return
        creds = load_credentials()
        if username in creds and creds[username] == password:
            SESSION["user"] = username
            result['action'] = 'success'
            login_window.destroy()
        else:
            error_label.config(text="Incorrect username or password.")

    def register_action():
        result['action'] = 'register'
        login_window.destroy()

    RoundedButton(body, "Sign In", command=login_action, width=300, height=46).pack()

    tk.Frame(body, bg=COLOR_SEPARATOR, height=1).pack(fill="x", pady=(22, 14))
    tk.Label(body, text="New to ScanIA?", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9)).pack(pady=(0, 6))
    RoundedButton(body, "Create an account", command=register_action, width=300, height=40,
                  bg=COLOR_BG, hover_bg=COLOR_CARD, fg=COLOR_ACCENT, font=(FONT_FAMILY, 10, "bold")).pack()

    username_entry.bind_key("<Return>", login_action)
    password_entry.bind_key("<Return>", login_action)

    login_window.protocol("WM_DELETE_WINDOW", root.destroy)
    fit_window(login_window, root, 400)
    login_window.grab_set()
    username_entry.focus_set()
    root.wait_window(login_window)
    return result['action']


def show_register(root):
    reg_window = tk.Toplevel(root)
    reg_window.withdraw()  # se muestra ya dimensionada (fit_window)
    reg_window.title("Create Account · ScanIA")
    reg_window.configure(bg=COLOR_BG)
    reg_window.resizable(False, False)

    body = tk.Frame(reg_window, bg=COLOR_BG)
    body.pack(padx=50, pady=(40, 34))

    _auth_header(body, "Create account", "Register to access ScanIA")

    username_entry = _auth_field(body, "Username")
    password_entry = _auth_field(body, "Password", show="*")
    confirm_entry = _auth_field(body, "Confirm password", show="*")
    error_label = _auth_error_label(body)

    result = {'action': None}

    def create_action(event=None):
        username = username_entry.get().strip()
        password = password_entry.get().strip()
        confirm = confirm_entry.get().strip()
        if not username or not password:
            error_label.config(text="Please fill in all fields.")
            return
        if password != confirm:
            error_label.config(text="Passwords do not match.")
            return
        creds = load_credentials()
        if username in creds:
            error_label.config(text="Username already exists.")
            return
        creds[username] = password
        save_credentials(creds)
        messagebox.showinfo("Success", f"Account '{username}' created successfully.")
        result['action'] = 'registered'
        reg_window.destroy()

    RoundedButton(body, "Create Account", command=create_action, width=300, height=46).pack()

    tk.Frame(body, bg=COLOR_SEPARATOR, height=1).pack(fill="x", pady=(22, 14))
    tk.Label(body, text="Already have an account?", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9)).pack(pady=(0, 6))
    RoundedButton(body, "Back to sign in", command=reg_window.destroy, width=300, height=40,
                  bg=COLOR_BG, hover_bg=COLOR_CARD, fg=COLOR_ACCENT, font=(FONT_FAMILY, 10, "bold")).pack()

    username_entry.bind_key("<Return>", lambda e: password_entry.focus_set())
    password_entry.bind_key("<Return>", lambda e: confirm_entry.focus_set())
    confirm_entry.bind_key("<Return>", create_action)

    reg_window.protocol("WM_DELETE_WINDOW", reg_window.destroy)
    fit_window(reg_window, root, 400)
    reg_window.grab_set()
    username_entry.focus_set()
    root.wait_window(reg_window)
    return result['action']


# ======================================================
# SPLASH SCREEN
# ======================================================

# Color clave que Windows vuelve transparente: permite esquinas redondeadas
# en una ventana sin bordes (overrideredirect).
SPLASH_KEY_COLOR = "#010203"


def _rounded_logo(logo_path, size, radius, bg):
    """Logo con esquinas redondeadas sobre el fondo de la tarjeta (tipo icono de app)."""
    img = Image.open(logo_path).convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    alpha = ImageChops.multiply(img.getchannel("A"), mask)
    base = Image.new("RGBA", (size, size), bg)
    base.paste(img, (0, 0), alpha)
    return ImageTk.PhotoImage(base)


def show_splash(root, callback, logo_path, duration_ms=1900):
    """Splash sin bordes: tarjeta redondeada, fade in/out y barra de carga
    animada. `callback` se llama al empezar el fade out, de modo que la ventana
    principal aparece detrás mientras el splash se desvanece."""
    splash = tk.Toplevel(root)
    splash.withdraw()
    splash.overrideredirect(True)
    splash.configure(bg=SPLASH_KEY_COLOR)
    try:
        splash.attributes("-transparentcolor", SPLASH_KEY_COLOR)
        splash.attributes("-topmost", True)
        splash.attributes("-alpha", 0.0)
        can_fade = True
    except tk.TclError:
        splash.configure(bg=COLOR_BG)
        can_fade = False

    # Contenido en un frame que se mide antes de dibujar la tarjeta: el tamaño
    # se adapta a las fuentes/escala del sistema en lugar de ser fijo.
    shell = tk.Canvas(splash, bg=_safe_bg(splash), highlightthickness=0, bd=0)
    shell.pack()
    content = tk.Frame(shell, bg=COLOR_BG)

    try:
        logo = _rounded_logo(logo_path, 84, 20, COLOR_BG)
        tk.Label(content, image=logo, bg=COLOR_BG).pack()
        content.logo = logo
    except Exception:
        badge = tk.Canvas(content, width=84, height=84, bg=COLOR_BG, highlightthickness=0)
        badge.create_oval(2, 2, 82, 82, fill=COLOR_ACCENT_SOFT, outline="")
        badge.create_text(42, 43, text="🧠", fill=COLOR_ACCENT, font=(FONT_FAMILY, 28))
        badge.pack()

    tk.Label(content, text="ScanIA", fg=COLOR_TEXT, bg=COLOR_BG, font=(FONT_FAMILY, 22, "bold")).pack(pady=(20, 2))
    tk.Label(content, text="Clinical Brain Tumor Detection System", fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG,
             font=(FONT_FAMILY, 10)).pack(pady=(0, 28))

    track_w, track_h, seg_w = 160, 4, 52
    loader = tk.Canvas(content, width=track_w, height=track_h, bg=COLOR_BG, highlightthickness=0)
    loader.pack()
    draw_round_rect(loader, 0, 0, track_w, track_h, track_h / 2, fill=COLOR_CARD_ALT, outline="")
    segment = draw_round_rect(loader, 0, 0, seg_w, track_h, track_h / 2, fill=COLOR_ACCENT, outline="")

    tk.Label(content, text="Preparing workspace…", fg=COLOR_TEXT_TERTIARY, bg=COLOR_BG,
             font=(FONT_FAMILY, 8)).pack(pady=(12, 0))

    content.update_idletasks()
    pad_x, pad_y = 64, 44
    w = max(400, content.winfo_reqwidth() + 2 * pad_x)
    h = content.winfo_reqheight() + 2 * pad_y
    shell.configure(width=w, height=h)
    draw_round_rect(shell, 1, 1, w - 1, h - 1, 26, fill=COLOR_BG, outline=COLOR_SEPARATOR)
    shell.create_window(w / 2, h / 2, window=content)
    center_window(splash, root, w, h)
    splash.deiconify()

    start = time.time()

    def animate_loader():
        if not splash.winfo_exists():
            return
        # vaivén con aceleración suave (coseno), sin saltos
        phase = ((time.time() - start) % 1.4) / 1.4
        ease = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
        x1 = ease * (track_w - seg_w)
        loader.coords(segment, *_round_rect_points(x1, 0, x1 + seg_w, track_h, track_h / 2))
        splash.after(16, animate_loader)

    def fade(from_a, to_a, ms, then=None):
        steps = max(1, ms // 16)

        def step(i=0):
            if not splash.winfo_exists():
                return
            t = i / steps
            eased = 1 - (1 - t) ** 3  # ease-out cúbico
            splash.attributes("-alpha", from_a + (to_a - from_a) * eased)
            if i < steps:
                splash.after(16, step, i + 1)
            elif then:
                then()
        step()

    def finish():
        callback()
        if can_fade:
            fade(1.0, 0.0, 320, splash.destroy)
        else:
            splash.destroy()

    animate_loader()
    if can_fade:
        fade(0.0, 1.0, 280)
    splash.after(duration_ms, finish)


# ======================================================
# LOADING & PROGRESS
# ======================================================

def show_loading(text="Analyzing image... This may take a few seconds."):
    set_upload_state("analyzing", detail=text)


def hide_loading():
    progress.stop()
    progress.grid_remove()


# ======================================================
# UPLOAD CARD STATES
# ======================================================
# Un único punto de verdad para la tarjeta de subida: icono, textos, preview,
# barra de progreso y botón cambian siempre juntos.

UPLOAD_STATES = {
    "idle": {
        "glyph": "↑", "fg": COLOR_ACCENT, "soft": COLOR_ACCENT_SOFT,
        "title": "Select an MRI scan to begin",
        "detail": "JPG, PNG or NIfTI (.nii / .nii.gz)",
        "button": "Select Image / NIfTI", "enabled": True,
    },
    "selected": {
        "glyph": "▣", "fg": COLOR_ACCENT, "soft": COLOR_ACCENT_SOFT,
        "title": "File selected",
        "detail": "{name}",
        "button": "Change file", "enabled": True,
    },
    "analyzing": {
        "glyph": "…", "fg": COLOR_ACCENT, "soft": COLOR_ACCENT_SOFT,
        "title": "Analyzing {name}",
        "detail": "Uploading and running the model…",
        "button": "Analyzing…", "enabled": False,
    },
    "done": {
        "glyph": "✓", "fg": COLOR_SUCCESS, "soft": COLOR_SUCCESS_SOFT,
        "title": "Result ready",
        "detail": "The full report opened in a new window.",
        "button": "Analyze another scan", "enabled": True,
    },
    "error": {
        "glyph": "!", "fg": COLOR_DANGER, "soft": COLOR_DANGER_SOFT,
        "title": "Analysis failed",
        "detail": "Check the connection and try again.",
        "button": "Try again", "enabled": True,
    },
}

# widgets de la tarjeta + archivo actual (se rellena en start_app)
UPLOAD_UI = {}


def _set_preview(file_path):
    """Rellena el preview_card según el archivo (o lo vacía si no hay)."""
    if not file_path:
        image_display.configure(image="", text="No file selected", fg=COLOR_TEXT_TERTIARY)
        image_display.image = None
        return
    name = os.path.basename(file_path)
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(file_path)
        img.thumbnail(UPLOAD_UI["preview_size"])
        photo = ImageTk.PhotoImage(img)
        image_display.configure(image=photo, text="")
        image_display.image = photo
    else:
        image_display.configure(image="", text=f"▦\n3D volume\n{name}", fg=COLOR_TEXT_SECONDARY)
        image_display.image = None


def set_upload_state(state, file_path=None, detail=None):
    """Cambia la tarjeta de subida a uno de UPLOAD_STATES.
    file_path solo se pasa al elegir archivo; el resto de estados lo reutilizan."""
    if state == "idle":
        UPLOAD_UI["file"] = None
        _set_preview(None)
    elif file_path:
        _set_preview(file_path)  # puede lanzar si la imagen no se puede abrir
        UPLOAD_UI["file"] = file_path

    spec = UPLOAD_STATES[state]
    name = os.path.basename(UPLOAD_UI.get("file") or "")

    icon = UPLOAD_UI["icon"]
    icon.itemconfig(UPLOAD_UI["icon_bg"], fill=spec["soft"])
    icon.itemconfig(UPLOAD_UI["icon_glyph"], text=spec["glyph"], fill=spec["fg"])

    UPLOAD_UI["title"].config(text=spec["title"].format(name=name))
    status_label.config(text=detail or spec["detail"].format(name=name))

    button = UPLOAD_UI["button"]
    button.set_text(spec["button"])
    button.set_enabled(spec["enabled"])

    if state == "analyzing":
        progress.grid()
        progress.start(10)
    else:
        hide_loading()


# ======================================================
# IMAGE RESULT MODAL
# ======================================================

def generate_unique_gradcam_filename():
    timestamp = int(time.time())
    unique_id = random.randint(1000, 9999)
    return f"gradcam_result_{timestamp}_{unique_id}.jpg"


def _disclaimer_label(parent, width=440):
    return tk.Label(parent, text=DISCLAIMER_TEXT, bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
                    font=(FONT_FAMILY, 8), wraplength=width, justify="center")


def show_prediction_modal(label, probability, gradcam_url=None, extra=None):
    extra = extra or {}
    modal = tk.Toplevel(root)
    modal.title("Analysis Result · ScanIA")
    modal.configure(bg=COLOR_BG)
    modal.resizable(False, False)

    is_tumor = label.lower() == "tumor"
    accent = COLOR_DANGER if is_tumor else COLOR_SUCCESS
    accent_soft = COLOR_DANGER_SOFT if is_tumor else COLOR_SUCCESS_SOFT
    icon = "⚠️" if is_tumor else "✅"
    diagnosis_text = "Tumor detected" if is_tumor else "No tumor detected"

    threshold = float(extra.get("threshold", APP_CONFIG.get("visual_threshold", 0.5)))
    near = abs((probability / 100.0) - threshold) <= 0.10  # cerca del umbral

    height = 760 if gradcam_url else (460 if near else 430)
    center_window(modal, root, 480, height)

    tk.Label(modal, text="Analysis Result", bg=COLOR_BG, fg=COLOR_TEXT, font=(FONT_FAMILY, 20, "bold")).pack(pady=(26, 16))

    content = tk.Frame(modal, bg=COLOR_BG)
    content.pack(fill="both", expand=True, padx=30)

    badge = pill_badge(content, f"{icon}  {diagnosis_text}", fg=accent, bg=accent_soft, width=420, height=54)
    badge.pack(pady=(0, 14))

    if near:
        warn = pill_badge(content, "⚠  Resultado cercano al umbral — revisar con cautela",
                          fg=COLOR_WARNING, bg=COLOR_WARNING_SOFT, width=420, height=40,
                          font=(FONT_FAMILY, 10, "bold"))
        warn.pack(pady=(0, 12))

    prob_card = RoundedCard(content, width=420, height=118, radius=20)
    prob_card.pack(pady=(0, 4))
    tk.Label(prob_card.inner, text="ESTIMATED PROBABILITY", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=20, pady=(14, 2))
    row = tk.Frame(prob_card.inner, bg=COLOR_CARD)
    row.pack(anchor="w", padx=20, fill="x")
    tk.Label(row, text=f"{probability:.2f}%", bg=COLOR_CARD, fg=COLOR_TEXT, font=(FONT_FAMILY, 22, "bold")).pack(side="left")
    tk.Label(row, text=f"  ·  no tumor {100 - probability:.2f}%", bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 11)).pack(side="left")

    bar_canvas = tk.Canvas(prob_card.inner, width=380, height=8, bg=COLOR_CARD, highlightthickness=0)
    bar_canvas.pack(anchor="w", padx=20, pady=(10, 0))
    draw_round_rect(bar_canvas, 0, 0, 380, 8, 4, fill=COLOR_CARD_ALT, outline="")
    fill_w = max(6, min(380, 380 * probability / 100.0))
    draw_round_rect(bar_canvas, 0, 0, fill_w, 8, 4, fill=accent, outline="")
    # marca de umbral
    tx = max(0, min(380, 380 * threshold))
    bar_canvas.create_line(tx, 0, tx, 8, fill=COLOR_TEXT, width=1)
    tk.Label(prob_card.inner, text=f"Umbral: {threshold * 100:.0f}%", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 8)).pack(anchor="w", padx=20, pady=(4, 0))

    if gradcam_url:
        tk.Label(content, text="GRAD-CAM HEATMAP (explicabilidad, no segmentación)", bg=COLOR_BG,
                 fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(18, 8))
        try:
            response = requests.get(gradcam_url, stream=True, timeout=30)
            response.raise_for_status()
            gradcam_img = Image.open(BytesIO(response.content))
            gradcam_img = gradcam_img.resize((300, 300))
            gradcam_photo = ImageTk.PhotoImage(gradcam_img)

            img_card = RoundedCard(content, width=300, height=300, radius=20, bg=COLOR_CARD_ALT)
            img_card.pack()
            gradcam_label = tk.Label(img_card.inner, image=gradcam_photo, bg=COLOR_CARD_ALT)
            gradcam_label.image = gradcam_photo
            gradcam_label.pack(expand=True)
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Image Download Error", f"Could not download Grad-CAM image: {str(e)}")
        except IOError as e:
            messagebox.showerror("Image Processing Error", f"Could not process Grad-CAM image: {str(e)}")

    _disclaimer_label(content, width=420).pack(pady=(16, 4))

    RoundedButton(content, "Close", command=modal.destroy, width=420, height=46,
                  bg=COLOR_CARD, hover_bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
                  font=(FONT_FAMILY, 11, "bold")).pack(pady=(10, 20))


# ======================================================
# CHENG MODEL RESULT MODAL (multicategoría)
# ======================================================

def show_cheng_prediction_modal(prediction):
    """Ventana de resultado para el modelo 'Cheng Brain Tumor Classification'
    (multicategoría). Muestra: modelo utilizado, clase predicha, probabilidad,
    distribución de probabilidades por clase y Grad-CAM."""
    prediction = prediction or {}
    modal = tk.Toplevel(root)
    modal.title("Analysis Result (Cheng) · ScanIA")
    modal.configure(bg=COLOR_BG)
    modal.resizable(False, False)

    label = prediction.get("label", "N/A")
    probability = float(prediction.get("prediction", 0.0)) * 100
    classes_probability = prediction.get("classes_probability", {}) or {}
    gradcam_url = prediction.get("gradcam_url")
    model_version = prediction.get("model_version", "")

    height = 780 if gradcam_url else 560
    center_window(modal, root, 480, height)

    tk.Label(modal, text="Analysis Result", bg=COLOR_BG, fg=COLOR_TEXT, font=(FONT_FAMILY, 20, "bold")).pack(pady=(26, 4))
    tk.Label(modal, text=AI_MODEL_LABELS.get("cheng", "Cheng Brain Tumor Classification"),
             bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 10)).pack(pady=(0, 16))

    content = tk.Frame(modal, bg=COLOR_BG)
    content.pack(fill="both", expand=True, padx=30)

    badge = pill_badge(content, f"🧠  {label}", fg=COLOR_ACCENT, bg=COLOR_ACCENT_SOFT, width=420, height=54)
    badge.pack(pady=(0, 14))

    prob_card = RoundedCard(content, width=420, height=90, radius=20)
    prob_card.pack(pady=(0, 4))
    tk.Label(prob_card.inner, text="ESTIMATED PROBABILITY (PREDICTED CLASS)", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=20, pady=(14, 2))
    tk.Label(prob_card.inner, text=f"{probability:.2f}%", bg=COLOR_CARD, fg=COLOR_TEXT,
             font=(FONT_FAMILY, 22, "bold")).pack(anchor="w", padx=20)

    # ---- Distribución de probabilidades por clase ----
    if classes_probability:
        tk.Label(content, text="CLASS PROBABILITY DISTRIBUTION", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(18, 8))

        dist_card = RoundedCard(content, width=420, height=32 * len(classes_probability) + 24, radius=18,
                                bg=COLOR_CARD_ALT)
        dist_card.pack(pady=(0, 6))
        dist_inner = tk.Frame(dist_card.inner, bg=COLOR_CARD_ALT)
        dist_inner.pack(fill="both", expand=True, padx=16, pady=12)

        sorted_classes = sorted(classes_probability.items(), key=lambda kv: kv[1], reverse=True)
        for class_name, class_prob in sorted_classes:
            row = tk.Frame(dist_inner, bg=COLOR_CARD_ALT)
            row.pack(fill="x", pady=4)
            is_top = (class_name == label)
            tk.Label(row, text=class_name, bg=COLOR_CARD_ALT,
                     fg=COLOR_TEXT if is_top else COLOR_TEXT_SECONDARY,
                     font=(FONT_FAMILY, 10, "bold" if is_top else "normal"), width=16, anchor="w").pack(side="left")

            bar_bg = tk.Canvas(row, width=180, height=8, bg=COLOR_CARD_ALT, highlightthickness=0)
            bar_bg.pack(side="left", padx=(6, 8))
            draw_round_rect(bar_bg, 0, 0, 180, 8, 4, fill=COLOR_CARD, outline="")
            fill_w = max(4, min(180, 180 * float(class_prob)))
            draw_round_rect(bar_bg, 0, 0, fill_w, 8, 4, fill=COLOR_ACCENT if is_top else COLOR_TEXT_TERTIARY, outline="")

            tk.Label(row, text=f"{float(class_prob) * 100:.1f}%", bg=COLOR_CARD_ALT,
                     fg=COLOR_TEXT if is_top else COLOR_TEXT_SECONDARY,
                     font=(FONT_FAMILY, 10, "bold" if is_top else "normal")).pack(side="left")

    # ---- Grad-CAM ----
    if gradcam_url:
        tk.Label(content, text="GRAD-CAM HEATMAP (explicabilidad, no segmentación)", bg=COLOR_BG,
                 fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(18, 8))
        try:
            response = requests.get(gradcam_url, stream=True, timeout=30)
            response.raise_for_status()
            gradcam_img = Image.open(BytesIO(response.content))
            gradcam_img = gradcam_img.resize((300, 300))
            gradcam_photo = ImageTk.PhotoImage(gradcam_img)

            img_card = RoundedCard(content, width=300, height=300, radius=20, bg=COLOR_CARD_ALT)
            img_card.pack()
            gradcam_label = tk.Label(img_card.inner, image=gradcam_photo, bg=COLOR_CARD_ALT)
            gradcam_label.image = gradcam_photo
            gradcam_label.pack(expand=True)
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Image Download Error", f"Could not download Grad-CAM image: {str(e)}")
        except IOError as e:
            messagebox.showerror("Image Processing Error", f"Could not process Grad-CAM image: {str(e)}")

    if model_version:
        tk.Label(content, text=f"Model version: {model_version}", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
                 font=(FONT_FAMILY, 8)).pack(anchor="w", pady=(10, 0))

    _disclaimer_label(content, width=420).pack(pady=(16, 4))

    RoundedButton(content, "Close", command=modal.destroy, width=420, height=46,
                  bg=COLOR_CARD, hover_bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
                  font=(FONT_FAMILY, 11, "bold")).pack(pady=(10, 20))


# ======================================================
# NIFTI 3D RESULT MODAL
# ======================================================

def show_nifti_modal(result_json):
    modal = tk.Toplevel(root)
    modal.title("3D Analysis Result (NIfTI) · ScanIA")
    modal.configure(bg=COLOR_BG)
    modal.resizable(False, False)
    extra_height = 58 if result_json.get("cheng_label") else 0
    center_window(modal, root, 600, 820 + extra_height)

    tk.Label(modal, text="3D Volumetric Analysis", bg=COLOR_BG, fg=COLOR_TEXT,
             font=(FONT_FAMILY, 20, "bold")).pack(pady=(24, 6))
    _disclaimer_label(modal, width=540).pack(pady=(0, 10))

    content = tk.Frame(modal, bg=COLOR_BG)
    content.pack(fill="both", expand=True, padx=30)

    nifti_filename = result_json.get("nifti_filename", "N/A")
    volume_shape = result_json.get("volume_shape", [])
    best_slice = result_json.get("best_slice", None)
    best_score = result_json.get("best_score", None)
    top10 = result_json.get("top10_slices", [])
    render3d_url = result_json.get("render3d_url") or result_json.get("viewer_3d_url")
    stats = result_json.get("score_statistics", {}) or {}
    slice_scores = result_json.get("slice_scores", []) or []
    slice_count = result_json.get("slice_count")
    times = result_json.get("processing_times", {}) or {}
    threshold = float(result_json.get("threshold", 0.75))

    file_card = RoundedCard(content, width=540, height=52, radius=16, bg=COLOR_CARD_ALT)
    file_card.pack(pady=(0, 14))
    tk.Label(file_card.inner, text=f"📄  {nifti_filename}", bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
             font=(FONT_FAMILY, 11, "bold")).pack(expand=True)

    # ---- Tipo más probable según el modelo Cheng (solo si estaba seleccionado) ----
    cheng_label = result_json.get("cheng_label")
    if cheng_label:
        cheng_prob = result_json.get("cheng_probability")
        cheng_text = f"🧠  Tipo más probable: {cheng_label}"
        if cheng_prob is not None:
            cheng_text += f" ({float(cheng_prob) * 100:.1f}%)"
        cheng_badge = pill_badge(content, cheng_text, fg=COLOR_ACCENT, bg=COLOR_ACCENT_SOFT,
                                 width=540, height=44, font=(FONT_FAMILY, 11, "bold"))
        cheng_badge.pack(pady=(0, 14))

    stats_row = tk.Frame(content, bg=COLOR_BG)
    stats_row.pack(pady=(0, 14))
    if volume_shape:
        stat_pill(stats_row, width=175, height=86, label_text="VOLUME (H,W,D)",
                  value_text=str(volume_shape), value_color=COLOR_ACCENT).pack(side="left", padx=(0, 10))
    if best_slice is not None and best_score is not None:
        stat_pill(stats_row, width=175, height=86, label_text="BEST SLICE",
                  value_text=f"{best_slice} · {float(best_score) * 100:.1f}%",
                  value_color=COLOR_SUCCESS).pack(side="left", padx=(0, 10))
    if slice_count is not None:
        stat_pill(stats_row, width=160, height=86, label_text="SLICES",
                  value_text=str(slice_count), value_color=COLOR_TEXT).pack(side="left")

    # ---- Estadística avanzada (si el backend la envía) ----
    if stats:
        tk.Label(content, text="SCORE STATISTICS", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(2, 6))
        st_card = RoundedCard(content, width=540, height=92, radius=16, bg=COLOR_CARD_ALT)
        st_card.pack(pady=(0, 12))
        grid = tk.Frame(st_card.inner, bg=COLOR_CARD_ALT)
        grid.pack(expand=True, padx=14, pady=10)

        def cell(r, c, k, v):
            f = tk.Frame(grid, bg=COLOR_CARD_ALT)
            f.grid(row=r, column=c, padx=12, sticky="w")
            tk.Label(f, text=k, bg=COLOR_CARD_ALT, fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 8, "bold")).pack(anchor="w")
            tk.Label(f, text=v, bg=COLOR_CARD_ALT, fg=COLOR_TEXT, font=(FONT_FAMILY, 12, "bold")).pack(anchor="w")

        def pct(x):
            return f"{float(x) * 100:.1f}%" if isinstance(x, (int, float)) else "—"

        cell(0, 0, "MÁX", pct(stats.get("maximum")))
        cell(0, 1, "MEDIA", pct(stats.get("mean")))
        cell(0, 2, "MEDIANA", pct(stats.get("median")))
        cell(0, 3, "STD", pct(stats.get("standard_deviation")))
        cell(1, 0, "P90", pct(stats.get("percentile_90")))
        cell(1, 1, "P95", pct(stats.get("percentile_95")))
        cell(1, 2, "> UMBRAL", str(stats.get("slices_above_threshold", "—")))
        cell(1, 3, "% > UMB.", f"{stats.get('percentage_above_threshold', 0):.1f}%")

    # ---- Sparkline de probabilidad por corte ----
    if slice_scores:
        tk.Label(content, text="PROBABILIDAD POR CORTE", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(2, 6))
        spark = tk.Canvas(content, width=540, height=70, bg=COLOR_BG, highlightthickness=0)
        spark.pack(pady=(0, 12))
        draw_sparkline(spark, slice_scores, 540, 70, color=COLOR_ACCENT, threshold=threshold)

    # ---- Top slices ----
    tk.Label(content, text="TOP 10 SLICES", bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(0, 6))
    top10_text = ", ".join([str(x) for x in top10]) if top10 else "N/A"
    top10_card = RoundedCard(content, width=540, height=48, radius=14, bg=COLOR_CARD_ALT)
    top10_card.pack(pady=(0, 12))
    tk.Label(top10_card.inner, text=top10_text, bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
             font=(FONT_FAMILY, 10), wraplength=500, justify="left").pack(expand=True, padx=16)

    # ---- Tiempos de procesamiento ----
    if times:
        total = times.get("total_seconds")
        tsub = []
        if "download_seconds" in times: tsub.append(f"subida {times['download_seconds']}s")
        if "inference_seconds" in times: tsub.append(f"inferencia {times['inference_seconds']}s")
        if "gradcam_seconds" in times: tsub.append(f"grad-cam {times['gradcam_seconds']}s")
        if "viewer_generation_seconds" in times: tsub.append(f"visor {times['viewer_generation_seconds']}s")
        txt = (f"⏱ Total {total}s  ·  " + "  ·  ".join(tsub)) if total is not None else "  ·  ".join(tsub)
        tk.Label(content, text=txt, bg=COLOR_BG, fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(0, 12))

    def open_3d():
        if render3d_url:
            webbrowser.open(render3d_url)
        else:
            messagebox.showwarning("No render available", "No se devolvió una URL de render 3D en la respuesta.")

    RoundedButton(content, "🧊  Abrir visor 3D", command=open_3d, width=540, height=48).pack(pady=(0, 12))

    if render3d_url:
        url_entry = RoundedEntry(content, width=540, height=40, bg=COLOR_CARD_ALT)
        url_entry.pack(pady=(0, 12))
        url_entry.insert_text(render3d_url)
        url_entry.configure_readonly()

    RoundedButton(content, "Close", command=modal.destroy, width=540, height=44,
                  bg=COLOR_CARD, hover_bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
                  font=(FONT_FAMILY, 11, "bold")).pack(pady=(0, 18))


# ======================================================
# SETTINGS MODAL
# ======================================================

def show_settings():
    modal = tk.Toplevel(root)
    modal.title("Settings · ScanIA")
    modal.configure(bg=COLOR_BG)
    modal.resizable(False, False)
    center_window(modal, root, 460, 520)

    tk.Label(modal, text="Configuración", bg=COLOR_BG, fg=COLOR_TEXT, font=(FONT_FAMILY, 20, "bold")).pack(pady=(24, 18))
    form = tk.Frame(modal, bg=COLOR_BG)
    form.pack()

    tk.Label(form, text="URL de la API (Cloud Run)", bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 10)).pack(anchor="w", pady=(0, 6))
    api_entry = RoundedEntry(form, width=380, height=44)
    api_entry.pack(pady=(0, 14))
    api_entry.insert_text(APP_CONFIG.get("api_url", DEFAULT_CLOUD_RUN_URL))

    tk.Label(form, text="Timeout NIfTI (segundos)", bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 10)).pack(anchor="w", pady=(0, 6))
    to_entry = RoundedEntry(form, width=380, height=44)
    to_entry.pack(pady=(0, 14))
    to_entry.insert_text(str(APP_CONFIG.get("timeout_nifti", 900)))

    tk.Label(form, text="Umbral visual (0-1)", bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 10)).pack(anchor="w", pady=(0, 6))
    th_entry = RoundedEntry(form, width=380, height=44)
    th_entry.pack(pady=(0, 6))
    th_entry.insert_text(str(APP_CONFIG.get("visual_threshold", 0.5)))

    status = tk.Label(form, text="", bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 9))
    status.pack(anchor="w", pady=(10, 8))

    def test_conn():
        status.config(text="Comprobando conexión…", fg=COLOR_TEXT_SECONDARY)

        def _do():
            APP_CONFIG["api_url"] = api_entry.get().strip() or DEFAULT_CLOUD_RUN_URL
            ok, info = check_cloud_health()
            def upd():
                CLOUD_HEALTH.update(checked=True, ok=ok, info=info)
                notify_cloud_listeners()
                if ok:
                    model_type = APP_CONFIG.get("model_type", "binary")
                    state, ver = model_health(model_type)
                    text, color = MODEL_STATUS_STYLE[state]
                    status.config(text=f"✅ Conectado · {AI_MODEL_SHORT.get(model_type, model_type)}: "
                                       f"{text.lower()} {ver}", fg=color)
                else:
                    status.config(text="❌ Sin conexión con la API", fg=COLOR_DANGER)
            root.after(0, upd)
        threading.Thread(target=_do, daemon=True).start()

    def save_and_close():
        APP_CONFIG["api_url"] = api_entry.get().strip() or DEFAULT_CLOUD_RUN_URL
        try:
            APP_CONFIG["timeout_nifti"] = int(float(to_entry.get().strip()))
        except Exception:
            pass
        try:
            APP_CONFIG["visual_threshold"] = max(0.0, min(1.0, float(th_entry.get().strip())))
        except Exception:
            pass
        save_config(APP_CONFIG)
        modal.destroy()
        refresh_cloud_health()  # la URL puede haber cambiado

    RoundedButton(form, "Comprobar conexión con Google Cloud", command=test_conn, width=380, height=44,
                  bg=COLOR_CARD, hover_bg=COLOR_CARD_ALT, fg=COLOR_TEXT, font=(FONT_FAMILY, 11)).pack(pady=(6, 10))
    RoundedButton(form, "Guardar", command=save_and_close, width=380, height=48).pack()


# ======================================================
# AI MODEL SELECTOR
# ======================================================

def build_model_selector(parent, width=520, on_change=None):
    """Tarjeta de selección de modelo de IA ('AI Model'). Guarda la elección
    en APP_CONFIG['model_type'] y la persiste en config.json. Cada fila muestra
    el estado de ese modelo en Cloud (según el último /health)."""
    card = RoundedCard(parent, width=width, height=184, radius=20)
    card.pack(pady=(0, 20))

    tk.Label(card.inner, text="AI MODEL", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY,
             font=(FONT_FAMILY, 8, "bold")).pack(anchor="w", padx=24, pady=(18, 6))

    rows_frame = tk.Frame(card.inner, bg=COLOR_CARD)
    rows_frame.pack(fill="x", padx=24)

    option_widgets = {}

    def refresh_visuals():
        current = APP_CONFIG.get("model_type", "binary")
        for value, widgets in option_widgets.items():
            active = (value == current)
            widgets["dot"].config(text="◉" if active else "○",
                                  fg=COLOR_ACCENT if active else COLOR_TEXT_TERTIARY)
            widgets["label"].config(fg=COLOR_TEXT if active else COLOR_TEXT_SECONDARY,
                                    font=(FONT_FAMILY, 11, "bold" if active else "normal"))
            state, version = model_health(value)
            text, color = MODEL_STATUS_STYLE[state]
            widgets["status"].config(text=f"● {text}", fg=color)
            widgets["caption"].config(text=version or " ")

    def select_model(value):
        if APP_CONFIG.get("model_type") == value:
            return
        APP_CONFIG["model_type"] = value
        save_config(APP_CONFIG)
        refresh_visuals()
        if on_change:
            on_change(value)

    for i, (value, display_name) in enumerate(AI_MODEL_OPTIONS):
        if i:
            tk.Frame(rows_frame, bg=COLOR_SEPARATOR, height=1).pack(fill="x", padx=(30, 0))

        row = tk.Frame(rows_frame, bg=COLOR_CARD, cursor="hand2")
        row.pack(fill="x", pady=8)

        dot = tk.Label(row, text="○", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 13))
        dot.pack(side="left", anchor="n", padx=(0, 12))

        status = tk.Label(row, text="", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 9))
        status.pack(side="right", anchor="n", pady=(3, 0))

        text_col = tk.Frame(row, bg=COLOR_CARD)
        text_col.pack(side="left", fill="x", expand=True)
        lbl = tk.Label(text_col, text=display_name, bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY,
                       font=(FONT_FAMILY, 11), anchor="w")
        lbl.pack(fill="x")
        caption = tk.Label(text_col, text=" ", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY,
                           font=(FONT_FAMILY, 8), anchor="w")
        caption.pack(fill="x")

        option_widgets[value] = {"dot": dot, "label": lbl, "caption": caption, "status": status}

        for widget in (row, dot, text_col, lbl, caption, status):
            widget.bind("<Button-1>", lambda _e, v=value: select_model(v))

    refresh_visuals()
    _cloud_listeners.append(refresh_visuals)
    return card


# ======================================================
# FILE SELECTION & ANALYSIS
# ======================================================

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select MRI file",
        filetypes=[
            ("JPG Image", "*.jpg"),
            ("PNG Image", "*.png"),
            ("NIfTI", "*.nii"),
            ("NIfTI GZ", "*.nii.gz"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        return

    try:
        ext_lower = file_path.lower()

        # Preview + textos coherentes con el archivo elegido
        set_upload_state("selected", file_path=file_path)

        cleaned_name = clean_filename(os.path.basename(file_path))

        if ext_lower.endswith(".nii") or ext_lower.endswith(".nii.gz"):
            show_loading("Analyzing 3D volume (NIfTI)... This may take 1-3 minutes.")
        else:
            show_loading("Analyzing image... This may take a few seconds.")

        def result_ready(summary):
            root.after(0, lambda: set_upload_state(
                "done", detail=f"{summary}\nThe full report opened in a new window."))

        def task():
            try:
                selected_model_type = APP_CONFIG.get("model_type", "binary")

                if ext_lower.endswith(".nii") or ext_lower.endswith(".nii.gz"):
                    public_url = upload_to_gcs(BUCKET_NIFTI, file_path, cleaned_name)
                    gs_url = convert_public_url_to_gs(public_url)
                    prediction = predict_file(gs_url, selected_model_type)
                    cheng_label = prediction.get("cheng_label")
                    result_ready("3D volume analyzed" + (f" · Cheng: {cheng_label}" if cheng_label else ""))
                    root.after(0, lambda: show_nifti_modal(prediction))
                    return

                public_url = upload_to_gcs(BUCKET_NAME, file_path, cleaned_name)
                gs_url = convert_public_url_to_gs(public_url)
                prediction = predict_image(gs_url, selected_model_type)

                response_model_type = prediction.get("model", selected_model_type)

                if response_model_type == "cheng":
                    cheng_prob = float(prediction.get("prediction", 0.0)) * 100
                    result_ready(f"Cheng · {prediction.get('label', 'N/A')} · {cheng_prob:.1f}%")
                    root.after(0, lambda: show_cheng_prediction_modal(prediction))
                    return

                label = prediction.get("label")
                score = float(prediction.get("prediction", 0.0))
                gradcam_url = prediction.get("gradcam_url") if score >= 0.9 else None
                probability = score * 100

                if gradcam_url:
                    gradcam_url = gradcam_url.replace("gradcam_result.jpg", generate_unique_gradcam_filename())

                result_ready(f"Binary · {label} · {probability:.1f}%")
                root.after(0, lambda: show_prediction_modal(label, probability, gradcam_url, prediction))

            except Exception as e:
                root.after(0, lambda: set_upload_state("error"))
                root.after(0, lambda error_message=str(e): messagebox.showerror("Error", error_message))

        threading.Thread(target=task, daemon=True).start()

    except Exception as e:
        set_upload_state("idle")
        messagebox.showerror("Error", f"Cannot load file:\n{e}")
        return


# ======================================================
# MAIN APPLICATION
# ======================================================

def start_app():
    global root, status_label, progress, image_display, cloud_status_label

    root = tk.Tk()
    style_app(root)
    root.withdraw()

    while True:
        action = show_login(root)
        if action == 'success':
            break
        elif action == 'register':
            show_register(root)
            continue
        else:
            root.destroy()
            return

    show_splash(root, lambda: root.deiconify(), LOGO_PATH)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    win_width = int(screen_width * 0.65)
    win_height = int(screen_height * 0.75)
    x = (screen_width - win_width) // 2
    y = (screen_height - win_height) // 2
    root.geometry(f"{win_width}x{win_height}+{x}+{y}")
    root.configure(bg=COLOR_BG)
    root.minsize(760, 680)

    def on_close():
        if APP_CONFIG.get("confirm_on_close", True):
            if not messagebox.askokcancel("Salir", "¿Cerrar ScanIA?"):
                return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Los listeners de una sesión anterior (logout -> start_app) ya no sirven
    _cloud_listeners.clear()

    CONTENT_WIDTH = 520  # ancho común de todas las tarjetas

    # ---- Top bar ----
    topbar = tk.Frame(root, bg=COLOR_BG_ELEVATED, height=64)
    topbar.pack(side="top", fill="x")
    topbar.pack_propagate(False)

    brand = tk.Frame(topbar, bg=COLOR_BG_ELEVATED)
    brand.pack(side="left", padx=28)
    tk.Label(brand, text="🧠", bg=COLOR_BG_ELEVATED, fg=COLOR_ACCENT, font=(FONT_FAMILY, 15)).pack(side="left")
    tk.Label(brand, text="  ScanIA", bg=COLOR_BG_ELEVATED, fg=COLOR_TEXT, font=(FONT_FAMILY, 13, "bold")).pack(side="left")

    # Zona derecha: estado del modelo seleccionado + usuario + ajustes + logout
    right = tk.Frame(topbar, bg=COLOR_BG_ELEVATED)
    right.pack(side="right", padx=24)

    status_chip = RoundedCard(right, width=196, height=30, radius=15, bg=COLOR_CARD, border=COLOR_CARD,
                              border_width=0, parent_bg=COLOR_BG_ELEVATED)
    status_chip.pack(side="left", padx=(0, 18))
    chip_row = tk.Frame(status_chip.inner, bg=COLOR_CARD)
    chip_row.pack(expand=True)
    chip_dot = tk.Label(chip_row, text="●", bg=COLOR_CARD, fg=COLOR_TEXT_TERTIARY, font=(FONT_FAMILY, 8))
    chip_dot.pack(side="left", padx=(0, 6))
    cloud_status_label = tk.Label(chip_row, text="Comprobando Cloud…", bg=COLOR_CARD,
                                  fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 9))
    cloud_status_label.pack(side="left")

    def render_cloud_status():
        """Estado del modelo SELECCIONADO (no siempre el binario)."""
        model_type = APP_CONFIG.get("model_type", "binary")
        state, _version = model_health(model_type)
        text, color = MODEL_STATUS_STYLE[state]
        chip_dot.config(fg=color)
        if state == "offline":
            cloud_status_label.config(text="Cloud sin conexión")
        else:
            cloud_status_label.config(text=f"{AI_MODEL_SHORT.get(model_type, model_type)} · {text}")

    _cloud_listeners.append(render_cloud_status)

    tk.Label(right, text=f"👤 {SESSION.get('user', '')}", bg=COLOR_BG_ELEVATED, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 10))

    RoundedButton(right, "⚙ Ajustes", command=show_settings, width=90, height=30,
                  bg=COLOR_BG_ELEVATED, hover_bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 9),
                  parent_bg=COLOR_BG_ELEVATED).pack(side="left", padx=(0, 4))

    def logout():
        if messagebox.askokcancel("Cerrar sesión", "¿Cerrar la sesión actual?"):
            SESSION["user"] = None
            root.destroy()
            start_app()

    RoundedButton(right, "⎋ Salir", command=logout, width=76, height=30,
                  bg=COLOR_BG_ELEVATED, hover_bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, 9),
                  parent_bg=COLOR_BG_ELEVATED).pack(side="left")

    sep = tk.Frame(root, bg=COLOR_SEPARATOR, height=1)
    sep.pack(side="top", fill="x")

    # Ping de salud a Cloud (en segundo plano, no bloquea la UI)
    def schedule_poll():
        refresh_cloud_health()
        try:
            root.after(30000, schedule_poll)  # cada 30 s
        except Exception:
            pass

    schedule_poll()

    # ---- Scrollable body ----
    body_wrap = tk.Frame(root, bg=COLOR_BG)
    body_wrap.pack(side="top", fill="both", expand=True)

    canvas = tk.Canvas(body_wrap, bg=COLOR_BG, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(body_wrap, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas, bg=COLOR_BG)
    window_id = canvas.create_window(0, 0, window=frame, anchor="n")

    def update_scrollregion(event=None):
        # Región anclada en x=0 y con el ancho del canvas: con bbox("all") Tk
        # alineaba la vista al borde del contenido y la columna quedaba descentrada.
        canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), frame.winfo_reqheight()))

    def center_frame(event=None):
        canvas_width = canvas.winfo_width()
        canvas.coords(window_id, canvas_width // 2, 0)
        update_scrollregion()

    canvas.bind("<Configure>", center_frame)
    frame.bind("<Configure>", update_scrollregion)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    content = tk.Frame(frame, bg=COLOR_BG)
    content.pack(pady=(52, 60))

    ttk.Label(content, text="Brain MRI Tumor Analysis", style="Title.TLabel").pack()
    ttk.Label(content, text="Upload a scan to run an automated AI-assisted analysis",
              style="Subtitle.TLabel").pack(pady=(8, 32))

    # Aviso clínico: discreto, mismo lenguaje que el resto de tarjetas
    disc_card = RoundedCard(content, width=CONTENT_WIDTH, height=68, radius=16, bg=COLOR_BG_ELEVATED)
    disc_card.pack(pady=(0, 20))
    disc_row = tk.Frame(disc_card.inner, bg=COLOR_BG_ELEVATED)
    disc_row.pack(expand=True, fill="x", padx=22)
    tk.Label(disc_row, text="ⓘ", bg=COLOR_BG_ELEVATED, fg=COLOR_WARNING,
             font=(FONT_FAMILY, 13)).pack(side="left", padx=(0, 14))
    tk.Label(disc_row, text=DISCLAIMER_TEXT, bg=COLOR_BG_ELEVATED, fg=COLOR_TEXT_SECONDARY,
             font=(FONT_FAMILY, 9), wraplength=CONTENT_WIDTH - 90, justify="left").pack(side="left")

    def on_model_change(_value):
        render_cloud_status()   # al instante, con el último /health conocido
        refresh_cloud_health()  # y confirmación fresca en segundo plano

    build_model_selector(content, width=CONTENT_WIDTH, on_change=on_model_change)

    # ---- Upload card (estados: idle / selected / analyzing / done / error) ----
    upload_card = RoundedCard(content, width=CONTENT_WIDTH, height=486, radius=24)
    upload_card.pack()

    icon = tk.Canvas(upload_card.inner, width=60, height=60, bg=COLOR_CARD, highlightthickness=0)
    icon.pack(pady=(36, 0))
    icon_bg = icon.create_oval(2, 2, 58, 58, fill=COLOR_ACCENT_SOFT, outline="")
    icon_glyph = icon.create_text(30, 30, text="↑", fill=COLOR_ACCENT, font=(FONT_FAMILY, 20, "bold"))

    upload_title = tk.Label(upload_card.inner, text="", bg=COLOR_CARD, fg=COLOR_TEXT,
                            font=(FONT_FAMILY, 14, "bold"), wraplength=CONTENT_WIDTH - 80)
    upload_title.pack(pady=(18, 4))

    status_label = tk.Label(upload_card.inner, text="", bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY,
                            font=(FONT_FAMILY, 10), wraplength=CONTENT_WIDTH - 80, justify="center",
                            height=2)
    status_label.pack(pady=(0, 18))

    preview_w, preview_h = 260, 150
    preview_card = RoundedCard(upload_card.inner, width=preview_w, height=preview_h, radius=16,
                               bg=COLOR_CARD_ALT, border=COLOR_CARD_ALT, parent_bg=COLOR_CARD)
    preview_card.pack()
    image_display = tk.Label(preview_card.inner, bg=COLOR_CARD_ALT, fg=COLOR_TEXT_TERTIARY,
                             font=(FONT_FAMILY, 9), justify="center")
    image_display.pack(expand=True)

    # Hueco de altura fija para que la barra no desplace el resto al aparecer
    progress_holder = tk.Frame(upload_card.inner, bg=COLOR_CARD, height=26)
    progress_holder.pack(fill="x")
    progress_holder.grid_propagate(False)
    progress_holder.grid_columnconfigure(0, weight=1)
    progress_holder.grid_rowconfigure(0, weight=1)
    progress = ttk.Progressbar(progress_holder, mode="indeterminate", length=preview_w,
                               style="Loading.Horizontal.TProgressbar")
    progress.grid(row=0, column=0)
    progress.grid_remove()

    select_button = RoundedButton(upload_card.inner, "", command=select_image,
                                  width=260, height=46, parent_bg=COLOR_CARD)
    select_button.pack(pady=(4, 0))

    UPLOAD_UI.update(icon=icon, icon_bg=icon_bg, icon_glyph=icon_glyph, title=upload_title,
                     button=select_button, preview_size=(preview_w - 20, preview_h - 20), file=None)
    set_upload_state("idle")

    root.mainloop()


# ======================================================
# START APPLICATION
# ======================================================

if __name__ == "__main__":
    start_app()