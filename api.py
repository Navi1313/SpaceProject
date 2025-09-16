# from fastapi import FastAPI, Query
# from pred import make_pred, model_path
# import numpy as np

# app = FastAPI()

# @app.get('/')
# def index_route():
#     return {"health": "ok"}

# @app.post('/predict')
# def prediction(temperature, luminosity, radius, abs_mag):
#     input_features = [[temperature, luminosity, radius, abs_mag]]
#     pred_class, probs, classes = make_pred(model_path, input_features)
    
  
#     # Convert numpy arrays to lists
#     return {
#          "Predicted_class": pred_class,
#     }
# api.py (only the relevant parts shown — keep your imports)
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pred import make_pred, model_path
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def to_py(obj):
    """Convert numpy objects into Python-native structures where possible."""
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            try:
                return np.ravel(obj).tolist()
            except Exception:
                return obj
    return obj


def extract_rows(probs):
    """
    Return a list-of-rows (each row is a list of floats).
    Handles np.array, nested lists, lists of arrays, etc.
    """
    probs = to_py(probs)

    if not isinstance(probs, list):
        try:
            return [[float(probs)]]
        except Exception:
            return []

    if all(isinstance(x, (int, float, np.floating, np.integer)) for x in probs):
        return [[float(x) for x in probs]]

    rows = []
    for item in probs:
        if isinstance(item, (int, float, np.floating, np.integer)):
            rows.append([float(item)])
            continue

        try:
            arr = np.ravel(item).tolist()
            numeric = []
            for v in arr:
                try:
                    numeric.append(float(v))
                except Exception:
                    pass
            if numeric:
                rows.append(numeric)
        except Exception:
            pass

    if rows:
        return rows

    try:
        flat = np.ravel(probs).tolist()
        return [[float(x) for x in flat if isinstance(x, (int, float, np.floating, np.integer))]]
    except Exception:
        return []


def extract_prediction(pred):
    """Return (predicted_star, predicted_color) from pred_class."""
    pred = to_py(pred)
    if isinstance(pred, str):
        return pred, "Not provided"
    if isinstance(pred, (list, tuple)):
        flat = []
        for p in np.ravel(pred).tolist():
            flat.append(str(p))
        if len(flat) >= 2:
            return flat[0], flat[1]
        if len(flat) == 1:
            return flat[0], "Not provided"
    return str(pred), "Not provided"


def normalize_row(arr):
    """Normalize a row so it sums to 1.0. Apply softmax if needed."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return arr
    s = np.sum(arr)
    if np.isfinite(s) and s > 0 and abs(s - 1.0) > 1e-6:
        return arr / s
    if np.isfinite(s) and abs(s - 1.0) <= 1e-6:
        return arr
    exps = np.exp(arr - np.max(arr))
    return exps / np.sum(exps)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
def prediction(
    request: Request,
    temperature: float = Form(...),
    luminosity: float = Form(...),
    radius: float = Form(...),
    abs_mag: float = Form(...)
):
    input_features = [[temperature, luminosity, radius, abs_mag]]
    pred_class, probs, classes = make_pred(model_path, input_features)

    # Normalize to Python structures
    probs_py = to_py(probs)
    classes_py = to_py(classes)

    star_results = []
    color_results = []

    # Case 1: single-output model
    if isinstance(classes_py, list) and classes_py and all(isinstance(x, str) for x in classes_py):
        rows = extract_rows(probs_py)
        if rows:
            row = normalize_row(rows[0])
            n = len(classes_py)
            if len(row) < n:
                row = np.concatenate([row, np.zeros(n - len(row))])
            else:
                row = row[:n]
            star_results = [(str(lbl), float(val)) for lbl, val in zip(classes_py, row)]

    # Case 2: multi-output model (class group + color group)
    else:
        class_groups = []
        if isinstance(classes_py, list):
            for g in classes_py:
                if isinstance(g, list) and all(isinstance(x, str) for x in g):
                    class_groups.append([str(x) for x in g])
                else:
                    try:
                        vals = np.ravel(g).tolist()
                        class_groups.append([str(x) for x in vals])
                    except Exception:
                        class_groups.append([str(g)])
        else:
            class_groups = [[str(classes_py)]]

        rows = extract_rows(probs_py)

        if len(rows) >= len(class_groups):
            for i, grp in enumerate(class_groups):
                row = normalize_row(rows[i])
                n = len(grp)
                if len(row) < n:
                    row = np.concatenate([row, np.zeros(n - len(row))])
                else:
                    row = row[:n]
                if i == 0:
                    star_results = [(str(lbl), float(val)) for lbl, val in zip(grp, row)]
                elif i == 1:
                    color_results = [(str(lbl), float(val)) for lbl, val in zip(grp, row)]
        elif len(rows) == 1 and len(class_groups) >= 2:
            flat = normalize_row(rows[0])
            idx = 0
            for i, grp in enumerate(class_groups):
                n = len(grp)
                chunk = flat[idx: idx + n]
                idx += n
                while len(chunk) < n:
                    chunk = np.concatenate([chunk, np.zeros(n - len(chunk))])
                if i == 0:
                    star_results = [(str(lbl), float(val)) for lbl, val in zip(grp, chunk)]
                elif i == 1:
                    color_results = [(str(lbl), float(val)) for lbl, val in zip(grp, chunk)]
        else:
            flat_all = []
            for r in rows:
                flat_all.extend(r)
            idx = 0
            for i, grp in enumerate(class_groups):
                n = len(grp)
                chunk = flat_all[idx: idx + n]
                idx += n
                while len(chunk) < n:
                    chunk.append(0.0)
                chunk = normalize_row(chunk)
                if i == 0:
                    star_results = [(str(lbl), float(val)) for lbl, val in zip(grp, chunk)]
                elif i == 1:
                    color_results = [(str(lbl), float(val)) for lbl, val in zip(grp, chunk)]

    # Extract predicted star and color
    predicted_star, predicted_color = extract_prediction(pred_class)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predicted_star": predicted_star,
            "predicted_color": predicted_color,
            "star_results": star_results,
            "color_results": color_results
        }
    )

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})
