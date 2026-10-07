"""
Image-quality and page-geometry layer that sits in front of OCR.

Classic OpenCV / NumPy only (no models), in four parts:

    assess_quality()     blur (variance of the Laplacian), glare blobs, exposure
                         -> QualityReport.hint(): "Hold still", "Glare on the right..."
    find_page_quad()     page outline (contour + approxPolyDP)
    warp_page()          homography to a flat, top-down page
    estimate_dewarp()    curved lines (a book bending into its spine) modelled as a
                         vertical shift per column; dewarp() flattens them and
                         unwarp_box() maps OCR boxes back onto the camera frame
    enhance_tone()       shadow removal + CLAHE, only applied when the quality is low
    binarize()           adaptive threshold (a last-resort variant for hard cases)

Optional .env settings:
    VISION_ENHANCE = 0 turns the whole layer off (default on)
    BLUR_MIN       = sharpness below this is "blurry" (default 60, see assess_quality)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

ENABLED = os.getenv("VISION_ENHANCE", "1") != "0"
BLUR_MIN = float(os.getenv("BLUR_MIN", "60"))
GLARE_MIN_FRACTION = 0.01      # a saturated blob must cover this much of the image
DARK_MEAN = 50                 # mean brightness (0-255) below this is "too dark"
BRIGHT_MEAN = 205              # ... above this is "too bright"
_ANALYSIS_WIDTH = 640          # quality is measured at this width so it doesn't depend on camera size


# =====================================================================
# 1. Quality scoring
# =====================================================================
@dataclass
class QualityReport:
    sharpness: float = 0.0                       # variance of the Laplacian (higher = sharper)
    brightness: float = 0.0                      # mean grey level, 0-255
    contrast: float = 0.0                        # std of grey levels
    clipped_dark: float = 0.0                    # fraction of near-black pixels
    clipped_bright: float = 0.0                  # fraction of near-white pixels
    glare_fraction: float = 0.0                  # fraction of the image inside glare blobs
    glare_at: Optional[Tuple[float, float]] = None   # centre of the glare, 0-1 (x from the left, y from the top)
    problems: List[str] = field(default_factory=list)   # subset of: dark, bright, blurry, glare, flat

    @property
    def ok(self) -> bool:
        return not self.problems

    def hint(self) -> str:
        """One short spoken instruction for the most important problem ("" when fine)."""
        if "dark" in self.problems:
            return "Low light"
        if "glare" in self.problems and self.glare_at is not None:
            gx, gy = self.glare_at
            side = "on the right" if gx > 0.62 else "on the left" if gx < 0.38 else \
                   "at the bottom" if gy > 0.62 else "at the top" if gy < 0.38 else "in the middle"
            return f"Glare {side}. Tilt the page"
        if "blurry" in self.problems:
            return "Hold still, the image is blurry"
        if "bright" in self.problems:
            return "Too bright. Move away from the light"
        return ""

    def summary(self) -> str:
        return (f"sharp {self.sharpness:.0f}  light {self.brightness:.0f}  "
                f"glare {self.glare_fraction * 100:.1f}%" + (f"  [{', '.join(self.problems)}]" if self.problems else ""))


def _analysis_gray(image: np.ndarray, box: Optional[Box] = None) -> np.ndarray:
    """Grey image at a fixed width, optionally cropped to the text box."""
    if box is not None:
        x1, y1, x2, y2 = box
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size and min(crop.shape[:2]) >= 32:
            image = crop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = _ANALYSIS_WIDTH / gray.shape[1]
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def assess_quality(image: np.ndarray, text_box: Optional[Box] = None) -> QualityReport:
    """
    Blur, glare and exposure of a BGR frame. Pass the text box (when known) so
    sharpness is measured where the text is, not on the desk around it.

    Sharpness is the variance of the Laplacian: sharp print has strong
    edges (high variance), blur removes them. It also depends on how much text
    there is, so BLUR_MIN (default 60) may need adjusting for your camera.
    """
    full = _analysis_gray(image)
    text = _analysis_gray(image, text_box)
    report = QualityReport(
        sharpness=float(cv2.Laplacian(text, cv2.CV_64F).var()),
        brightness=float(full.mean()),
        contrast=float(full.std()),
        clipped_dark=float((full < 15).mean()),
        clipped_bright=float((full > 245).mean()),
    )

    # Glare: large blobs of fully saturated pixels (paper at 250+ is not glare unless
    # it forms a compact, saturated patch, so ignore small specks with an opening)
    saturated = (full >= 250).astype(np.uint8)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(saturated)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        report.glare_fraction = float(saturated.sum()) / saturated.size
        cx, cy = centroids[biggest]
        report.glare_at = (cx / saturated.shape[1], cy / saturated.shape[0])

    if report.brightness < DARK_MEAN:
        report.problems.append("dark")
    elif report.brightness > BRIGHT_MEAN and report.clipped_bright > 0.4:
        report.problems.append("bright")
    if report.glare_fraction >= GLARE_MIN_FRACTION:
        report.problems.append("glare")
    if report.sharpness < BLUR_MIN:
        report.problems.append("blurry")
    if report.contrast < 25:
        report.problems.append("flat")
    return report


# =====================================================================
# 2. Page detection and perspective correction
# =====================================================================
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_page_quad(frame: np.ndarray, min_area: float = 0.25) -> Optional[np.ndarray]:
    """
    The page outline as 4 corners (top-left, top-right, bottom-right, bottom-left)
    in frame pixels, or None when there is no clear page: paper is the bright
    region, so Otsu threshold, close the text holes, take the biggest contour and
    simplify it with approxPolyDP.
    """
    h, w = frame.shape[:2]
    scale = 480 / max(h, w)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    size = max(3, int(min(small.shape[:2]) * 0.04)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area * small.shape[0] * small.shape[1]:
        return None

    hull = cv2.convexHull(biggest)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        return None
    quad = _order_corners(quad)
    # The hull of a page must not be wildly non-rectangular (rules out a table edge, a hand ...)
    if cv2.contourArea(quad) < 0.8 * cv2.contourArea(hull):
        return None
    quad = quad / min(scale, 1.0)
    return quad


def _needs_warp(quad: np.ndarray, frame_shape) -> bool:
    """A page that is already square-on and fills the frame doesn't need warping."""
    h, w = frame_shape[:2]
    margin = 0.02 * max(h, w)
    on_edge = sum(1 for x, y in quad if x < margin or y < margin or x > w - margin or y > h - margin)
    if on_edge >= 2:
        return False        # the page runs out of the frame: warping would cut text off
    tl, tr, br, bl = quad
    angles = [abs(np.degrees(np.arctan2(*(b - a)[::-1]))) for a, b in ((tl, tr), (bl, br))]
    tilt = max(min(a, 180 - a) for a in angles)                        # rotation of the top / bottom edges
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    keystone = max(abs(top - bottom) / max(top, bottom), abs(left - right) / max(left, right))
    return tilt > 2.0 or keystone > 0.04


def warp_page(frame: np.ndarray, quad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Flatten the page with a homography. Returns (image, matrix) where matrix maps
    frame -> image (None when nothing was done: no clear page, or already flat).
    """
    quad = find_page_quad(frame) if quad is None else quad
    if quad is None or not _needs_warp(quad, frame.shape):
        return frame, None
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 100 or height < 100:
        return frame, None
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE), matrix


def unproject_box(box: Box, matrix: Optional[np.ndarray]) -> Box:
    """Map a box from the warped image back to the original frame."""
    if matrix is None:
        return box
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    back = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))[0]
    return (int(back[:, 0].min()), int(back[:, 1].min()), int(back[:, 0].max()), int(back[:, 1].max()))


# =====================================================================
# 3. Curved-page dewarping
# =====================================================================
@dataclass
class DewarpModel:
    """shift(x) = how far the text lines have moved down at column x, relative to a flat part of the page."""
    coeffs: np.ndarray
    width: int

    def shift(self, x) -> np.ndarray:
        return np.polyval(self.coeffs, np.asarray(x, dtype=np.float64))


def _ink(gray: np.ndarray) -> np.ndarray:
    """Dark print on a bright page -> float 0/1 mask."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(15, (min(gray.shape) // 20) | 1)
    return (cv2.adaptiveThreshold(blurred, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
            ).astype(np.float32)


def _line_pitch(profile: np.ndarray) -> int:
    """Distance between text lines: the first strong peak of the profile's autocorrelation."""
    p = profile - profile.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    if ac[0] <= 0:
        return 0
    ac = ac / ac[0]
    lo, hi = 6, min(len(p) // 3, 200)
    if hi <= lo + 2:
        return 0
    window = ac[lo:hi]
    peaks = [i for i in range(1, len(window) - 1) if window[i] > window[i - 1] and window[i] >= window[i + 1] and window[i] > 0.15]
    return lo + peaks[0] if peaks else 0


def _best_lag(profile: np.ndarray, ref: np.ndarray, max_lag: int) -> Tuple[int, float]:
    """The vertical shift (+ = lines lower than the reference) that best aligns `profile` to `ref`."""
    a, b = profile - profile.mean(), ref - ref.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    best, best_score = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            score = float(np.dot(a[lag:], b[:len(b) - lag]))
        else:
            score = float(np.dot(a[:lag], b[-lag:]))
        if score > best_score:
            best, best_score = lag, score
    return best, best_score / norm


def estimate_dewarp(image: np.ndarray, strips: int = 16, degree: int = 3,
                    min_gain: float = 1.08, min_shift: float = 2.0) -> Optional[DewarpModel]:
    """
    Fit how the text lines bend across the page. Cut the page into vertical strips,
    take each strip's row profile (ink per row) and slide it against its neighbour
    (starting from the flattest strip and moving outwards, so a shift never exceeds
    half a line pitch); the accumulated shifts are fitted with a polynomial.
    Returns None when the page is already flat or the fit doesn't make the lines
    any crisper (the line structure of the whole page must sharpen by min_gain).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, 900 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    h, w = small.shape
    ink = _ink(small)
    edges = np.linspace(0, w, strips + 1).astype(int)
    profiles, weights = [], []
    for x0, x1 in zip(edges[:-1], edges[1:]):
        p = ink[:, x0:x1].sum(axis=1)
        p = cv2.GaussianBlur(p.reshape(-1, 1), (1, 0), 1.5).ravel()
        profiles.append(p)
        weights.append(float(p.var()))
    weights = np.array(weights)
    if weights.max() < 1e-3 or (weights > 0.25 * weights.max()).sum() < 4:
        return None                                   # too little text to model

    ref_i = int(np.argmax(weights))
    pitch = _line_pitch(profiles[ref_i]) or max(8, h // 30)
    max_lag = max(2, int(pitch * 0.45))

    shifts = np.zeros(strips)
    valid = weights > 0.25 * weights.max()
    for direction in (1, -1):
        prev = ref_i
        for i in range(ref_i + direction, strips if direction > 0 else -1, direction):
            if not valid[i]:
                shifts[i] = shifts[prev]
                continue
            lag, _ = _best_lag(profiles[i], profiles[prev], max_lag)
            shifts[i] = shifts[prev] + lag
            prev = i
    centres = (edges[:-1] + edges[1:]) / 2.0
    if not valid.any() or np.abs(shifts[valid]).max() < min_shift * scale:
        return None

    coeffs = np.polyfit(centres[valid], shifts[valid], min(degree, int(valid.sum()) - 1), w=np.sqrt(weights[valid]))
    small_model = DewarpModel(coeffs, w)

    # Keep it only if the lines really got crisper
    if _profile_sharpness(_apply_shift(small, small_model)) < min_gain * _profile_sharpness(small):
        return None
    # The fit lives in the downscaled image; re-express it in full-size pixels
    xs = np.linspace(0, gray.shape[1], 64)
    return DewarpModel(np.polyfit(xs, small_model.shift(xs * scale) / scale, degree), gray.shape[1])


def _profile_sharpness(gray: np.ndarray) -> float:
    """How crisply text lines stand out: variance of the ink-per-row profile (higher = straighter lines)."""
    profile = _ink(gray).sum(axis=1)
    return float(profile.var()) + 1e-9


def _apply_shift(image: np.ndarray, model: DewarpModel) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    map_x = np.tile(xs, (h, 1))
    map_y = np.arange(h, dtype=np.float32).reshape(-1, 1) + model.shift(xs).astype(np.float32).reshape(1, -1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def dewarp(image: np.ndarray, model: Optional[DewarpModel]) -> np.ndarray:
    """Straighten the text lines: dst(x, y) = src(x, y + shift(x))."""
    return image if model is None else _apply_shift(image, model)


def unwarp_box(box: Box, model: Optional[DewarpModel]) -> Box:
    """Map a box found on the dewarped image back onto the original frame."""
    if model is None:
        return box
    x1, y1, x2, y2 = box
    s1, s2 = float(model.shift(x1)), float(model.shift(x2))
    return (x1, int(round(y1 + min(s1, s2))), x2, int(round(y2 + max(s1, s2))))


# =====================================================================
# 4. Adaptive preprocessing
# =====================================================================
def enhance_tone(image: np.ndarray, clahe_clip: float = 2.0) -> np.ndarray:
    """
    Shadow removal + CLAHE, geometry unchanged (so OCR boxes stay valid).
    The background (paper) brightness is estimated with a dilate + large blur at
    low resolution and divided out; CLAHE then evens the local contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    small = cv2.resize(light, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = max(3, int(min(small.shape[:2]) * 0.06)) | 1
    background = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    background = cv2.GaussianBlur(background, (0, 0), size)
    background = cv2.resize(background, (light.shape[1], light.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(light, np.maximum(background, 1), scale=245)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(flat)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive (local) threshold: black print on white, robust to uneven light."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    block = max(15, (min(gray.shape) // 25) | 1)
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# =====================================================================
# Pipelines used by the app
# =====================================================================
def capture_candidates(frame: np.ndarray, report: Optional[QualityReport] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Images to try reading for a single item (label, page, ...): the untouched frame
    first, then, only when the page is skewed or the quality is poor, a flattened
    (homography) and / or tone-corrected version. The caller keeps the best read.
    """
    candidates = [("original", frame)]
    if not ENABLED:
        return candidates
    report = report or assess_quality(frame)
    warped, matrix = warp_page(frame)
    if matrix is not None:
        candidates.append(("flattened", warped))
    if not report.ok:
        base = warped
        candidates.append(("tone" if matrix is None else "flattened+tone", enhance_tone(base)))
        if "blurry" not in report.problems:
            candidates.append(("binarized", binarize(base)))
    return candidates


def prepare_page(page: np.ndarray, report: Optional[QualityReport] = None) -> Tuple[np.ndarray, Optional[DewarpModel]]:
    """
    One book page for OCR with geometry kept invertible (the reading highlight
    needs boxes in camera coordinates): dewarp curved lines, and tone-correct
    only when the quality is poor. Returns (image, model); unwarp_box() maps boxes back.
    """
    if not ENABLED:
        return page, None
    model = estimate_dewarp(page)
    image = dewarp(page, model)
    if not (report or assess_quality(page)).ok:
        image = enhance_tone(image)
    return image, model
