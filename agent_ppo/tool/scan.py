#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foot / body_names / contact sensor static scanner.

Purpose:
- Scan official/project code for accurate Go2 foot/body keyword strings.
- Pure Python only.
- No torch import.
- No environment creation.
- No training.

How to use in web IDE:
1. Save as:
   /data/projects/legged_robot_competition_26/agent_ppo/tool/foot_keyword_static_scan.py
2. Click Run / 运行.
3. Open:
   /data/projects/legged_robot_competition_26/foot_keyword_static_scan.log
4. Send back sections:
   [BEST_CANDIDATE_STRINGS]
   [CLASSIFIED_FOOT_ALIASES]
   [HIGH_VALUE_CONTEXT]
"""

import os
import re
import sys
import site
from pathlib import Path
from collections import defaultdict, Counter


# =========================
# Basic paths
# =========================

DEFAULT_PROJECT_ROOT = Path("/data/projects/legged_robot_competition_26")
PROJECT_ROOT = DEFAULT_PROJECT_ROOT if DEFAULT_PROJECT_ROOT.exists() else Path.cwd()
LOG_PATH = PROJECT_ROOT / "foot_keyword_static_scan.log"

MAX_FILE_SIZE = 3_000_000
MAX_RESULTS = 360
MAX_CANDIDATE_STRINGS = 260
MAX_CONTEXT_PER_FILE = 8


# =========================
# Scan config
# =========================

FILE_SUFFIXES = {
    ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".md",
    ".xml", ".cfg", ".ini", ".usd", ".usda",
}

BANNED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "outputs", "output", "logs", "log", "wandb",
    ".mypy_cache", ".pytest_cache", ".cache",
    "dist", "build",
}

# For site-packages or big SDK dirs, only descend into likely relevant dirs.
RELEVANT_PATH_HINTS = (
    "go2", "unitree", "legged", "quadruped", "isaac", "lab",
    "locomotion", "robot", "robots", "rsl", "contact", "terrain",
    "agent_ppo", "env", "mdp",
)

# Strong search keywords.
KEYWORDS = [
    # Robot / asset names
    "go2", "unitree", "UNITREE_GO2", "Go2",

    # Foot/body naming
    "foot", "feet", "toe", "paw",
    "front_left", "left_front", "front_right", "right_front",
    "rear_left", "left_rear", "rear_right", "right_rear",
    "hind_left", "left_hind", "hind_right", "right_hind",
    "FL", "FR", "RL", "RR", "lf", "rf", "lh", "rh",

    # Isaac Lab / contact sensor related
    "ContactSensorCfg", "contact_sensor", "contact_forces",
    "net_forces_w", "net_forces_w_history",
    "current_contact_time", "body_names", "body_ids",
    "find_bodies", "find_joints", "joint_names",

    # Rewards / gait / air time
    "feet_air_time", "air_time", "feet_air_time_positive_biped",
    "undesired_contacts", "desired_contacts",
    "gait", "trot", "pace", "bound", "bunny",

    # Our target functions
    "_get_foot_sensor_cfg", "_get_foot_contacts",
    "_get_foot_order_indices", "foot_order",
]


LINE_PATTERN = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)

QUOTED_STRING_RE = re.compile(
    r"""(?P<quote>["'])(?P<value>(?:\\.|(?!\1).)*?)(?P=quote)"""
)

BODY_NAMES_ASSIGN_RE = re.compile(
    r"(body_names|joint_names|prim_path|name|asset_name|sensor_name)\s*=\s*(.+)",
    re.IGNORECASE,
)


# =========================
# Candidate classification
# =========================

FOOT_CLASS_PATTERNS = {
    "FL": [
        "front_left", "left_front", "front-l", "left-f", "front.l", "left.f",
        "front_l", "left_f", "fl", "lf", "foot_fl", "fl_foot", "lf_foot",
        "front_left_foot", "left_front_foot", "fl_foot_link", "lf_foot_link",
    ],
    "FR": [
        "front_right", "right_front", "front-r", "right-f", "front.r", "right.f",
        "front_r", "right_f", "fr", "rf", "foot_fr", "fr_foot", "rf_foot",
        "front_right_foot", "right_front_foot", "fr_foot_link", "rf_foot_link",
    ],
    "RL": [
        "rear_left", "left_rear", "hind_left", "left_hind",
        "rear-l", "hind-l", "left-r", "left-h",
        "rear_l", "hind_l", "left_r", "left_h",
        "rl", "lh", "foot_rl", "rl_foot", "lh_foot",
        "rear_left_foot", "left_rear_foot", "hind_left_foot",
        "rl_foot_link", "lh_foot_link",
    ],
    "RR": [
        "rear_right", "right_rear", "hind_right", "right_hind",
        "rear-r", "hind-r", "right-r", "right-h",
        "rear_r", "hind_r", "right_r", "right_h",
        "rr", "rh", "foot_rr", "rr_foot", "rh_foot",
        "rear_right_foot", "right_rear_foot", "hind_right_foot",
        "rr_foot_link", "rh_foot_link",
    ],
}


def normalize_token(s: str) -> str:
    s = s.strip().strip("\"'`")
    s = s.replace("\\", "/")
    s = s.lower()
    return s


def split_identifier_tokens(s: str):
    s = normalize_token(s)
    return [x for x in re.split(r"[^a-z0-9]+", s) if x]


def classify_candidate(s: str):
    """
    Classify a string into FL/FR/RL/RR if it strongly looks like a foot/body name.
    Avoid over-trusting short substring hits.
    """
    low = normalize_token(s)
    tokens = split_identifier_tokens(low)
    token_set = set(tokens)

    classes = []

    for cls, pats in FOOT_CLASS_PATTERNS.items():
        for p in pats:
            p_low = normalize_token(p)
            p_tokens = split_identifier_tokens(p_low)

            # Exact whole string.
            if low == p_low:
                classes.append(cls)
                break

            # Exact token match for short aliases.
            if len(p_low) <= 2 and p_low in token_set:
                classes.append(cls)
                break

            # Multi-token contained in sequence.
            if len(p_tokens) >= 2:
                joined = "_".join(tokens)
                if "_".join(p_tokens) in joined:
                    classes.append(cls)
                    break

            # Common suffix / prefix.
            if low.endswith("_" + p_low) or low.startswith(p_low + "_"):
                classes.append(cls)
                break

    return sorted(set(classes))


def is_foot_like_string(s: str) -> bool:
    low = normalize_token(s)
    if len(low) < 2:
        return False

    if any(x in low for x in [
        "foot", "feet", "toe", "paw",
        "front_left", "left_front", "front_right", "right_front",
        "rear_left", "left_rear", "rear_right", "right_rear",
        "hind_left", "left_hind", "hind_right", "right_hind",
    ]):
        return True

    # Short aliases only count if tokenized.
    tokens = set(split_identifier_tokens(low))
    return bool(tokens & {"fl", "fr", "rl", "rr", "lf", "rf", "lh", "rh"})


# =========================
# Scoring
# =========================

def score_line(line: str, path: Path) -> int:
    low = line.lower()
    p_low = str(path).lower()
    score = 0

    # Path context.
    for k in ("go2", "unitree", "robot", "asset", "mdp", "locomotion", "agent_ppo"):
        if k in p_low:
            score += 2

    # Highest-value exact config/code signals.
    for k in [
        "contactsensorcfg", "body_names", "body_ids",
        "find_bodies", "net_forces_w", "current_contact_time",
        "_get_foot_sensor_cfg", "_get_foot_order_indices",
        "feet_air_time", "undesired_contacts",
    ]:
        if k in low:
            score += 8

    # Robot/asset signals.
    for k in ["unitree_go2", "go2", "unitree"]:
        if k in low:
            score += 5

    # Foot/body signals.
    for k in [
        "front_left", "left_front", "front_right", "right_front",
        "rear_left", "left_rear", "rear_right", "right_rear",
        "hind_left", "left_hind", "hind_right", "right_hind",
    ]:
        if k in low:
            score += 8

    for k in ["foot", "feet", "toe", "paw"]:
        if k in low:
            score += 5

    # Joint order signals.
    for k in ["hip", "thigh", "calf"]:
        if k in low:
            score += 2

    # Gait/reward related.
    for k in ["trot", "pace", "gait", "air_time", "contact"]:
        if k in low:
            score += 2

    # De-prioritize monitor-only aggregate names.
    for k in [
        "completed_count", "timeout_count", "abnormal_count",
        "reward_", "monitor", "logger", "print(",
    ]:
        if k in low:
            score -= 2

    return score


def score_candidate_string(s: str, path: Path, line: str) -> int:
    low = normalize_token(s)
    p_low = str(path).lower()
    line_low = line.lower()

    if not is_foot_like_string(low):
        return -999

    score = 0

    if "go2" in p_low or "unitree" in p_low:
        score += 6

    if "body_names" in line_low or "contactsensorcfg" in line_low:
        score += 10

    if "find_bodies" in line_low or "body_ids" in line_low:
        score += 8

    if "foot" in low or "feet" in low or "toe" in low:
        score += 6

    if classify_candidate(low):
        score += 8

    if any(x in low for x in [
        "front_left", "left_front", "front_right", "right_front",
        "rear_left", "left_rear", "rear_right", "right_rear",
        "hind_left", "left_hind", "hind_right", "right_hind",
    ]):
        score += 8

    # Regex body patterns like .*_foot are valuable.
    if ".*" in low or "regex" in line_low:
        score += 3

    # Too generic is less useful.
    if low in {"foot", "feet", "toe", "paw", "contact", "body"}:
        score -= 5

    return score


# =========================
# Root discovery
# =========================

def add_root_if_exists(roots, p):
    p = Path(p)
    if p.exists() and p.is_dir():
        roots.append(p)


def discover_scan_roots():
    roots = []

    add_root_if_exists(roots, PROJECT_ROOT)
    add_root_if_exists(roots, PROJECT_ROOT / "agent_ppo")
    add_root_if_exists(roots, PROJECT_ROOT / "exts")
    add_root_if_exists(roots, PROJECT_ROOT / "source")
    add_root_if_exists(roots, PROJECT_ROOT / "env")
    add_root_if_exists(roots, PROJECT_ROOT / "envs")
    add_root_if_exists(roots, PROJECT_ROOT / "assets")

    # Competition/project area.
    add_root_if_exists(roots, "/data/projects")

    # Common Isaac Lab locations.
    for p in [
        "/workspace/isaaclab",
        "/workspace/IsaacLab",
        "/isaac-sim",
        "/opt/isaaclab",
        "/opt/IsaacLab",
    ]:
        add_root_if_exists(roots, p)

    # Add only relevant site-packages subdirs, not the whole site-packages.
    try:
        sp_list = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        sp_list = []

    wanted_names = [
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "omni",
        "rsl_rl",
        "legged_gym",
    ]

    for sp in sp_list:
        sp_path = Path(sp)
        if not sp_path.exists():
            continue

        for name in wanted_names:
            add_root_if_exists(roots, sp_path / name)

    # Deduplicate.
    dedup = []
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key not in seen:
            seen.add(key)
            dedup.append(r)

    return dedup


def should_skip_dir(path: Path, root: Path) -> bool:
    parts = set(path.parts)
    if parts & BANNED_DIRS:
        return True

    # Avoid scanning huge irrelevant SDK dirs too deeply.
    p_low = str(path).lower()
    r_low = str(root).lower()

    # Project root should be scanned normally.
    if str(PROJECT_ROOT).lower() in p_low:
        return False

    # For non-project roots, keep only likely relevant path trees.
    if any(h in p_low for h in RELEVANT_PATH_HINTS):
        return False

    # Allow first two levels under the root, so we can reach relevant dirs.
    try:
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) <= 2:
            return False
    except Exception:
        pass

    return True


def iter_files(root: Path):
    if not root.exists():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)

        if should_skip_dir(dpath, root):
            dirnames[:] = []
            continue

        dirnames[:] = [
            d for d in dirnames
            if d not in BANNED_DIRS and not d.startswith(".")
        ]

        for name in filenames:
            p = dpath / name
            if p.suffix.lower() not in FILE_SUFFIXES:
                continue

            try:
                if p.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            yield p


# =========================
# Formatting
# =========================

def format_context(path: Path, lines, idx: int, span: int = 3) -> str:
    start = max(0, idx - span)
    end = min(len(lines), idx + span + 1)

    out = []
    out.append("\n" + "=" * 120)
    out.append(f"FILE: {path}")
    out.append(f"HIT_LINE: {idx + 1}")
    out.append("-" * 120)

    for i in range(start, end):
        marker = ">>" if i == idx else "  "
        out.append(f"{marker} L{i + 1}: {lines[i].rstrip()[:260]}")

    return "\n".join(out)


def extract_quoted_strings(line: str):
    out = []
    for m in QUOTED_STRING_RE.finditer(line):
        v = m.group("value")
        if v:
            out.append(v)
    return out


def extract_assignment_value(line: str):
    m = BODY_NAMES_ASSIGN_RE.search(line)
    if not m:
        return None
    return m.group(2).strip()[:260]


# =========================
# Main
# =========================

def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("[FootKeywordStaticScan] start")
    log(f"[FootKeywordStaticScan] python={sys.version.split()[0]}")
    log(f"[FootKeywordStaticScan] project_root={PROJECT_ROOT}")
    log(f"[FootKeywordStaticScan] log_path={LOG_PATH}")
    log("[FootKeywordStaticScan] pure python; no torch import; no env creation; no training")

    roots = discover_scan_roots()
    log("\n[SCAN_ROOTS]")
    for r in roots:
        log(f"  - {r}")

    hits = []
    candidate_strings = []
    assignment_hits = []
    scanned_files = 0
    matched_files = 0
    seen_files = set()
    contexts_per_file = Counter()

    for root in roots:
        for path in iter_files(root):
            try:
                real = str(path.resolve())
            except Exception:
                real = str(path)

            if real in seen_files:
                continue
            seen_files.add(real)
            scanned_files += 1

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not LINE_PATTERN.search(text):
                # Fast skip.
                continue

            matched_files += 1
            lines = text.splitlines()

            for idx, line in enumerate(lines):
                if not LINE_PATTERN.search(line):
                    continue

                score = score_line(line, path)
                if score <= 0:
                    continue

                hits.append((score, str(path), idx, line.strip()))

                # Extract quoted candidate strings.
                for s in extract_quoted_strings(line):
                    c_score = score_candidate_string(s, path, line)
                    if c_score > 0:
                        candidate_strings.append((c_score, s, str(path), idx, line.strip()))

                # Extract body_names/joint_names/etc assignment text.
                val = extract_assignment_value(line)
                if val:
                    assignment_hits.append((score, str(path), idx, val, line.strip()))

    hits.sort(key=lambda x: (-x[0], x[1], x[2]))
    candidate_strings.sort(key=lambda x: (-x[0], normalize_token(x[1]), x[2], x[3]))
    assignment_hits.sort(key=lambda x: (-x[0], x[1], x[2]))

    log(f"\n[FootKeywordStaticScan] scanned_files={scanned_files}")
    log(f"[FootKeywordStaticScan] matched_files={matched_files}")
    log(f"[FootKeywordStaticScan] total_hits={len(hits)}")
    log(f"[FootKeywordStaticScan] candidate_strings={len(candidate_strings)}")
    log(f"[FootKeywordStaticScan] assignment_hits={len(assignment_hits)}")

    # -------------------------
    # Candidate string summary
    # -------------------------
    log("\n" + "#" * 120)
    log("[BEST_CANDIDATE_STRINGS]")
    log("# Format: score | classes | string | file:line")
    log("# These are the most useful literal strings found near foot/body/contact code.")
    log("#" * 120)

    seen_candidate_keys = set()
    shown = 0

    for c_score, s, path_str, idx, line in candidate_strings:
        key = normalize_token(s)
        if key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(key)

        classes = classify_candidate(s)
        classes_text = ",".join(classes) if classes else "-"
        log(f"{c_score:>3} | {classes_text:<8} | {s!r} | {path_str}:{idx + 1}")
        shown += 1
        if shown >= MAX_CANDIDATE_STRINGS:
            break

    # -------------------------
    # Classified aliases
    # -------------------------
    classified = defaultdict(list)

    for c_score, s, path_str, idx, line in candidate_strings:
        classes = classify_candidate(s)
        if not classes:
            continue
        for cls in classes:
            classified[cls].append((c_score, s, path_str, idx, line))

    log("\n" + "#" * 120)
    log("[CLASSIFIED_FOOT_ALIASES]")
    log("# Candidate strings grouped by foot class.")
    log("# Use this section to refine _find((...)) aliases.")
    log("#" * 120)

    for cls in ["FL", "FR", "RL", "RR"]:
        log(f"\n[{cls}]")
        seen = set()
        count = 0
        for c_score, s, path_str, idx, line in sorted(
            classified.get(cls, []),
            key=lambda x: (-x[0], normalize_token(x[1]), x[2], x[3]),
        ):
            key = normalize_token(s)
            if key in seen:
                continue
            seen.add(key)
            log(f"  {c_score:>3} | {s!r} | {path_str}:{idx + 1}")
            count += 1
            if count >= 80:
                break

    # -------------------------
    # body_names / assignment contexts
    # -------------------------
    log("\n" + "#" * 120)
    log("[BODY_NAMES_ASSIGNMENTS]")
    log("# Lines that look like body_names / joint_names / prim_path / name assignment.")
    log("#" * 120)

    for n, (score, path_str, idx, val, line) in enumerate(assignment_hits[:180], 1):
        log(f"\n[ASSIGN {n}] score={score} file={path_str} line={idx + 1}")
        log(f"  value: {val}")
        log(f"  line : {line[:260]}")

    # -------------------------
    # High value contexts
    # -------------------------
    log("\n" + "#" * 120)
    log("[HIGH_VALUE_CONTEXT]")
    log("# Full nearby contexts for top hits.")
    log("# Prioritize ContactSensorCfg / body_names / find_bodies / feet_air_time / Unitree Go2.")
    log("#" * 120)

    shown = 0

    for score, path_str, idx, line in hits:
        path = Path(path_str)

        # Avoid too many contexts from same file.
        if contexts_per_file[path_str] >= MAX_CONTEXT_PER_FILE:
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        contexts_per_file[path_str] += 1
        shown += 1

        log(f"\n[HIT {shown}] score={score} file={path} line={idx + 1}")
        log(format_context(path, lines, idx))

        if shown >= MAX_RESULTS:
            break

    # -------------------------
    # Suggested _find block skeleton
    # -------------------------
    log("\n" + "#" * 120)
    log("[SUGGESTED_FIND_BLOCK_BASELINE]")
    log("# Baseline aliases. Replace/extend with strings proven by BEST_CANDIDATE_STRINGS.")
    log("# Do not use short aliases by loose substring; use exact/tokenized match first.")
    log("#" * 120)

    log(r'''
fl = _find((
    "fl", "lf",
    "f_l", "l_f",
    "front_left", "left_front",
    "front_l", "left_f",
    "foot_fl", "fl_foot",
    "lf_foot", "foot_lf",
    "front_left_foot", "left_front_foot",
    "fl_foot_link", "lf_foot_link",
    "front_left_foot_link", "left_front_foot_link",
    "fl_toe", "lf_toe",
))

fr = _find((
    "fr", "rf",
    "f_r", "r_f",
    "front_right", "right_front",
    "front_r", "right_f",
    "foot_fr", "fr_foot",
    "rf_foot", "foot_rf",
    "front_right_foot", "right_front_foot",
    "fr_foot_link", "rf_foot_link",
    "front_right_foot_link", "right_front_foot_link",
    "fr_toe", "rf_toe",
))

rl = _find((
    "rl", "lh",
    "r_l", "l_h",
    "rear_left", "left_rear",
    "hind_left", "left_hind",
    "rear_l", "hind_l", "left_r", "left_h",
    "foot_rl", "rl_foot",
    "lh_foot", "foot_lh",
    "rear_left_foot", "left_rear_foot",
    "hind_left_foot", "left_hind_foot",
    "rl_foot_link", "lh_foot_link",
    "rear_left_foot_link", "hind_left_foot_link",
    "rl_toe", "lh_toe",
))

rr = _find((
    "rr", "rh",
    "r_r", "r_h",
    "rear_right", "right_rear",
    "hind_right", "right_hind",
    "rear_r", "hind_r", "right_r", "right_h",
    "foot_rr", "rr_foot",
    "rh_foot", "foot_rh",
    "rear_right_foot", "right_rear_foot",
    "hind_right_foot", "right_hind_foot",
    "rr_foot_link", "rh_foot_link",
    "rear_right_foot_link", "hind_right_foot_link",
    "rr_toe", "rh_toe",
))
'''.strip())

    log("\n[FootKeywordStaticScan] done")
    log("[FootKeywordStaticScan] Send back:")
    log("  1. [BEST_CANDIDATE_STRINGS]")
    log("  2. [CLASSIFIED_FOOT_ALIASES]")
    log("  3. Top [HIGH_VALUE_CONTEXT] around ContactSensorCfg/body_names/find_bodies/feet_air_time")

    try:
        LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"[FootKeywordStaticScan] wrote log: {LOG_PATH}")
    except Exception as exc:
        print(f"[FootKeywordStaticScan] failed to write log: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()