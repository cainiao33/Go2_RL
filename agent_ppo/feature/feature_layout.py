# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Unified observation protocol layouts for Foundation-v2.1.
"""


class BasePolicyLayout:
    PROPRIO = slice(0, 45)
    BASE_ANG_VEL = slice(0, 3)
    PROJECTED_GRAVITY = slice(3, 6)
    VELOCITY_COMMANDS = slice(6, 9)
    JOINT_POS_REL = slice(9, 21)
    JOINT_VEL_REL = slice(21, 33)
    LAST_ACTION = slice(33, 45)
    HEIGHT_SCAN = slice(45, 301)
    DIM = 301


class BaseCriticLayout:
    CRITIC_PROPRIO = slice(0, 60)
    BASE_LIN_VEL = slice(0, 3)
    BASE_ANG_VEL = slice(3, 6)
    PROJECTED_GRAVITY = slice(6, 9)
    VELOCITY_COMMANDS = slice(9, 12)
    JOINT_POS_REL = slice(12, 24)
    JOINT_VEL_REL = slice(24, 36)
    JOINT_EFFORT = slice(36, 48)
    LAST_ACTION = slice(48, 60)
    HEIGHT_SCAN = slice(60, 316)
    DIM = 316


class PolicyObsLayout:
    BASE = slice(0, 301)
    TERRAIN = slice(301, 317)
    HISTORY = slice(317, 389)
    GOAL = slice(389, 405)
    NAV = slice(405, 489)
    SCORE = slice(489, 496)
    RESERVED = slice(496, 512)
    DIM = 512


class CriticObsLayout:
    BASE = slice(0, 316)
    TERRAIN = slice(316, 332)
    HISTORY = slice(332, 404)
    GOAL = slice(404, 420)
    NAV = slice(420, 504)
    SCORE = slice(504, 511)
    PRIVILEGED = slice(511, 520)
    RESERVED = slice(520, 528)
    DIM = 528


class GoalFeatureLayout:
    GOAL_POS_VALID = 0
    ROBOT_POS_VALID = 1
    GOAL_DIST_VALID = 2
    GOAL_YAW_VALID = 3
    ROBOT_YAW_VALID = 4

    DX_WORLD_NORM = 5
    DY_WORLD_NORM = 6
    DIST_NORM = 7

    DX_BODY_NORM = 8
    DY_BODY_NORM = 9

    HEADING_SIN = 10
    HEADING_COS = 11

    GOAL_YAW_SIN = 12
    GOAL_YAW_COS = 13

    YAW_ERROR_SIN = 14
    YAW_ERROR_COS = 15

    DIM = 16


class NavFeatureLayout:
    # P7-2F ABI note: clearance fields are normalized clipped meter distance
    # (clipped_m / max_clearance_m). Raw meter values are never stored here.
    NAV_VALID = 0
    NAV_TENSOR_AVAILABLE = 1
    NAV_OBJECT_EXISTS = 2
    NAV_SOURCE_ID_NORM = 3

    FRONT_CLEARANCE_NORM = 4
    LEFT_CLEARANCE_NORM = 5
    RIGHT_CLEARANCE_NORM = 6
    MIN_CLEARANCE_NORM = 7
    MEAN_CLEARANCE_NORM = 8

    BLOCKED_FRONT_FLAG = 9
    WALL_CLOSE_FLAG = 10
    # DEADEND_SCORE is canonical; deadend flag is derived as
    # DEADEND_SCORE >= p7_2f_deadend_score_threshold to avoid adding obs bits.
    DEADEND_SCORE = 11

    BEST_FREE_DIR_SIN = 12
    BEST_FREE_DIR_COS = 13
    BEST_FREE_CLEARANCE_NORM = 14

    SECTOR_CLEARANCE_START = 15
    SECTOR_BLOCKED_START = 31
    SECTOR_VARIANCE_START = 47
    RESERVED_START = 63

    DIM = 84


class HistoryFeatureLayout:
    DIM = 72
    WINDOW = 8
    CHANNELS = 9

    ACTION_ABS_MEAN = slice(0, 8)
    ACTION_RATE_MEAN = slice(8, 16)
    POSTURE_ERROR_NORM = slice(16, 24)
    BASE_ANG_VEL_ABS_NORM = slice(24, 32)
    JOINT_VEL_ABS_MEAN_NORM = slice(32, 40)
    HEIGHT_SCAN_RISK_NORM = slice(40, 48)
    TASK_PROGRESS_STEP_NORM = slice(48, 56)
    FRONT_CLEARANCE_NORM = slice(56, 64)
    RISK_SCORE = slice(64, 72)
    COMMON = slice(0, 48)
    TRACK_ONLY = slice(48, 72)


class AuxTargetLayout:
    # Canonical v2 names
    SUCCESS = 0
    FAILURE = 1
    TIMEOUT = 2
    PROGRESS_SCORE = 3
    FINAL_GOAL_DIST_NORM = 4
    TIME_SCORE = 5
    ENERGY_SCORE = 6
    POSTURE_SCORE = 7
    STUCK_SCORE = 8
    WALL_CLOSE_RATIO = 9
    TERRAIN_LEVEL_NORM = 10
    RESERVED_11 = 11
    DIM = 12

    # Backward-compatible aliases
    COMPLETION = SUCCESS
    FINAL_PROGRESS = PROGRESS_SCORE
    MIN_GOAL_DIST_NORM = FINAL_GOAL_DIST_NORM
    NORMALIZED_EPISODE_LENGTH = TIME_SCORE
    AVG_ENERGY_PROXY = ENERGY_SCORE
    AVG_POSTURE_ERROR = POSTURE_SCORE
    MAX_STUCK_NORM = STUCK_SCORE
    RESERVED_10 = TERRAIN_LEVEL_NORM

    CANONICAL_NAMES = (
        "SUCCESS",
        "FAILURE",
        "TIMEOUT",
        "PROGRESS_SCORE",
        "FINAL_GOAL_DIST_NORM",
        "TIME_SCORE",
        "ENERGY_SCORE",
        "POSTURE_SCORE",
        "STUCK_SCORE",
        "WALL_CLOSE_RATIO",
        "TERRAIN_LEVEL_NORM",
        "RESERVED_11",
    )


def assert_aux_layout_valid():
    indices = [getattr(AuxTargetLayout, name) for name in AuxTargetLayout.CANONICAL_NAMES]
    assert len(indices) == AuxTargetLayout.DIM
    assert sorted(indices) == list(range(AuxTargetLayout.DIM))


def assert_policy_obs_shape(obs):
    assert obs.shape[-1] == PolicyObsLayout.DIM


def assert_critic_obs_shape(obs):
    assert obs.shape[-1] == CriticObsLayout.DIM


def assert_aux_shape(x):
    assert x.shape[-1] == AuxTargetLayout.DIM
