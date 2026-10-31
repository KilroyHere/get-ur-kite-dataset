"""Constants that define the LEAP hand family. See DATA_GENERATION_PLAN.md (Stage A)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LEAP_DIR = THIRD_PARTY / "get_zero" / "get_zero" / "rl" / "assets" / "leap" / "leap_hand"
BASE_URDF = LEAP_DIR / "original_refactor.urdf"
GETZERO_URDF_DIR = LEAP_DIR / "generated" / "urdf"
MENAGERIE_XML = THIRD_PARTY / "mujoco_menagerie" / "leap_hand" / "right_hand.xml"
GET_ZERO_SHA = "279652da160823521e881952f67870b0d98674ce"
MENAGERIE_SHA = "b0655057a9df2abe39ea7fc989914ba6f248301c"

# ---------------------------------------------------------------- fingers
FINGERS = ("index", "middle", "ring", "thumb")
PALM = "palm_lower"
# Base-URDF joint names (LEAP motor IDs), proximal -> distal. Joint k moves FINGER_LINKS[f][k].
FINGER_JOINTS = {
    "index": ("1", "0", "2", "3"),
    "middle": ("5", "4", "6", "7"),
    "ring": ("9", "8", "10", "11"),
    "thumb": ("12", "13", "14", "15"),
}
FINGER_LINKS = {
    "index": ("mcp_joint", "pip", "dip", "fingertip"),
    "middle": ("mcp_joint_2", "pip_2", "dip_2", "fingertip_2"),
    "ring": ("mcp_joint_3", "pip_3", "dip_3", "fingertip_3"),
    "thumb": ("pip_4", "thumb_pip", "thumb_dip", "thumb_fingertip"),
}

# ---------------------------------------------------------------- topology
JOINT_COUNTS = (2, 3, 4)  # joints per present finger; 0 means the finger is absent
MIN_FINGERS = 2
# Fewer joints = lock joints from the fingertip inward at this angle (the hand's zero pose).
LOCK_ANGLE = 0.0

# ---------------------------------------------------------------- geometry
# Links whose length varies, with the sign of their long axis (link-frame y) pointing distally.
# These are the links GET-Zero also lengthens; the short motor-to-motor brackets
# (pip, pip_4, thumb_pip) stay fixed.
LONG_LINKS = {"mcp_joint": +1, "dip": -1, "fingertip": -1, "thumb_dip": +1, "thumb_fingertip": -1}
LENGTH_SCALE_RANGE = (0.8, 1.2)
# Only this band of the link (fractions of its length) is stretched, so motors and pads keep their shape.
STRETCH_BAND = (0.25, 0.75)

# Mount shifts move index/middle/ring along the palm's finger-spacing axis. The thumb's motor
# housing is fused into the palm mesh with no clear boundary, so the thumb mount stays fixed.
MOUNT_FINGERS = ("index", "middle", "ring")
MOUNT_AXIS = (0.0, 1.0, 0.0)  # palm frame
MOUNT_SHIFT_MAX = 0.004  # metres (current spacing is 45.5 mm)
# Palm-mesh region that moves with each finger: Menagerie's motor-housing box, grown by
# MOUNT_CORE_GROW, with a linear falloff over MOUNT_FALLOFF (>= MOUNT_SHIFT_MAX / 0.6 keeps the
# deformation's Jacobian determinant >= 0.4, so the palm never folds).
MOUNT_BOX = {"index": "palm_collision_1", "middle": "palm_collision_2", "ring": "palm_collision_3"}
MOUNT_CORE_GROW = 0.002
MOUNT_FALLOFF = 0.007

# ---------------------------------------------------------------- contact regions (KITE App. A)
REGIONS = ("palm", "index", "middle", "ring", "thumb")

# ---------------------------------------------------------------- surface template (KITE App. A/B)
TEMPLATE_SIZE = 4096
FPS_SIZE = 32
CANDIDATES_PER_MESH = 20000
VISIBILITY_RAYS = 32
VISIBILITY_MIN_ESCAPE = 8  # keep points where at least this many hemisphere rays escape the part

# ---------------------------------------------------------------- collision
MENAGERIE_BODY = {
    "palm_lower": "palm",
    "mcp_joint": "if_bs", "pip": "if_px", "dip": "if_md", "fingertip": "if_ds",
    "mcp_joint_2": "mf_bs", "pip_2": "mf_px", "dip_2": "mf_md", "fingertip_2": "mf_ds",
    "mcp_joint_3": "rf_bs", "pip_3": "rf_px", "dip_3": "rf_md", "fingertip_3": "rf_ds",
    "pip_4": "th_mp", "thumb_pip": "th_bs", "thumb_dip": "th_px", "thumb_fingertip": "th_ds",
}
PENETRATION_TOL = 1e-4  # metres; contacts shallower than this count as touching, not colliding
GATE_SAMPLES = 500
GATE_MIN_FREE_RATE = 0.05
GATE_MESH_FACES = 3000  # simplified meshes for the exact zero-pose check
