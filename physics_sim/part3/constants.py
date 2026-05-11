from ..common import vec3

EPS = 1.0e-8
SAT_EPS = 1.0e-6
AXIS_CROSS_EPS = 1.0e-7
CONTACT_SLOP = 1.0e-4
MANIFOLD_KEEP_SLOP = 2.0e-3
FACE_AXIS_BIAS = 1.0e-5
MAX_CONTACT_POINTS = 4
MAX_BOX_VERTS = 8
MAX_FACE_VERTS = 4
MAX_CLIP_VERTS = 16
PLANE_BODY_INDEX = -1

PLANE_FACE_PARALLEL_DOT = 0.90
PLANE_FACE_KEEP_SLOP = 1.25e-2
PLANE_CORNER_KEEP_SLOP = 6.0e-3

CROSS_AXIS_PARALLEL_SKIP_DOT = 0.995
CROSS_AXIS_BASE_BIAS = 0.010
CROSS_AXIS_ALIGN_BIAS = 0.004
FACE_AXIS_ALIGN_BIAS = 0.00075
CROSS_AXIS_WIN_REL = 0.15
CROSS_AXIS_WIN_ABS = 2.0e-3

BOX_BOX_FACE_BIAS = 5.0e-4
BOX_BOX_CROSS_PARALLEL_SKIP_DOT = 0.999
BOX_BOX_CROSS_WIN_EPS = 1.0e-3
BOX_BOX_FACE_FALLBACK_SLOP = 2.0e-3

CONTACT_ID_KIND_FACE = 1
CONTACT_ID_KIND_EDGE = 2
CONTACT_ID_KIND_PLANE = 3

BOX_LOCAL_CORNERS = (
    vec3(-1.0, -1.0, -1.0),
    vec3(+1.0, -1.0, -1.0),
    vec3(+1.0, +1.0, -1.0),
    vec3(-1.0, +1.0, -1.0),
    vec3(-1.0, -1.0, +1.0),
    vec3(+1.0, -1.0, +1.0),
    vec3(+1.0, +1.0, +1.0),
    vec3(-1.0, +1.0, +1.0),
)

BOX_FACE_VERTS = (
    (0, 3, 7, 4),  # -X
    (1, 5, 6, 2),  # +X
    (0, 4, 5, 1),  # -Y
    (3, 2, 6, 7),  # +Y
    (0, 1, 2, 3),  # -Z
    (4, 7, 6, 5),  # +Z
)

BOX_FACE_AXIS_SIGN = (
    (0, -1.0),
    (0, +1.0),
    (1, -1.0),
    (1, +1.0),
    (2, -1.0),
    (2, +1.0),
)

BOX_EDGES = (
    (0, 1, 0), (3, 2, 0), (4, 5, 0), (7, 6, 0),  # X edges
    (0, 3, 1), (1, 2, 1), (4, 7, 1), (5, 6, 1),  # Y edges
    (0, 4, 2), (1, 5, 2), (2, 6, 2), (3, 7, 2),  # Z edges
)
