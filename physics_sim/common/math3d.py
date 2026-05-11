from pyglm import glm

Vec3 = glm.vec3
Mat3 = glm.mat3
Quat = glm.quat


def vec3(x=0.0, y=0.0, z=0.0):
    return Vec3(float(x), float(y), float(z))


def as_vec3(v):
    if isinstance(v, Vec3):
        return Vec3(v.x, v.y, v.z)
    return Vec3(v[0], v[1], v[2])


def v_to_list(v):
    return [float(v.x), float(v.y), float(v.z)]


def q_to_list(q):
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def safe_norm(v):
    return float(glm.length(v))

def mat3_from_diag(d):
    return Mat3(
        Vec3(d.x, 0.0, 0.0),
        Vec3(0.0, d.y, 0.0),
        Vec3(0.0, 0.0, d.z),
    )


def skew_mat(v):
    return Mat3(
        Vec3(0.0, v.z, -v.y),
        Vec3(-v.z, 0.0, v.x),
        Vec3(v.y, -v.x, 0.0),
    )