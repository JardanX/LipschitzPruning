#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kPrimitiveStride = 8;
constexpr int kMaxStack = 256;
constexpr int kWarpRowsPerEntry = 9;
constexpr int kWarpPackScale = 256;
constexpr int kDualContourHermiteSubdivisions = 2;
constexpr int kDualContourZeroCrossingIterations = 10;
constexpr int kWarpKindMirror = 1;
constexpr int kWarpKindGrid = 2;
constexpr int kWarpKindRadial = 3;
constexpr int kMirrorAxisX = 1;
constexpr int kMirrorAxisY = 2;
constexpr int kMirrorAxisZ = 4;
constexpr int kMirrorSideX = 8;
constexpr int kMirrorSideY = 16;
constexpr int kMirrorSideZ = 32;
constexpr double kPi = 3.14159265358979323846;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    Vec3 operator/(double scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

struct Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;
};

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2 operator+(const Vec2& other) const {
        return {x + other.x, y + other.y};
    }

    Vec2 operator-(const Vec2& other) const {
        return {x - other.x, y - other.y};
    }

    Vec2 operator*(double scalar) const {
        return {x * scalar, y * scalar};
    }
};

struct Primitive {
    Vec4 meta;
    Vec4 row0;
    Vec4 row1;
    Vec4 row2;
    Vec3 scale;
    Vec4 extra0;
    Vec4 extra1;
    Vec4 extra2;
    int warp_offset = 0;
    int warp_count = 0;
};

struct CompiledScene {
    std::vector<Primitive> primitives;
    std::vector<Vec4> polygon_rows;
    std::vector<Vec4> warp_rows;
    std::vector<Vec4> instructions;
};

struct CellVertex {
    bool active = false;
    Vec3 position;
    Vec3 normal;
};

struct QEFData {
    double ata[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double atb[3] = {0.0, 0.0, 0.0};
    double btb = 0.0;
    Vec3 centroid;
    int count = 0;
};

double clampd(double value, double lo, double hi)
{
    return std::max(lo, std::min(value, hi));
}

double mixd(double a, double b, double t)
{
    return a * (1.0 - t) + b * t;
}

double dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double dot(const Vec2& a, const Vec2& b)
{
    return a.x * b.x + a.y * b.y;
}

double length_sq(const Vec3& v)
{
    return dot(v, v);
}

double length(const Vec3& v)
{
    return std::sqrt(length_sq(v));
}

double length(const Vec2& v)
{
    return std::sqrt(dot(v, v));
}

Vec3 normalize(const Vec3& v)
{
    const double len = length(v);
    if (len <= 1.0e-12) {
        return {0.5773502691896257, 0.5773502691896257, 0.5773502691896257};
    }
    return v / len;
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 abs_vec(const Vec3& v)
{
    return {std::abs(v.x), std::abs(v.y), std::abs(v.z)};
}

Vec3 max_vec(const Vec3& v, double value)
{
    return {std::max(v.x, value), std::max(v.y, value), std::max(v.z, value)};
}

Vec3 clamp_vec(const Vec3& v, const Vec3& lo, const Vec3& hi)
{
    return {
        clampd(v.x, lo.x, hi.x),
        clampd(v.y, lo.y, hi.y),
        clampd(v.z, lo.z, hi.z),
    };
}

double max_component(const Vec3& v)
{
    return std::max(v.x, std::max(v.y, v.z));
}

double min_component(const Vec3& v)
{
    return std::min(v.x, std::min(v.y, v.z));
}

double mod_glsl(double x, double y)
{
    return x - y * std::floor(x / y);
}

double fract(double x)
{
    return x - std::floor(x);
}

CompiledScene build_scene(const std::vector<std::array<float, 4>>& primitive_rows,
                          const std::vector<std::array<float, 4>>& polygon_rows,
                          const std::vector<std::array<float, 4>>& warp_rows,
                          const std::vector<std::array<float, 4>>& instruction_rows)
{
    if ((primitive_rows.size() % kPrimitiveStride) != 0) {
        throw std::runtime_error("Primitive row count must be a multiple of 8");
    }

    CompiledScene scene;
    scene.primitives.reserve(primitive_rows.size() / kPrimitiveStride);
    for (std::size_t i = 0; i < primitive_rows.size(); i += kPrimitiveStride) {
        Primitive primitive;
        const auto& meta = primitive_rows[i + 0];
        const auto& row0 = primitive_rows[i + 1];
        const auto& row1 = primitive_rows[i + 2];
        const auto& row2 = primitive_rows[i + 3];
        const auto& scale = primitive_rows[i + 4];
        const auto& extra0 = primitive_rows[i + 5];
        const auto& extra1 = primitive_rows[i + 6];
        const auto& extra2 = primitive_rows[i + 7];
        primitive.meta = {meta[0], meta[1], meta[2], meta[3]};
        primitive.row0 = {row0[0], row0[1], row0[2], row0[3]};
        primitive.row1 = {row1[0], row1[1], row1[2], row1[3]};
        primitive.row2 = {row2[0], row2[1], row2[2], row2[3]};
        primitive.scale = {
            std::max<double>(scale[0], 1.0e-6),
            std::max<double>(scale[1], 1.0e-6),
            std::max<double>(scale[2], 1.0e-6),
        };
        primitive.extra0 = {extra0[0], extra0[1], extra0[2], extra0[3]};
        primitive.extra1 = {extra1[0], extra1[1], extra1[2], extra1[3]};
        primitive.extra2 = {extra2[0], extra2[1], extra2[2], extra2[3]};
        const int packed_warp = std::max(0, static_cast<int>(std::lround(scale[3])));
        primitive.warp_offset = packed_warp / kWarpPackScale;
        primitive.warp_count = packed_warp - primitive.warp_offset * kWarpPackScale;
        scene.primitives.push_back(primitive);
    }

    scene.polygon_rows.reserve(polygon_rows.size());
    for (const auto& row : polygon_rows) {
        scene.polygon_rows.push_back({row[0], row[1], row[2], row[3]});
    }

    scene.warp_rows.reserve(warp_rows.size());
    for (const auto& row : warp_rows) {
        scene.warp_rows.push_back({row[0], row[1], row[2], row[3]});
    }

    scene.instructions.reserve(instruction_rows.size());
    for (const auto& row : instruction_rows) {
        scene.instructions.push_back({row[0], row[1], row[2], row[3]});
    }
    return scene;
}

Vec3 world_to_primitive_local(const Primitive& primitive, const Vec3& world_point)
{
    return {
        primitive.row0.x * world_point.x + primitive.row0.y * world_point.y + primitive.row0.z * world_point.z + primitive.row0.w,
        primitive.row1.x * world_point.x + primitive.row1.y * world_point.y + primitive.row1.z * world_point.z + primitive.row1.w,
        primitive.row2.x * world_point.x + primitive.row2.y * world_point.y + primitive.row2.z * world_point.z + primitive.row2.w,
    };
}

Vec3 primitive_local_to_world(const Primitive& primitive, const Vec3& local_point)
{
    const Vec3 basis_x{primitive.row0.x, primitive.row1.x, primitive.row2.x};
    const Vec3 basis_y{primitive.row0.y, primitive.row1.y, primitive.row2.y};
    const Vec3 basis_z{primitive.row0.z, primitive.row1.z, primitive.row2.z};
    const Vec3 origin = {
        -(basis_x.x * primitive.row0.w + basis_y.x * primitive.row1.w + basis_z.x * primitive.row2.w),
        -(basis_x.y * primitive.row0.w + basis_y.y * primitive.row1.w + basis_z.y * primitive.row2.w),
        -(basis_x.z * primitive.row0.w + basis_y.z * primitive.row1.w + basis_z.z * primitive.row2.w),
    };
    return origin + basis_x * local_point.x + basis_y * local_point.y + basis_z * local_point.z;
}

Vec4 unpack_warp_rotation(const Vec3& packed_xyz)
{
    const double packed_w = std::sqrt(std::max(1.0 - dot(packed_xyz, packed_xyz), 0.0));
    const double inv_len = 1.0 / std::max(std::sqrt(dot(packed_xyz, packed_xyz) + packed_w * packed_w), 1.0e-12);
    return {packed_xyz.x * inv_len, packed_xyz.y * inv_len, packed_xyz.z * inv_len, packed_w * inv_len};
}

Vec4 quaternion_conjugate(const Vec4& rotation)
{
    return {-rotation.x, -rotation.y, -rotation.z, rotation.w};
}

Vec3 rotate_by_quaternion(const Vec4& rotation, const Vec3& point)
{
    const Vec3 axis{rotation.x, rotation.y, rotation.z};
    const double scalar = rotation.w;
    return axis * (2.0 * dot(axis, point))
         + point * (scalar * scalar - dot(axis, axis))
         + cross(axis, point) * (2.0 * scalar);
}

Vec3 safe_array_scale(const Vec3& scale)
{
    return {std::max(std::abs(scale.x), 1.0e-6), std::max(std::abs(scale.y), 1.0e-6), std::max(std::abs(scale.z), 1.0e-6)};
}

Vec3 world_to_array_local(const Vec3& world_point, const Vec3& origin, const Vec4& rotation, const Vec3& scale, const Vec3& branch_center)
{
    const Vec3 safe_scale = safe_array_scale(scale);
    const Vec3 local = rotate_by_quaternion(quaternion_conjugate(rotation), world_point - origin);
    return {branch_center.x + (local.x / safe_scale.x), branch_center.y + (local.y / safe_scale.y), branch_center.z + (local.z / safe_scale.z)};
}

Vec3 apply_instance_offset_inverse(const Vec3& point, const Vec3& branch_center,
                                  const Vec3& offset_location, const Vec4& offset_rotation,
                                  const Vec3& offset_scale)
{
    const Vec3 centered = point - (branch_center + offset_location);
    const Vec3 safe_scale = safe_array_scale(offset_scale);
    const Vec3 local = rotate_by_quaternion(quaternion_conjugate(offset_rotation), centered);
    return {
        branch_center.x + (local.x / safe_scale.x),
        branch_center.y + (local.y / safe_scale.y),
        branch_center.z + (local.z / safe_scale.z),
    };
}

double sabs(double x, double k)
{
    if (k <= 0.0001) {
        return std::abs(x);
    }
    const double ax = std::abs(x);
    if (ax >= k) {
        return ax;
    }
    const double t = ax / k;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t2 * t2;
    return k * (0.25 + 1.5 * t2 - t3 + 0.25 * t4);
}

double fold_finite_grid_axis(double axis_value, double origin, double primitive_center,
                             double spacing, int count, double blend)
{
    if (count <= 1 || spacing <= 1.0e-6) {
        return axis_value;
    }
    const double base = (primitive_center - origin) / spacing;
    const double center_offset = 0.5 * static_cast<double>(count - 1);
    const double shifted = (axis_value - origin) / spacing - base + center_offset;
    const double id = clampd(std::round(shifted), 0.0, static_cast<double>(count - 1));
    double local = shifted - id;
    const double d_r = (0.5 - local) * spacing;
    double pull_r = d_r - sabs(d_r, blend);
    const double d_l = (0.5 + local) * spacing;
    double pull_l = d_l - sabs(d_l, blend);
    if (id < 0.5) {
        pull_l = 0.0;
    }
    if (id > static_cast<double>(count) - 1.5) {
        pull_r = 0.0;
    }
    local += (pull_r - pull_l) / spacing;
    return origin + (local + base) * spacing;
}

Vec3 apply_radial_array(const Vec3& world_point, const Vec3& origin,
                        const Vec3& primitive_center, double radius,
                        int count, double blend)
{
    if (count <= 1 || radius <= 1.0e-6) {
        return world_point;
    }
    const Vec3 q = world_point - origin;
    const Vec3 base_offset = primitive_center - origin + Vec3{radius, 0.0, 0.0};
    const double base_angle = std::atan2(base_offset.y, base_offset.x);
    const double base_radius = std::max(std::sqrt(base_offset.x * base_offset.x + base_offset.y * base_offset.y), 1.0e-4);
    const double sector = (2.0 * kPi) / static_cast<double>(count);
    const double angle = std::atan2(q.y, q.x);
    const double angle_rel = mod_glsl(angle - base_angle + kPi, 2.0 * kPi) - kPi;
    const double norm_a = angle_rel / sector;
    const double id = std::round(norm_a);
    double local = norm_a - id;
    const double arc = sector * base_radius;
    const double d_r = (0.5 - local) * arc;
    const double pull_r = d_r - sabs(d_r, blend);
    const double d_l = (0.5 + local) * arc;
    const double pull_l = d_l - sabs(d_l, blend);
    local += (pull_r - pull_l) / std::max(arc, 1.0e-6);
    const double fold_a = local * sector + base_angle;
    const double r = std::sqrt(q.x * q.x + q.y * q.y);
    const Vec3 q_fold{r * std::cos(fold_a), r * std::sin(fold_a), q.z};
    return primitive_center + (q_fold - base_offset);
}

Vec3 to_local_point(const CompiledScene& scene, int primitive_index, const Vec3& world_point, double& distance_scale)
{
    const Primitive& primitive = scene.primitives.at(static_cast<std::size_t>(primitive_index));
    Vec3 warped_point = world_point;
    distance_scale = 1.0;
    for (int index = 0; index < primitive.warp_count; ++index) {
        const int warp_row = primitive.warp_offset + index * kWarpRowsPerEntry;
        if ((warp_row + 8) >= static_cast<int>(scene.warp_rows.size())) {
            break;
        }
        const Vec4& warp0 = scene.warp_rows[static_cast<std::size_t>(warp_row + 0)];
        const Vec4& warp1 = scene.warp_rows[static_cast<std::size_t>(warp_row + 1)];
        const Vec4& warp2 = scene.warp_rows[static_cast<std::size_t>(warp_row + 2)];
        const Vec4& warp3 = scene.warp_rows[static_cast<std::size_t>(warp_row + 3)];
        const Vec4& warp4 = scene.warp_rows[static_cast<std::size_t>(warp_row + 4)];
        const Vec4& warp5 = scene.warp_rows[static_cast<std::size_t>(warp_row + 5)];
        const Vec4& warp6 = scene.warp_rows[static_cast<std::size_t>(warp_row + 6)];
        const Vec4& warp7 = scene.warp_rows[static_cast<std::size_t>(warp_row + 7)];
        const Vec4& warp8 = scene.warp_rows[static_cast<std::size_t>(warp_row + 8)];
        const int warp_kind = static_cast<int>(std::lround(warp0.x));
        if (warp_kind == kWarpKindMirror) {
            const int flags = static_cast<int>(std::lround(warp0.y));
            const double blend = std::max(warp0.z, 0.0);
            const Vec3 origin{warp1.x, warp1.y, warp1.z};
            if ((flags & kMirrorAxisX) != 0) {
                const double side = ((flags & kMirrorSideX) != 0) ? 1.0 : -1.0;
                warped_point.x = origin.x + side * sabs(side * (warped_point.x - origin.x), blend);
            }
            if ((flags & kMirrorAxisY) != 0) {
                const double side = ((flags & kMirrorSideY) != 0) ? 1.0 : -1.0;
                warped_point.y = origin.y + side * sabs(side * (warped_point.y - origin.y), blend);
            }
            if ((flags & kMirrorAxisZ) != 0) {
                const double side = ((flags & kMirrorSideZ) != 0) ? 1.0 : -1.0;
                warped_point.z = origin.z + side * sabs(side * (warped_point.z - origin.z), blend);
            }
            continue;
        }
        if (warp_kind == kWarpKindGrid) {
            const int count_x = static_cast<int>(std::lround(warp0.y));
            const int count_y = static_cast<int>(std::lround(warp0.z));
            const int count_z = static_cast<int>(std::lround(warp0.w));
            const Vec3 spacing{std::abs(warp1.x), std::abs(warp1.y), std::abs(warp1.z)};
            const double blend = std::max(warp1.w, 0.0);
            const Vec3 origin{warp2.x, warp2.y, warp2.z};
            const Vec4 rotation = unpack_warp_rotation({warp3.x, warp3.y, warp3.z});
            const Vec3 scale{warp4.x, warp4.y, warp4.z};
            const Vec3 branch_center{warp5.x, warp5.y, warp5.z};
            const Vec3 offset_location{warp6.x, warp6.y, warp6.z};
            const Vec4 offset_rotation = unpack_warp_rotation({warp7.x, warp7.y, warp7.z});
            const Vec3 offset_scale{warp8.x, warp8.y, warp8.z};
            distance_scale *= std::max(min_component(safe_array_scale(scale)), 1.0e-6);
            distance_scale *= std::max(min_component(safe_array_scale(offset_scale)), 1.0e-6);
            Vec3 array_point = world_to_array_local(warped_point, origin, rotation, scale, branch_center);
            const Vec3 offset_center = branch_center + offset_location;
            array_point.x = fold_finite_grid_axis(array_point.x, offset_center.x, offset_center.x, spacing.x, count_x, blend);
            array_point.y = fold_finite_grid_axis(array_point.y, offset_center.y, offset_center.y, spacing.y, count_y, blend);
            array_point.z = fold_finite_grid_axis(array_point.z, offset_center.z, offset_center.z, spacing.z, count_z, blend);
            warped_point = apply_instance_offset_inverse(array_point, branch_center, offset_location, offset_rotation, offset_scale);
            continue;
        }
        if (warp_kind == kWarpKindRadial) {
            const int count = static_cast<int>(std::lround(warp0.y));
            const double blend = std::max(warp0.z, 0.0);
            const double radius = std::max(warp0.w, 0.0);
            const Vec3 repeat_origin{warp1.x, warp1.y, warp1.z};
            const Vec3 field_origin{warp2.x, warp2.y, warp2.z};
            const Vec4 rotation = unpack_warp_rotation({warp3.x, warp3.y, warp3.z});
            const Vec3 scale{warp4.x, warp4.y, warp4.z};
            const Vec3 branch_center{warp5.x, warp5.y, warp5.z};
            const Vec3 offset_location{warp6.x, warp6.y, warp6.z};
            const Vec4 offset_rotation = unpack_warp_rotation({warp7.x, warp7.y, warp7.z});
            const Vec3 offset_scale{warp8.x, warp8.y, warp8.z};
            distance_scale *= std::max(min_component(safe_array_scale(scale)), 1.0e-6);
            distance_scale *= std::max(min_component(safe_array_scale(offset_scale)), 1.0e-6);
            Vec3 array_point = world_to_array_local(warped_point, field_origin, rotation, scale, branch_center);
            array_point = apply_radial_array(array_point, repeat_origin, branch_center + offset_location, radius, count, blend);
            warped_point = apply_instance_offset_inverse(array_point, branch_center, offset_location, offset_rotation, offset_scale);
        }
    }
    return world_to_primitive_local(primitive, warped_point);
}

double sd_ellipsoid(const Vec3& p, const Vec3& r)
{
    const Vec3 safe_r = max_vec(r, 1.0e-6);
    const Vec3 p_over_r{p.x / safe_r.x, p.y / safe_r.y, p.z / safe_r.z};
    const Vec3 rr{safe_r.x * safe_r.x, safe_r.y * safe_r.y, safe_r.z * safe_r.z};
    const Vec3 p_over_rr{p.x / rr.x, p.y / rr.y, p.z / rr.z};
    const double k0 = length(p_over_r);
    const double k1 = std::max(length(p_over_rr), 1.0e-6);
    return k0 * (k0 - 1.0) / k1;
}

double sd_box(const Vec3& p, const Vec3& b)
{
    const Vec3 q = abs_vec(p) - b;
    const Vec3 positive = max_vec(q, 0.0);
    return length(positive) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0);
}

double sd_round_box_2d(const Vec2& p, const Vec2& b, const Vec4& r)
{
    const double rx = (p.x > 0.0) ? r.x : r.z;
    const double ry = (p.x > 0.0) ? r.y : r.w;
    const double rc = (p.y > 0.0) ? rx : ry;
    const Vec2 q{std::abs(p.x) - b.x + rc, std::abs(p.y) - b.y + rc};
    const Vec2 positive{std::max(q.x, 0.0), std::max(q.y, 0.0)};
    return std::min(std::max(q.x, q.y), 0.0) + length(positive) - rc;
}

double sd_chamfer_box_2d(const Vec2& p, const Vec2& b, const Vec4& r)
{
    const double rx = (p.x > 0.0) ? r.x : r.z;
    const double ry = (p.x > 0.0) ? r.y : r.w;
    const double rc = (p.y > 0.0) ? rx : ry;
    const Vec2 q{std::abs(p.x) - b.x, std::abs(p.y) - b.y};
    if (rc < 0.001) {
        const Vec2 positive{std::max(q.x, 0.0), std::max(q.y, 0.0)};
        return length(positive) + std::min(std::max(q.x, q.y), 0.0);
    }
    const double d_cham = (q.x + q.y + rc) * 0.70710678;
    const double d_box = std::max(q.x, q.y);
    const double d = std::max(d_box, d_cham);
    if (d <= 0.0) {
        return d;
    }
    if (d_box <= 0.0) {
        return d_cham;
    }
    if (q.y <= -rc || q.x <= -rc) {
        const Vec2 positive{std::max(q.x, 0.0), std::max(q.y, 0.0)};
        return length(positive);
    }
    const double t = (-q.x + q.y + rc) / (2.0 * rc);
    if (t <= 0.0) {
        return length(q - Vec2{0.0, -rc});
    }
    if (t >= 1.0) {
        return length(q - Vec2{-rc, 0.0});
    }
    return d_cham;
}

double sd_advanced_box(const Vec3& p,
                       const Vec3& size,
                       const Vec4& corners,
                       const double edge_top,
                       const double edge_bottom,
                       const double taper,
                       const int corner_mode,
                       const int edge_mode)
{
    const double taper_z = std::max(size.z, 0.001);
    const double zn = clampd(p.z / taper_z, -1.0, 1.0);
    const double mix_value = (zn + 1.0) * 0.5;
    const double tap_top = std::max(taper, 0.0);
    const double tap_bottom = std::max(-taper, 0.0);
    const double tap_factor = std::max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001);
    const Vec2 size_xy{size.x * tap_factor, size.y * tap_factor};
    const double slope = std::sqrt(size.x * size.x + size.y * size.y) * (tap_top + tap_bottom) / (2.0 * taper_z);
    const double lipschitz = std::sqrt(1.0 + slope * slope);
    const double max_radius = std::min(size_xy.x, size_xy.y);
    const Vec4 scaled_corners{corners.x * max_radius, corners.y * max_radius, corners.z * max_radius, corners.w * max_radius};
    const double d2d = (corner_mode == 0) ? sd_round_box_2d({p.x, p.y}, size_xy, scaled_corners) : sd_chamfer_box_2d({p.x, p.y}, size_xy, scaled_corners);
    const double max_face_radius = std::min(size_xy.x, size_xy.y);
    const double dz = std::abs(p.z) - size.z;
    const double edge_radius = ((p.z > 0.0) ? edge_top : edge_bottom) * std::min(max_face_radius, size.z);
    if (edge_radius > 0.001) {
        if (edge_mode == 0) {
            const double dd0 = d2d + edge_radius;
            const double dd1 = dz + edge_radius;
            return (std::min(std::max(dd0, dd1), 0.0) + std::sqrt(std::max(dd0, 0.0) * std::max(dd0, 0.0) + std::max(dd1, 0.0) * std::max(dd1, 0.0)) - edge_radius) / lipschitz;
        }
        const double base = std::max(d2d, dz);
        const double cham = (d2d + dz + edge_radius) * 0.70710678;
        const double dd = std::max(base, cham);
        if (dd <= 0.0) { return dd / lipschitz; }
        if (d2d <= 0.0 && dz <= 0.0) { return cham / lipschitz; }
        if (dz <= -edge_radius) { return d2d / lipschitz; }
        if (d2d <= -edge_radius) { return dz / lipschitz; }
        const double tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius);
        if (tc2 <= 0.0) { return std::hypot(d2d, dz + edge_radius) / lipschitz; }
        if (tc2 >= 1.0) { return std::hypot(d2d + edge_radius, dz) / lipschitz; }
        return cham / lipschitz;
    }
    const double dd0 = std::max(d2d, 0.0);
    const double dd1 = std::max(dz, 0.0);
    return (std::sqrt(dd0 * dd0 + dd1 * dd1) + std::min(std::max(d2d, dz), 0.0)) / lipschitz;
}

double sd_cylinder(const Vec3& p, double radius, double half_height)
{
    const double d0 = std::abs(std::sqrt(p.x * p.x + p.y * p.y)) - radius;
    const double d1 = std::abs(p.z) - half_height;
    const double outside_x = std::max(d0, 0.0);
    const double outside_y = std::max(d1, 0.0);
    return std::min(std::max(d0, d1), 0.0) + std::sqrt(outside_x * outside_x + outside_y * outside_y);
}

double sd_advanced_cylinder(const Vec3& p,
                            const double radius,
                            const double half_height,
                            const double bevel_top,
                            const double bevel_bottom,
                            const double taper,
                            const int bevel_mode)
{
    const double taper_h = std::max(half_height, 0.001);
    const double zn = clampd(p.z / taper_h, -1.0, 1.0);
    const double mix_value = (zn + 1.0) * 0.5;
    const double tap_top = std::max(taper, 0.0);
    const double tap_bottom = std::max(-taper, 0.0);
    const double tap_factor = std::max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001);
    const double scaled_radius = radius * tap_factor;
    const double slope = radius * (tap_top + tap_bottom) / (2.0 * taper_h);
    const double lipschitz = std::sqrt(1.0 + slope * slope);
    const double d2d = std::sqrt(p.x * p.x + p.y * p.y) - scaled_radius;
    const double dz = std::abs(p.z) - half_height;
    const double edge_bevel = (p.z > 0.0) ? bevel_top : bevel_bottom;
    const double edge_radius = std::min(std::max(edge_bevel, 0.0), std::max(std::min(scaled_radius, half_height) - 0.001, 0.0));
    if (edge_radius > 0.001) {
        if (bevel_mode == 0) {
            const double dd0 = d2d + edge_radius;
            const double dd1 = dz + edge_radius;
            return (std::min(std::max(dd0, dd1), 0.0) + std::sqrt(std::max(dd0, 0.0) * std::max(dd0, 0.0) + std::max(dd1, 0.0) * std::max(dd1, 0.0)) - edge_radius) / lipschitz;
        }
        const double base = std::max(d2d, dz);
        const double cham = (d2d + dz + edge_radius) * 0.70710678;
        const double dd = std::max(base, cham);
        if (dd <= 0.0) { return dd / lipschitz; }
        if (d2d <= 0.0 && dz <= 0.0) { return cham / lipschitz; }
        if (dz <= -edge_radius) { return d2d / lipschitz; }
        if (d2d <= -edge_radius) { return dz / lipschitz; }
        const double tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius);
        if (tc2 <= 0.0) { return std::hypot(d2d, dz + edge_radius) / lipschitz; }
        if (tc2 >= 1.0) { return std::hypot(d2d + edge_radius, dz) / lipschitz; }
        return cham / lipschitz;
    }
    const double dd0 = std::max(d2d, 0.0);
    const double dd1 = std::max(dz, 0.0);
    return (std::sqrt(dd0 * dd0 + dd1 * dd1) + std::min(std::max(d2d, dz), 0.0)) / lipschitz;
}

double sd_torus(const Vec3& p, double major_radius, double minor_radius)
{
    const double qx = std::sqrt(p.x * p.x + p.y * p.y) - major_radius;
    const double qy = p.z;
    return std::sqrt(qx * qx + qy * qy) - minor_radius;
}

double sd_capped_torus(Vec3 p, double half_angle, double major_radius, double minor_radius)
{
    p.x = std::abs(p.x);
    const Vec2 sc{std::sin(half_angle), std::cos(half_angle)};
    const double k = ((sc.y * p.x) > (sc.x * p.y)) ? (p.x * sc.x + p.y * sc.y) : std::sqrt(p.x * p.x + p.y * p.y);
    return std::sqrt(dot(p, p) + major_radius * major_radius - 2.0 * major_radius * k) - minor_radius;
}

double sd_cone(const Vec3& p, double bottom_radius, double top_radius, double half_height)
{
    const double qx = std::sqrt(p.x * p.x + p.y * p.y);
    const double qy = p.z;
    const Vec2 q{qx, qy};
    const Vec2 k1{top_radius, half_height};
    const Vec2 k2{top_radius - bottom_radius, 2.0 * half_height};
    const Vec2 ca{q.x - std::min(q.x, (q.y < 0.0) ? bottom_radius : top_radius), std::abs(q.y) - half_height};
    const double denom = std::max(dot(k2, k2), 1.0e-12);
    const double t = clampd(dot(k1 - q, k2) / denom, 0.0, 1.0);
    const Vec2 cb = q - k1 + (k2 * t);
    const double sign = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return sign * std::sqrt(std::min(dot(ca, ca), dot(cb, cb)));
}

double sd_advanced_cone(const Vec3& p,
                        const double bottom_radius,
                        const double top_radius,
                        const double half_height,
                        const double bevel,
                        const int bevel_mode)
{
    const double radial = std::sqrt(p.x * p.x + p.y * p.y);
    const Vec2 edge{top_radius - bottom_radius, 2.0 * half_height};
    const double edge_len = std::max(length(edge), 1.0e-12);
    const double dside = (edge.y * (radial - bottom_radius) - edge.x * (p.z + half_height)) / edge_len;
    const double dz = std::abs(p.z) - half_height;
    const double edge_radius = std::min(std::max(bevel, 0.0), std::max(half_height - 0.001, 0.0));
    if (edge_radius > 0.001) {
        if (bevel_mode == 0) {
            const double dd0 = dside + edge_radius;
            const double dd1 = dz + edge_radius;
            return std::min(std::max(dd0, dd1), 0.0) + std::sqrt(std::max(dd0, 0.0) * std::max(dd0, 0.0) + std::max(dd1, 0.0) * std::max(dd1, 0.0)) - edge_radius;
        }
        const double base = std::max(dside, dz);
        const double cham = (dside + dz + edge_radius) * 0.70710678;
        const double dd = std::max(base, cham);
        if (dd <= 0.0) { return dd; }
        if (dside <= 0.0 && dz <= 0.0) { return cham; }
        if (dz <= -edge_radius) { return dside; }
        if (dside <= -edge_radius) { return dz; }
        const double tc2 = (-dside + dz + edge_radius) / (2.0 * edge_radius);
        if (tc2 <= 0.0) { return std::hypot(dside, dz + edge_radius); }
        if (tc2 >= 1.0) { return std::hypot(dside + edge_radius, dz); }
        return cham;
    }
    return sd_cone(p, bottom_radius, top_radius, half_height);
}

double sd_round_cone(const Vec3& p, double bottom_radius, double top_radius, double half_height)
{
    const Vec2 q{std::sqrt(p.x * p.x + p.y * p.y), p.z + half_height};
    const double total_height = std::max(2.0 * half_height, 1.0e-6);
    if (std::abs(bottom_radius - top_radius) >= total_height) {
        return (bottom_radius >= top_radius) ? (length(q) - bottom_radius) : (length(q - Vec2{0.0, total_height}) - top_radius);
    }
    const double slope = (bottom_radius - top_radius) / total_height;
    const double axis = std::sqrt(std::max(1.0 - slope * slope, 1.0e-12));
    const double k = -slope * q.x + axis * q.y;
    if (k <= 0.0) {
        return length(q) - bottom_radius;
    }
    if (k >= axis * total_height) {
        return length(q - Vec2{0.0, total_height}) - top_radius;
    }
    return axis * q.x + slope * q.y - bottom_radius;
}

Vec2 capsule_end_radii(double radius, double taper)
{
    const double tap_top = std::max(taper, 0.0);
    const double tap_bottom = std::max(-taper, 0.0);
    return {radius * std::max(1.0 - tap_bottom, 0.0), radius * std::max(1.0 - tap_top, 0.0)};
}

double sd_capsule(const Vec3& p, double radius, double half_height, double taper)
{
    const Vec2 radii = capsule_end_radii(radius, taper);
    return sd_round_cone(p, radii.x, radii.y, half_height);
}

double sd_regular_polygon_2d(Vec2 p, double radius, int sides)
{
    const int safe_sides = std::max(sides, 3);
    const double angle_step = kPi / static_cast<double>(safe_sides);
    const double inner_radius = radius * std::cos(angle_step);
    const double half_edge = radius * std::sin(angle_step);
    p = {-p.y, p.x};
    const double sector = angle_step * std::floor((std::atan2(p.y, p.x) + angle_step) / angle_step / 2.0) * 2.0;
    const Vec2 cs{std::cos(sector), std::sin(sector)};
    p = {cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y};
    const Vec2 projected{inner_radius, clampd(p.y, -half_edge, half_edge)};
    const Vec2 diff = p - projected;
    return length(diff) * ((p.x - inner_radius) >= 0.0 ? 1.0 : -1.0);
}

double sd_star_polygon_2d(Vec2 p, double radius, int sides, double star)
{
    if (star < 0.001) {
        return sd_regular_polygon_2d(p, radius, sides);
    }
    const int safe_sides = std::max(sides, 3);
    const double angle_step = kPi / static_cast<double>(safe_sides);
    const double inner_radius = radius * std::cos(angle_step) * std::max(1.0 - star, 0.01);
    const double angle = std::atan2(p.y, p.x);
    const double sector = std::floor((angle + angle_step) / (2.0 * angle_step)) * 2.0 * angle_step;
    const Vec2 cs{std::cos(sector), std::sin(sector)};
    Vec2 q{cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y};
    q.y = std::abs(q.y);
    const Vec2 a{radius, 0.0};
    const Vec2 b{inner_radius * std::cos(angle_step), inner_radius * std::sin(angle_step)};
    const Vec2 ab = b - a;
    const double denom = std::max(dot(ab, ab), 1.0e-12);
    const double t = clampd(dot(q - a, ab) / denom, 0.0, 1.0);
    const Vec2 proj = a + (ab * t);
    const double dist = length(q - proj);
    const double cross_value = ab.x * (q.y - a.y) - ab.y * (q.x - a.x);
    return (cross_value > 0.0) ? -dist : dist;
}

Vec2 load_polygon_point(const CompiledScene& scene, int index)
{
    if (index < 0 || index >= static_cast<int>(scene.polygon_rows.size())) {
        return {0.0, 0.0};
    }
    const Vec4& row = scene.polygon_rows[static_cast<std::size_t>(index)];
    return {row.x, row.y};
}

double segment_distance_2d(const Vec2& p, const Vec2& a, const Vec2& b)
{
    const Vec2 edge = b - a;
    const double edge_len_sq = std::max(dot(edge, edge), 1.0e-12);
    const double t = clampd(dot(p - a, edge) / edge_len_sq, 0.0, 1.0);
    return length((p - a) - (edge * t));
}

double sd_polygon_2d(const CompiledScene& scene, const Vec2& p, int point_offset, int point_count)
{
    if (point_count < 3) {
        return 1.0e20;
    }
    double d = 1.0e20;
    int winding = 0;
    for (int i = 0; i < point_count; ++i) {
        const Vec2 a = load_polygon_point(scene, point_offset + i);
        const Vec2 b = load_polygon_point(scene, point_offset + ((i + 1) % point_count));
        d = std::min(d, segment_distance_2d(p, a, b));
        const Vec2 edge = b - a;
        const Vec2 rel = p - a;
        const double cross_value = edge.x * rel.y - edge.y * rel.x;
        if (a.y <= p.y && b.y > p.y && cross_value > 0.0) {
            ++winding;
        }
        if (a.y > p.y && b.y <= p.y && cross_value < 0.0) {
            --winding;
        }
    }
    return (winding != 0) ? -d : d;
}

double sd_prism(double distance_2d, double z, double half_height)
{
    const double dz = std::abs(z) - half_height;
    const double outside_x = std::max(distance_2d, 0.0);
    const double outside_y = std::max(dz, 0.0);
    return std::min(std::max(distance_2d, dz), 0.0) + std::sqrt(outside_x * outside_x + outside_y * outside_y);
}

double sd_advanced_ngon(const Vec3& p,
                        const double radius,
                        const double half_height,
                        const int sides,
                        const double corner,
                        const double edge_top,
                        const double edge_bottom,
                        const double taper,
                        const int edge_mode,
                        const double star)
{
    const double taper_h = std::max(half_height, 0.001);
    const double zn = clampd(p.z / taper_h, -1.0, 1.0);
    const double mix_value = (zn + 1.0) * 0.5;
    const double tap_top = std::max(taper, 0.0);
    const double tap_bottom = std::max(-taper, 0.0);
    const double tap_factor = std::max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001);
    const double scaled_radius = radius * tap_factor;
    const double slope = radius * (tap_top + tap_bottom) / (2.0 * taper_h);
    const double lipschitz = std::sqrt(1.0 + slope * slope);
    const double an = kPi / static_cast<double>(std::max(sides, 3));
    const double apothem = scaled_radius * std::cos(an);
    const double bevel_radius = corner * apothem;
    const double inner_radius = scaled_radius - bevel_radius / std::max(std::cos(an), 0.001);
    const double d2d = ((star > 0.001) ? sd_star_polygon_2d({p.x, p.y}, inner_radius, sides, star) : sd_regular_polygon_2d({p.x, p.y}, inner_radius, sides)) - bevel_radius;
    const double dz = std::abs(p.z) - half_height;
    const double edge_radius = ((p.z > 0.0) ? edge_top : edge_bottom) * std::min(apothem, half_height);
    if (edge_radius > 0.001) {
        if (edge_mode == 0) {
            const double dd0 = d2d + edge_radius;
            const double dd1 = dz + edge_radius;
            return (std::min(std::max(dd0, dd1), 0.0) + std::sqrt(std::max(dd0, 0.0) * std::max(dd0, 0.0) + std::max(dd1, 0.0) * std::max(dd1, 0.0)) - edge_radius) / lipschitz;
        }
        const double base = std::max(d2d, dz);
        const double cham = (d2d + dz + edge_radius) * 0.70710678;
        const double dd = std::max(base, cham);
        if (dd <= 0.0) { return dd / lipschitz; }
        if (d2d <= 0.0 && dz <= 0.0) { return cham / lipschitz; }
        if (dz <= -edge_radius) { return d2d / lipschitz; }
        if (d2d <= -edge_radius) { return dz / lipschitz; }
        const double tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius);
        if (tc2 <= 0.0) { return std::hypot(d2d, dz + edge_radius) / lipschitz; }
        if (tc2 >= 1.0) { return std::hypot(d2d + edge_radius, dz) / lipschitz; }
        return cham / lipschitz;
    }
    const double dd0 = std::max(d2d, 0.0);
    const double dd1 = std::max(dz, 0.0);
    return (std::sqrt(dd0 * dd0 + dd1 * dd1) + std::min(std::max(d2d, dz), 0.0)) / lipschitz;
}

double sd_advanced_polygon(const CompiledScene& scene,
                           const Vec3& p,
                           const double half_height,
                           const int point_offset,
                           const int point_count,
                           const double edge_top,
                           const double edge_bottom,
                           const double taper,
                           const int edge_mode)
{
    const double taper_h = std::max(half_height, 0.001);
    const double zn = clampd(p.z / taper_h, -1.0, 1.0);
    const double mix_value = (zn + 1.0) * 0.5;
    const double tap_top = std::max(taper, 0.0);
    const double tap_bottom = std::max(-taper, 0.0);
    const double tap_factor = std::max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001);
    const double slope = (tap_top + tap_bottom) / (2.0 * taper_h);
    const double lipschitz = std::sqrt(1.0 + slope * slope);
    const double d2d = sd_polygon_2d(scene, {p.x / tap_factor, p.y / tap_factor}, point_offset, point_count) * tap_factor;
    const double dz = std::abs(p.z) - half_height;
    const double edge_radius = ((p.z > 0.0) ? edge_top : edge_bottom) * half_height;
    if (edge_radius > 0.001) {
        if (edge_mode == 0) {
            const double dd0 = d2d + edge_radius;
            const double dd1 = dz + edge_radius;
            return (std::min(std::max(dd0, dd1), 0.0) + std::sqrt(std::max(dd0, 0.0) * std::max(dd0, 0.0) + std::max(dd1, 0.0) * std::max(dd1, 0.0)) - edge_radius) / lipschitz;
        }
        const double base = std::max(d2d, dz);
        const double cham = (d2d + dz + edge_radius) * 0.70710678;
        const double dd = std::max(base, cham);
        if (dd <= 0.0) { return dd / lipschitz; }
        if (d2d <= 0.0 && dz <= 0.0) { return cham / lipschitz; }
        if (dz <= -edge_radius) { return d2d / lipschitz; }
        if (d2d <= -edge_radius) { return dz / lipschitz; }
        const double tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius);
        if (tc2 <= 0.0) { return std::hypot(d2d, dz + edge_radius) / lipschitz; }
        if (tc2 >= 1.0) { return std::hypot(d2d + edge_radius, dz) / lipschitz; }
        return cham / lipschitz;
    }
    const double dd0 = std::max(d2d, 0.0);
    const double dd1 = std::max(dz, 0.0);
    return (std::sqrt(dd0 * dd0 + dd1 * dd1) + std::min(std::max(d2d, dz), 0.0)) / lipschitz;
}

double kernel(double x, double k)
{
    if (k <= 0.0) {
        return 0.0;
    }
    const double m = std::max(0.0, k - x);
    return m * m * 0.25 / k;
}

double op_smooth_union(double a, double b, double k)
{
    const double h = clampd(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mixd(b, a, h) - k * h * (1.0 - h);
}

double op_smooth_subtract(double a, double b, double k)
{
    const double h = clampd(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    return mixd(a, -b, h) + k * h * (1.0 - h);
}

double op_smooth_intersect(double a, double b, double k)
{
    const double h = clampd(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mixd(b, a, h) + k * h * (1.0 - h);
}

double apply_op(int op, double a, double b, double blend)
{
    if (op == 1) {
        return (blend <= 1.0e-6) ? std::min(a, b) : op_smooth_union(a, b, blend);
    }
    if (op == 2) {
        return (blend <= 1.0e-6) ? std::max(a, -b) : op_smooth_subtract(a, b, blend);
    }
    if (op == 3) {
        return (blend <= 1.0e-6) ? std::max(a, b) : op_smooth_intersect(a, b, blend);
    }
    return b;
}

double eval_primitive(const CompiledScene& scene, int primitive_index, const Vec3& world_point)
{
    const Primitive& primitive = scene.primitives.at(static_cast<std::size_t>(primitive_index));
    double distance_scale = 1.0;
    const Vec3 local_point = to_local_point(scene, primitive_index, world_point, distance_scale);
    const Vec3 scale = primitive.scale;
    const double min_scale = std::max(min_component(scale), 1.0e-6);
    const int primitive_type = static_cast<int>(std::lround(primitive.meta.x));
    const double bevel = std::max(primitive.extra0.x, 0.0);
    double post_bevel = bevel;
    if (primitive_type == 0) {
        return (sd_ellipsoid(local_point, {primitive.meta.y * scale.x, primitive.meta.y * scale.y, primitive.meta.y * scale.z}) - bevel) * distance_scale;
    }
    if (primitive_type == 1) {
        const Vec3 size{primitive.meta.y * scale.x, primitive.meta.z * scale.y, primitive.meta.w * scale.z};
        const double detail_sum = primitive.extra0.y + primitive.extra0.z + primitive.extra0.w + primitive.extra1.x + primitive.extra1.y + primitive.extra1.z + std::abs(primitive.extra1.w);
        const double distance = (detail_sum > 0.001)
            ? sd_advanced_box(local_point, size, {primitive.extra0.y, primitive.extra0.z, primitive.extra0.w, primitive.extra1.x}, primitive.extra1.y, primitive.extra1.z, primitive.extra1.w, static_cast<int>(std::lround(primitive.extra2.x)), static_cast<int>(std::lround(primitive.extra2.y)))
            : sd_box(local_point, size);
        return (distance - bevel) * distance_scale;
    }
    if (primitive_type == 2) {
        const Vec3 scaled_point{local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z};
        const double bevel_top = std::max(primitive.extra0.x, 0.0);
        const double bevel_bottom = std::max(primitive.extra0.y, 0.0);
        const double distance = ((std::max(bevel_top, bevel_bottom) + std::abs(primitive.extra0.z)) > 0.001)
            ? sd_advanced_cylinder(scaled_point, primitive.meta.y, primitive.meta.z, bevel_top, bevel_bottom, primitive.extra0.z, static_cast<int>(std::lround(primitive.extra0.w)))
            : sd_cylinder(scaled_point, primitive.meta.y, primitive.meta.z);
        return distance * min_scale * distance_scale;
    }
    if (primitive_type == 3) {
        const Vec3 scaled_point{local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z};
        const double angle = clampd(primitive.extra0.y, 0.0, kPi * 2.0);
        const double distance = ((kPi * 2.0 - angle) > 0.001)
            ? sd_capped_torus(scaled_point, angle * 0.5, primitive.meta.y, primitive.meta.z)
            : sd_torus(scaled_point, primitive.meta.y, primitive.meta.z);
        return distance * min_scale * distance_scale;
    }
    if (primitive_type == 4) {
        const Vec3 scaled_point{local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z};
        const double distance = (bevel > 0.001)
            ? sd_advanced_cone(scaled_point, primitive.meta.y, primitive.meta.z, primitive.meta.w, bevel, static_cast<int>(std::lround(primitive.extra0.y)))
            : sd_cone(scaled_point, primitive.meta.y, primitive.meta.z, primitive.meta.w);
        return distance * min_scale * distance_scale;
    }
    if (primitive_type == 5) {
        return sd_capsule({local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z}, primitive.meta.y, primitive.meta.z, primitive.extra0.y) * min_scale * distance_scale;
    }
    if (primitive_type == 6) {
        const Vec3 scaled_point{local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z};
        const double detail_sum = primitive.extra0.y + primitive.extra0.z + primitive.extra0.w + std::abs(primitive.extra1.x) + primitive.extra1.z;
        const double distance = (detail_sum > 0.001)
            ? sd_advanced_ngon(scaled_point, primitive.meta.y, primitive.meta.z, static_cast<int>(std::lround(primitive.meta.w)), primitive.extra0.y, primitive.extra0.z, primitive.extra0.w, primitive.extra1.x, static_cast<int>(std::lround(primitive.extra1.y)), primitive.extra1.z)
            : sd_prism(sd_regular_polygon_2d({scaled_point.x, scaled_point.y}, primitive.meta.y, static_cast<int>(std::lround(primitive.meta.w))), scaled_point.z, primitive.meta.z);
        return distance * min_scale * distance_scale;
    }
    if (primitive_type == 7) {
        const Vec3 scaled_point{local_point.x / scale.x, local_point.y / scale.y, local_point.z / scale.z};
        const double detail_sum = primitive.extra0.y + primitive.extra0.z + std::abs(primitive.extra0.w);
        const double distance = (detail_sum > 0.001)
            ? sd_advanced_polygon(scene, scaled_point, primitive.meta.w, static_cast<int>(std::lround(primitive.meta.y)), static_cast<int>(std::lround(primitive.meta.z)), primitive.extra0.y, primitive.extra0.z, primitive.extra0.w, static_cast<int>(std::lround(primitive.extra1.x)))
            : sd_prism(sd_polygon_2d(scene, {scaled_point.x, scaled_point.y}, static_cast<int>(std::lround(primitive.meta.y)), static_cast<int>(std::lround(primitive.meta.z))), scaled_point.z, primitive.meta.w);
        return (distance - post_bevel) * min_scale * distance_scale;
    }
    return 1.0e20;
}

double eval_scene_distance(const CompiledScene& scene, const Vec3& world_point)
{
    double stack[kMaxStack];
    int stack_pointer = 0;
    for (const auto& instruction : scene.instructions) {
        const int kind = static_cast<int>(std::lround(instruction.x));
        double distance_value = 1.0e20;
        if (kind == 0) {
            distance_value = eval_primitive(scene, static_cast<int>(std::lround(instruction.y)), world_point);
        } else {
            if (stack_pointer < 2) {
                return 1.0e20;
            }
            const double rhs = stack[stack_pointer - 1];
            const double lhs = stack[stack_pointer - 2];
            stack_pointer -= 2;
            distance_value = apply_op(kind, lhs, rhs, std::max(0.0, static_cast<double>(instruction.w)));
        }
        if (stack_pointer >= kMaxStack) {
            return 1.0e20;
        }
        stack[stack_pointer++] = distance_value;
    }
    return (stack_pointer == 0) ? 1.0e20 : stack[stack_pointer - 1];
}

Vec3 eval_scene_normal(const CompiledScene& scene, const Vec3& world_point, double epsilon)
{
    const Vec3 dx{epsilon, 0.0, 0.0};
    const Vec3 dy{0.0, epsilon, 0.0};
    const Vec3 dz{0.0, 0.0, epsilon};
    const double sx = eval_scene_distance(scene, world_point + dx) - eval_scene_distance(scene, world_point - dx);
    const double sy = eval_scene_distance(scene, world_point + dy) - eval_scene_distance(scene, world_point - dy);
    const double sz = eval_scene_distance(scene, world_point + dz) - eval_scene_distance(scene, world_point - dz);
    return normalize({sx, sy, sz});
}

Vec3 edge_zero_crossing(const CompiledScene& scene, const Vec3& a, const Vec3& b,
                        double da, double db, int iterations)
{
    if (std::abs(da) <= 1.0e-9) {
        return a;
    }
    if (std::abs(db) <= 1.0e-9) {
        return b;
    }

    double lo = 0.0;
    double hi = 1.0;
    double vlo = da;
    for (int iter = 0; iter < iterations; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const Vec3 p = a + (b - a) * mid;
        const double vmid = eval_scene_distance(scene, p);
        const bool same_sign = (vmid <= 0.0) == (vlo <= 0.0);
        if (same_sign) {
            lo = mid;
            vlo = vmid;
        } else {
            hi = mid;
        }
    }
    return a + (b - a) * (0.5 * (lo + hi));
}

bool solve_qef(const QEFData& qef, Vec3& out)
{
    double aug[3][4] = {
        {qef.ata[0][0], qef.ata[0][1], qef.ata[0][2], qef.atb[0]},
        {qef.ata[1][0], qef.ata[1][1], qef.ata[1][2], qef.atb[1]},
        {qef.ata[2][0], qef.ata[2][1], qef.ata[2][2], qef.atb[2]},
    };

    const double trace = qef.ata[0][0] + qef.ata[1][1] + qef.ata[2][2];
    const double regularization = std::max(1.0e-8, trace * 1.0e-6);
    aug[0][0] += regularization;
    aug[1][1] += regularization;
    aug[2][2] += regularization;

    for (int pivot = 0; pivot < 3; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < 3; ++row) {
            if (std::abs(aug[row][pivot]) > std::abs(aug[best][pivot])) {
                best = row;
            }
        }
        if (std::abs(aug[best][pivot]) < 1.0e-10) {
            return false;
        }
        if (best != pivot) {
            for (int column = pivot; column < 4; ++column) {
                std::swap(aug[pivot][column], aug[best][column]);
            }
        }
        const double inv = 1.0 / aug[pivot][pivot];
        for (int column = pivot; column < 4; ++column) {
            aug[pivot][column] *= inv;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = aug[row][pivot];
            for (int column = pivot; column < 4; ++column) {
                aug[row][column] -= factor * aug[pivot][column];
            }
        }
    }

    out = {aug[0][3], aug[1][3], aug[2][3]};
    return true;
}

void accumulate_qef_plane(QEFData& qef, const Vec3& hit, const Vec3& normal)
{
    qef.centroid += hit;
    qef.count += 1;
    const double d = dot(normal, hit);
    qef.btb += d * d;
    qef.atb[0] += normal.x * d;
    qef.atb[1] += normal.y * d;
    qef.atb[2] += normal.z * d;
    qef.ata[0][0] += normal.x * normal.x;
    qef.ata[0][1] += normal.x * normal.y;
    qef.ata[0][2] += normal.x * normal.z;
    qef.ata[1][0] += normal.y * normal.x;
    qef.ata[1][1] += normal.y * normal.y;
    qef.ata[1][2] += normal.y * normal.z;
    qef.ata[2][0] += normal.z * normal.x;
    qef.ata[2][1] += normal.z * normal.y;
    qef.ata[2][2] += normal.z * normal.z;
}

double qef_error(const QEFData& qef, const Vec3& point)
{
    const double ax = qef.ata[0][0] * point.x + qef.ata[0][1] * point.y + qef.ata[0][2] * point.z;
    const double ay = qef.ata[1][0] * point.x + qef.ata[1][1] * point.y + qef.ata[1][2] * point.z;
    const double az = qef.ata[2][0] * point.x + qef.ata[2][1] * point.y + qef.ata[2][2] * point.z;
    return point.x * ax + point.y * ay + point.z * az
         - 2.0 * (point.x * qef.atb[0] + point.y * qef.atb[1] + point.z * qef.atb[2])
         + qef.btb;
}

Vec3 choose_dual_contour_vertex(const CompiledScene& scene,
                                const QEFData& qef,
                                const Vec3& cell_min,
                                const Vec3& cell_max,
                                double normal_epsilon)
{
    const Vec3 centroid = qef.centroid / static_cast<double>(std::max(qef.count, 1));
    const Vec3 cell_center = (cell_min + cell_max) * 0.5;

    std::vector<Vec3> candidates;
    candidates.reserve(5);
    candidates.push_back(clamp_vec(centroid, cell_min, cell_max));
    candidates.push_back(cell_center);

    Vec3 solved = centroid;
    if (solve_qef(qef, solved)) {
        candidates.push_back(clamp_vec(solved, cell_min, cell_max));
    }

    const Vec3 snap_from_centroid_normal = eval_scene_normal(scene, centroid, normal_epsilon);
    const double snap_from_centroid_distance = eval_scene_distance(scene, centroid);
    candidates.push_back(clamp_vec(centroid - snap_from_centroid_normal * snap_from_centroid_distance,
                                   cell_min, cell_max));

    const Vec3 snap_from_center_normal = eval_scene_normal(scene, cell_center, normal_epsilon);
    const double snap_from_center_distance = eval_scene_distance(scene, cell_center);
    candidates.push_back(clamp_vec(cell_center - snap_from_center_normal * snap_from_center_distance,
                                   cell_min, cell_max));

    Vec3 best = candidates.front();
    double best_error = qef_error(qef, best);
    for (const auto& candidate : candidates) {
        const double error = qef_error(qef, candidate);
        if (error < best_error) {
            best_error = error;
            best = candidate;
        }
    }
    return best;
}

struct MeshResult {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<std::uint32_t, 3>> triangles;
    std::array<int, 3> grid_dimensions{0, 0, 0};
    double cell_size = 0.0;
    int active_cells = 0;
};

struct QuantizedVertexKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const QuantizedVertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedVertexKeyHash {
    std::size_t operator()(const QuantizedVertexKey& key) const {
        std::size_t seed = static_cast<std::size_t>(key.x);
        seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.z) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::size_t grid_index(int x, int y, int z, int nx, int ny)
{
    return static_cast<std::size_t>((z * (ny + 1) + y) * (nx + 1) + x);
}

std::size_t cell_index(int x, int y, int z, int nx, int ny)
{
    return static_cast<std::size_t>((z * ny + y) * nx + x);
}

void append_oriented_triangle(std::vector<std::array<std::uint32_t, 3>>& triangles,
                              std::uint32_t a, std::uint32_t b, std::uint32_t c,
                              const std::vector<std::array<double, 3>>& vertices,
                              const Vec3& reference_normal)
{
    if (a == b || a == c || b == c) {
        return;
    }
    const Vec3 va{vertices[a][0], vertices[a][1], vertices[a][2]};
    const Vec3 vb{vertices[b][0], vertices[b][1], vertices[b][2]};
    const Vec3 vc{vertices[c][0], vertices[c][1], vertices[c][2]};
    const Vec3 tri_normal = cross(vb - va, vc - va);
    if (length_sq(tri_normal) <= 1.0e-16) {
        return;
    }
    if (dot(tri_normal, reference_normal) < 0.0) {
        triangles.push_back({a, c, b});
    } else {
        triangles.push_back({a, b, c});
    }
}

std::uint32_t vertex_index_for_position(
    const Vec3& position,
    double quant_step,
    std::unordered_map<QuantizedVertexKey, std::uint32_t, QuantizedVertexKeyHash>& vertex_map,
    std::vector<std::array<double, 3>>& vertices)
{
    const double step = std::max(quant_step, 1.0e-9);
    const QuantizedVertexKey key{
        static_cast<std::int64_t>(std::llround(position.x / step)),
        static_cast<std::int64_t>(std::llround(position.y / step)),
        static_cast<std::int64_t>(std::llround(position.z / step)),
    };
    const auto found = vertex_map.find(key);
    if (found != vertex_map.end()) {
        return found->second;
    }
    const std::uint32_t index = static_cast<std::uint32_t>(vertices.size());
    vertices.push_back({position.x, position.y, position.z});
    vertex_map.emplace(key, index);
    return index;
}

QEFData build_dual_contour_qef(const CompiledScene& scene,
                               const Vec3& cell_min,
                               const Vec3& cell_max,
                               double normal_epsilon,
                               int subdivisions,
                               int zero_crossing_iterations)
{
    QEFData qef;
    const int sample_count = subdivisions + 1;
    const Vec3 cell_step = (cell_max - cell_min) / static_cast<double>(subdivisions);
    const auto local_index = [sample_count](int x, int y, int z) {
        return static_cast<std::size_t>((z * sample_count + y) * sample_count + x);
    };

    std::vector<double> local_field(static_cast<std::size_t>(sample_count * sample_count * sample_count));
    std::vector<Vec3> local_points(static_cast<std::size_t>(sample_count * sample_count * sample_count));
    for (int z = 0; z < sample_count; ++z) {
        for (int y = 0; y < sample_count; ++y) {
            for (int x = 0; x < sample_count; ++x) {
                const Vec3 p{
                    cell_min.x + cell_step.x * static_cast<double>(x),
                    cell_min.y + cell_step.y * static_cast<double>(y),
                    cell_min.z + cell_step.z * static_cast<double>(z),
                };
                const std::size_t idx = local_index(x, y, z);
                local_points[idx] = p;
                local_field[idx] = eval_scene_distance(scene, p);
            }
        }
    }

    auto accumulate_edge = [&](int x0, int y0, int z0, int x1, int y1, int z1) {
        const std::size_t idx0 = local_index(x0, y0, z0);
        const std::size_t idx1 = local_index(x1, y1, z1);
        const double d0 = local_field[idx0];
        const double d1 = local_field[idx1];
        const bool s0 = d0 <= 0.0;
        const bool s1 = d1 <= 0.0;
        if (s0 == s1) {
            return;
        }
        const Vec3 hit = edge_zero_crossing(scene, local_points[idx0], local_points[idx1], d0, d1,
                                            zero_crossing_iterations);
        const Vec3 normal = eval_scene_normal(scene, hit, normal_epsilon);
        accumulate_qef_plane(qef, hit, normal);
    };

    for (int z = 0; z < sample_count; ++z) {
        for (int y = 0; y < sample_count; ++y) {
            for (int x = 0; x < subdivisions; ++x) {
                accumulate_edge(x, y, z, x + 1, y, z);
            }
        }
    }
    for (int z = 0; z < sample_count; ++z) {
        for (int y = 0; y < subdivisions; ++y) {
            for (int x = 0; x < sample_count; ++x) {
                accumulate_edge(x, y, z, x, y + 1, z);
            }
        }
    }
    for (int z = 0; z < subdivisions; ++z) {
        for (int y = 0; y < sample_count; ++y) {
            for (int x = 0; x < sample_count; ++x) {
                accumulate_edge(x, y, z, x, y, z + 1);
            }
        }
    }

    return qef;
}

MeshResult extract_dual_contour_impl(const std::vector<std::array<float, 4>>& primitive_rows,
                                     const std::vector<std::array<float, 4>>& polygon_rows,
                                     const std::vector<std::array<float, 4>>& warp_rows,
                                     const std::vector<std::array<float, 4>>& instruction_rows,
                                     const std::array<float, 3>& bounds_min_in,
                                     const std::array<float, 3>& bounds_max_in,
                                     int resolution)
{
    if (resolution < 2) {
        throw std::runtime_error("Resolution must be at least 2");
    }

    const CompiledScene scene = build_scene(primitive_rows, polygon_rows, warp_rows, instruction_rows);
    if (scene.primitives.empty() || scene.instructions.empty()) {
        return {};
    }

    Vec3 bounds_min{bounds_min_in[0], bounds_min_in[1], bounds_min_in[2]};
    Vec3 bounds_max{bounds_max_in[0], bounds_max_in[1], bounds_max_in[2]};
    Vec3 extent = bounds_max - bounds_min;
    const double max_extent = std::max(max_component(extent), 1.0e-4);
    const double padding = max_extent / static_cast<double>(resolution);
    bounds_min = bounds_min - Vec3{padding, padding, padding};
    bounds_max = bounds_max + Vec3{padding, padding, padding};
    extent = bounds_max - bounds_min;

    const double target_cell = std::max(max_component(extent) / static_cast<double>(resolution), 1.0e-5);
    const int nx = std::max(1, static_cast<int>(std::ceil(extent.x / target_cell)));
    const int ny = std::max(1, static_cast<int>(std::ceil(extent.y / target_cell)));
    const int nz = std::max(1, static_cast<int>(std::ceil(extent.z / target_cell)));
    const Vec3 step{extent.x / static_cast<double>(nx), extent.y / static_cast<double>(ny), extent.z / static_cast<double>(nz)};
    const double normal_epsilon = std::max(1.0e-4, min_component(step) * 0.25);
    const int hermite_subdivisions = kDualContourHermiteSubdivisions;
    const int zero_crossing_iterations = kDualContourZeroCrossingIterations;

    MeshResult result;
    result.grid_dimensions = {nx, ny, nz};
    result.cell_size = max_component(step);

    std::vector<double> field(static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny + 1) * static_cast<std::size_t>(nz + 1));
    for (int z = 0; z <= nz; ++z) {
        for (int y = 0; y <= ny; ++y) {
            for (int x = 0; x <= nx; ++x) {
                const Vec3 p{
                    bounds_min.x + step.x * static_cast<double>(x),
                    bounds_min.y + step.y * static_cast<double>(y),
                    bounds_min.z + step.z * static_cast<double>(z),
                };
                field[grid_index(x, y, z, nx, ny)] = eval_scene_distance(scene, p);
            }
        }
    }

    static constexpr int corner_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };
    static constexpr int edge_corners[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };

    const std::size_t cell_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    std::vector<CellVertex> cell_vertices(cell_count);
    std::vector<std::uint32_t> cell_to_vertex(cell_count, std::numeric_limits<std::uint32_t>::max());

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t cidx = cell_index(x, y, z, nx, ny);
                double corner_values[8];
                Vec3 corner_points[8];
                bool any_inside = false;
                bool any_outside = false;
                for (int corner = 0; corner < 8; ++corner) {
                    const int gx = x + corner_offsets[corner][0];
                    const int gy = y + corner_offsets[corner][1];
                    const int gz = z + corner_offsets[corner][2];
                    corner_values[corner] = field[grid_index(gx, gy, gz, nx, ny)];
                    any_inside = any_inside || (corner_values[corner] <= 0.0);
                    any_outside = any_outside || (corner_values[corner] > 0.0);
                    corner_points[corner] = {
                        bounds_min.x + step.x * static_cast<double>(gx),
                        bounds_min.y + step.y * static_cast<double>(gy),
                        bounds_min.z + step.z * static_cast<double>(gz),
                    };
                }
                if (!(any_inside && any_outside)) {
                    continue;
                }

                const Vec3 cell_min{
                    bounds_min.x + step.x * static_cast<double>(x),
                    bounds_min.y + step.y * static_cast<double>(y),
                    bounds_min.z + step.z * static_cast<double>(z),
                };
                const Vec3 cell_max = cell_min + step;

                QEFData qef = build_dual_contour_qef(scene, cell_min, cell_max,
                                                    normal_epsilon,
                                                    hermite_subdivisions,
                                                    zero_crossing_iterations);
                if (qef.count == 0) {
                    for (const auto& edge : edge_corners) {
                        const int c0 = edge[0];
                        const int c1 = edge[1];
                        const bool s0 = corner_values[c0] <= 0.0;
                        const bool s1 = corner_values[c1] <= 0.0;
                        if (s0 == s1) {
                            continue;
                        }
                        const Vec3 hit = edge_zero_crossing(scene, corner_points[c0], corner_points[c1], corner_values[c0], corner_values[c1], zero_crossing_iterations);
                        const Vec3 normal = eval_scene_normal(scene, hit, normal_epsilon);
                        accumulate_qef_plane(qef, hit, normal);
                    }
                }
                if (qef.count == 0) {
                    continue;
                }

                Vec3 vertex = choose_dual_contour_vertex(scene, qef, cell_min, cell_max, normal_epsilon);

                cell_vertices[cidx].active = true;
                cell_vertices[cidx].position = vertex;
                cell_vertices[cidx].normal = eval_scene_normal(scene, vertex, normal_epsilon);
                cell_to_vertex[cidx] = static_cast<std::uint32_t>(result.vertices.size());
                result.vertices.push_back({vertex.x, vertex.y, vertex.z});
                result.active_cells += 1;
            }
        }
    }

    auto emit_quad = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2, std::uint32_t i3,
                         const Vec3& reference_normal) {
        append_oriented_triangle(result.triangles, i0, i1, i2, result.vertices, reference_normal);
        append_oriented_triangle(result.triangles, i0, i2, i3, result.vertices, reference_normal);
    };

    for (int z = 1; z < nz; ++z) {
        for (int y = 1; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const bool s0 = field[grid_index(x, y, z, nx, ny)] <= 0.0;
                const bool s1 = field[grid_index(x + 1, y, z, nx, ny)] <= 0.0;
                if (s0 == s1) {
                    continue;
                }
                const std::size_t c0 = cell_index(x, y - 1, z - 1, nx, ny);
                const std::size_t c1 = cell_index(x, y, z - 1, nx, ny);
                const std::size_t c2 = cell_index(x, y, z, nx, ny);
                const std::size_t c3 = cell_index(x, y - 1, z, nx, ny);
                if (!(cell_vertices[c0].active && cell_vertices[c1].active && cell_vertices[c2].active && cell_vertices[c3].active)) {
                    continue;
                }
                const Vec3 ref = normalize(cell_vertices[c0].normal + cell_vertices[c1].normal + cell_vertices[c2].normal + cell_vertices[c3].normal);
                if (s0) {
                    emit_quad(cell_to_vertex[c0], cell_to_vertex[c1], cell_to_vertex[c2], cell_to_vertex[c3], ref);
                } else {
                    emit_quad(cell_to_vertex[c3], cell_to_vertex[c2], cell_to_vertex[c1], cell_to_vertex[c0], ref);
                }
            }
        }
    }

    for (int z = 1; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 1; x < nx; ++x) {
                const bool s0 = field[grid_index(x, y, z, nx, ny)] <= 0.0;
                const bool s1 = field[grid_index(x, y + 1, z, nx, ny)] <= 0.0;
                if (s0 == s1) {
                    continue;
                }
                const std::size_t c0 = cell_index(x - 1, y, z - 1, nx, ny);
                const std::size_t c1 = cell_index(x, y, z - 1, nx, ny);
                const std::size_t c2 = cell_index(x, y, z, nx, ny);
                const std::size_t c3 = cell_index(x - 1, y, z, nx, ny);
                if (!(cell_vertices[c0].active && cell_vertices[c1].active && cell_vertices[c2].active && cell_vertices[c3].active)) {
                    continue;
                }
                const Vec3 ref = normalize(cell_vertices[c0].normal + cell_vertices[c1].normal + cell_vertices[c2].normal + cell_vertices[c3].normal);
                if (s0) {
                    emit_quad(cell_to_vertex[c0], cell_to_vertex[c1], cell_to_vertex[c2], cell_to_vertex[c3], ref);
                } else {
                    emit_quad(cell_to_vertex[c3], cell_to_vertex[c2], cell_to_vertex[c1], cell_to_vertex[c0], ref);
                }
            }
        }
    }

    for (int z = 0; z < nz; ++z) {
        for (int y = 1; y < ny; ++y) {
            for (int x = 1; x < nx; ++x) {
                const bool s0 = field[grid_index(x, y, z, nx, ny)] <= 0.0;
                const bool s1 = field[grid_index(x, y, z + 1, nx, ny)] <= 0.0;
                if (s0 == s1) {
                    continue;
                }
                const std::size_t c0 = cell_index(x - 1, y - 1, z, nx, ny);
                const std::size_t c1 = cell_index(x, y - 1, z, nx, ny);
                const std::size_t c2 = cell_index(x, y, z, nx, ny);
                const std::size_t c3 = cell_index(x - 1, y, z, nx, ny);
                if (!(cell_vertices[c0].active && cell_vertices[c1].active && cell_vertices[c2].active && cell_vertices[c3].active)) {
                    continue;
                }
                const Vec3 ref = normalize(cell_vertices[c0].normal + cell_vertices[c1].normal + cell_vertices[c2].normal + cell_vertices[c3].normal);
                if (s0) {
                    emit_quad(cell_to_vertex[c0], cell_to_vertex[c1], cell_to_vertex[c2], cell_to_vertex[c3], ref);
                } else {
                    emit_quad(cell_to_vertex[c3], cell_to_vertex[c2], cell_to_vertex[c1], cell_to_vertex[c0], ref);
                }
            }
        }
    }

    return result;
}

MeshResult extract_iso_simplex_impl(const std::vector<std::array<float, 4>>& primitive_rows,
                                    const std::vector<std::array<float, 4>>& polygon_rows,
                                    const std::vector<std::array<float, 4>>& warp_rows,
                                    const std::vector<std::array<float, 4>>& instruction_rows,
                                    const std::array<float, 3>& bounds_min_in,
                                    const std::array<float, 3>& bounds_max_in,
                                    int resolution)
{
    if (resolution < 2) {
        throw std::runtime_error("Resolution must be at least 2");
    }

    const CompiledScene scene = build_scene(primitive_rows, polygon_rows, warp_rows, instruction_rows);
    if (scene.primitives.empty() || scene.instructions.empty()) {
        return {};
    }

    Vec3 bounds_min{bounds_min_in[0], bounds_min_in[1], bounds_min_in[2]};
    Vec3 bounds_max{bounds_max_in[0], bounds_max_in[1], bounds_max_in[2]};
    Vec3 extent = bounds_max - bounds_min;
    const double max_extent = std::max(max_component(extent), 1.0e-4);
    const double padding = max_extent / static_cast<double>(resolution);
    bounds_min = bounds_min - Vec3{padding, padding, padding};
    bounds_max = bounds_max + Vec3{padding, padding, padding};
    extent = bounds_max - bounds_min;

    const double target_cell = std::max(max_component(extent) / static_cast<double>(resolution), 1.0e-5);
    const int nx = std::max(1, static_cast<int>(std::ceil(extent.x / target_cell)));
    const int ny = std::max(1, static_cast<int>(std::ceil(extent.y / target_cell)));
    const int nz = std::max(1, static_cast<int>(std::ceil(extent.z / target_cell)));
    const Vec3 step{extent.x / static_cast<double>(nx), extent.y / static_cast<double>(ny), extent.z / static_cast<double>(nz)};
    const double normal_epsilon = std::max(1.0e-4, min_component(step) * 0.25);
    const double quant_step = std::max(1.0e-6, max_component(step) * 1.0e-5);

    MeshResult result;
    result.grid_dimensions = {nx, ny, nz};
    result.cell_size = max_component(step);

    std::vector<double> field(static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny + 1) * static_cast<std::size_t>(nz + 1));
    for (int z = 0; z <= nz; ++z) {
        for (int y = 0; y <= ny; ++y) {
            for (int x = 0; x <= nx; ++x) {
                const Vec3 p{
                    bounds_min.x + step.x * static_cast<double>(x),
                    bounds_min.y + step.y * static_cast<double>(y),
                    bounds_min.z + step.z * static_cast<double>(z),
                };
                field[grid_index(x, y, z, nx, ny)] = eval_scene_distance(scene, p);
            }
        }
    }

    static constexpr int corner_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };
    static constexpr int tetrahedra[6][4] = {
        {0, 5, 1, 6},
        {0, 1, 2, 6},
        {0, 2, 3, 6},
        {0, 3, 7, 6},
        {0, 7, 4, 6},
        {0, 4, 5, 6},
    };

    std::unordered_map<QuantizedVertexKey, std::uint32_t, QuantizedVertexKeyHash> vertex_map;
    vertex_map.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double corner_values[8];
                Vec3 corner_points[8];
                bool cell_active = false;
                for (int corner = 0; corner < 8; ++corner) {
                    const int gx = x + corner_offsets[corner][0];
                    const int gy = y + corner_offsets[corner][1];
                    const int gz = z + corner_offsets[corner][2];
                    corner_values[corner] = field[grid_index(gx, gy, gz, nx, ny)];
                    corner_points[corner] = {
                        bounds_min.x + step.x * static_cast<double>(gx),
                        bounds_min.y + step.y * static_cast<double>(gy),
                        bounds_min.z + step.z * static_cast<double>(gz),
                    };
                }

                for (const auto& tetra : tetrahedra) {
                    int inside_indices[4];
                    int outside_indices[4];
                    int inside_count = 0;
                    int outside_count = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int corner = tetra[i];
                        if (corner_values[corner] <= 0.0) {
                            inside_indices[inside_count++] = corner;
                        } else {
                            outside_indices[outside_count++] = corner;
                        }
                    }
                    if (inside_count == 0 || inside_count == 4) {
                        continue;
                    }

                    cell_active = true;
                    if (inside_count == 1 || inside_count == 3) {
                        const bool invert = inside_count == 3;
                        const int tip = invert ? outside_indices[0] : inside_indices[0];
                        const int* others = invert ? inside_indices : outside_indices;
                        Vec3 hits[3];
                        Vec3 normal_sum;
                        for (int i = 0; i < 3; ++i) {
                            hits[i] = edge_zero_crossing(scene,
                                                         corner_points[tip],
                                                         corner_points[others[i]],
                                                         corner_values[tip],
                                                         corner_values[others[i]],
                                                         kDualContourZeroCrossingIterations);
                            normal_sum += eval_scene_normal(scene, hits[i], normal_epsilon);
                        }
                        const Vec3 ref = normalize(normal_sum);
                        const auto i0 = vertex_index_for_position(hits[0], quant_step, vertex_map, result.vertices);
                        const auto i1 = vertex_index_for_position(hits[1], quant_step, vertex_map, result.vertices);
                        const auto i2 = vertex_index_for_position(hits[2], quant_step, vertex_map, result.vertices);
                        append_oriented_triangle(result.triangles, i0, i1, i2, result.vertices, ref);
                        continue;
                    }

                    Vec3 hits[4];
                    Vec3 normal_sum;
                    hits[0] = edge_zero_crossing(scene,
                                                 corner_points[inside_indices[0]],
                                                 corner_points[outside_indices[0]],
                                                 corner_values[inside_indices[0]],
                                                 corner_values[outside_indices[0]],
                                                 kDualContourZeroCrossingIterations);
                    hits[1] = edge_zero_crossing(scene,
                                                 corner_points[inside_indices[0]],
                                                 corner_points[outside_indices[1]],
                                                 corner_values[inside_indices[0]],
                                                 corner_values[outside_indices[1]],
                                                 kDualContourZeroCrossingIterations);
                    hits[2] = edge_zero_crossing(scene,
                                                 corner_points[inside_indices[1]],
                                                 corner_points[outside_indices[0]],
                                                 corner_values[inside_indices[1]],
                                                 corner_values[outside_indices[0]],
                                                 kDualContourZeroCrossingIterations);
                    hits[3] = edge_zero_crossing(scene,
                                                 corner_points[inside_indices[1]],
                                                 corner_points[outside_indices[1]],
                                                 corner_values[inside_indices[1]],
                                                 corner_values[outside_indices[1]],
                                                 kDualContourZeroCrossingIterations);
                    for (const auto& hit : hits) {
                        normal_sum += eval_scene_normal(scene, hit, normal_epsilon);
                    }
                    const Vec3 ref = normalize(normal_sum);
                    const auto i0 = vertex_index_for_position(hits[0], quant_step, vertex_map, result.vertices);
                    const auto i1 = vertex_index_for_position(hits[1], quant_step, vertex_map, result.vertices);
                    const auto i2 = vertex_index_for_position(hits[2], quant_step, vertex_map, result.vertices);
                    const auto i3 = vertex_index_for_position(hits[3], quant_step, vertex_map, result.vertices);
                    append_oriented_triangle(result.triangles, i0, i1, i3, result.vertices, ref);
                    append_oriented_triangle(result.triangles, i0, i3, i2, result.vertices, ref);
                }

                if (cell_active) {
                    result.active_cells += 1;
                }
            }
        }
    }

    return result;
}

std::vector<std::array<float, 4>> parse_rows(PyObject* obj, const char* name)
{
    PyObject* rows = PySequence_Fast(obj, name);
    if (rows == nullptr) {
        throw std::runtime_error(std::string(name) + " must be a sequence");
    }

    const Py_ssize_t row_count = PySequence_Fast_GET_SIZE(rows);
    PyObject** row_items = PySequence_Fast_ITEMS(rows);
    std::vector<std::array<float, 4>> out;
    out.reserve(static_cast<std::size_t>(row_count));
    for (Py_ssize_t row_index = 0; row_index < row_count; ++row_index) {
        PyObject* row_obj = PySequence_Fast(row_items[row_index], name);
        if (row_obj == nullptr) {
            Py_DECREF(rows);
            throw std::runtime_error(std::string(name) + " row must be a sequence of four floats");
        }
        if (PySequence_Fast_GET_SIZE(row_obj) != 4) {
            Py_DECREF(row_obj);
            Py_DECREF(rows);
            throw std::runtime_error(std::string(name) + " row must have four elements");
        }
        PyObject** values = PySequence_Fast_ITEMS(row_obj);
        std::array<float, 4> row{};
        for (int i = 0; i < 4; ++i) {
            row[static_cast<std::size_t>(i)] = static_cast<float>(PyFloat_AsDouble(values[i]));
            if (PyErr_Occurred()) {
                Py_DECREF(row_obj);
                Py_DECREF(rows);
                throw std::runtime_error(std::string(name) + " row contains a non-numeric value");
            }
        }
        out.push_back(row);
        Py_DECREF(row_obj);
    }
    Py_DECREF(rows);
    return out;
}

std::array<float, 3> parse_bounds(PyObject* obj, const char* name)
{
    PyObject* seq = PySequence_Fast(obj, name);
    if (seq == nullptr) {
        throw std::runtime_error(std::string(name) + " must be a sequence of three floats");
    }
    if (PySequence_Fast_GET_SIZE(seq) != 3) {
        Py_DECREF(seq);
        throw std::runtime_error(std::string(name) + " must have three elements");
    }
    PyObject** values = PySequence_Fast_ITEMS(seq);
    std::array<float, 3> out{};
    for (int i = 0; i < 3; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>(PyFloat_AsDouble(values[i]));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            throw std::runtime_error(std::string(name) + " contains a non-numeric value");
        }
    }
    Py_DECREF(seq);
    return out;
}

PyObject* mesh_result_to_python(const MeshResult& mesh)
{
    PyObject* result = PyDict_New();
    PyObject* vertices = PyList_New(static_cast<Py_ssize_t>(mesh.vertices.size()));
    PyObject* triangles = PyList_New(static_cast<Py_ssize_t>(mesh.triangles.size()));
    PyObject* grid_dimensions = PyTuple_New(3);
    if (result == nullptr || vertices == nullptr || triangles == nullptr || grid_dimensions == nullptr) {
        Py_XDECREF(result);
        Py_XDECREF(vertices);
        Py_XDECREF(triangles);
        Py_XDECREF(grid_dimensions);
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(mesh.vertices.size()); ++i) {
        const auto& vertex = mesh.vertices[static_cast<std::size_t>(i)];
        PyObject* tuple = Py_BuildValue("(ddd)", vertex[0], vertex[1], vertex[2]);
        if (tuple == nullptr) {
            Py_DECREF(result);
            Py_DECREF(vertices);
            Py_DECREF(triangles);
            Py_DECREF(grid_dimensions);
            return nullptr;
        }
        PyList_SET_ITEM(vertices, i, tuple);
    }

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(mesh.triangles.size()); ++i) {
        const auto& tri = mesh.triangles[static_cast<std::size_t>(i)];
        PyObject* tuple = Py_BuildValue("(III)", tri[0], tri[1], tri[2]);
        if (tuple == nullptr) {
            Py_DECREF(result);
            Py_DECREF(vertices);
            Py_DECREF(triangles);
            Py_DECREF(grid_dimensions);
            return nullptr;
        }
        PyList_SET_ITEM(triangles, i, tuple);
    }

    for (int axis = 0; axis < 3; ++axis) {
        PyObject* value = PyLong_FromLong(mesh.grid_dimensions[static_cast<std::size_t>(axis)]);
        if (value == nullptr) {
            Py_DECREF(result);
            Py_DECREF(vertices);
            Py_DECREF(triangles);
            Py_DECREF(grid_dimensions);
            return nullptr;
        }
        PyTuple_SET_ITEM(grid_dimensions, axis, value);
    }

    if (PyDict_SetItemString(result, "vertices", vertices) < 0 ||
        PyDict_SetItemString(result, "triangles", triangles) < 0 ||
        PyDict_SetItemString(result, "grid_dimensions", grid_dimensions) < 0) {
        Py_DECREF(result);
        Py_DECREF(vertices);
        Py_DECREF(triangles);
        Py_DECREF(grid_dimensions);
        return nullptr;
    }
    Py_DECREF(vertices);
    Py_DECREF(triangles);
    Py_DECREF(grid_dimensions);

    PyObject* cell_size = PyFloat_FromDouble(mesh.cell_size);
    PyObject* active_cells = PyLong_FromLong(mesh.active_cells);
    if (cell_size == nullptr || active_cells == nullptr) {
        Py_DECREF(result);
        Py_XDECREF(cell_size);
        Py_XDECREF(active_cells);
        return nullptr;
    }
    if (PyDict_SetItemString(result, "cell_size", cell_size) < 0 ||
        PyDict_SetItemString(result, "active_cells", active_cells) < 0) {
        Py_DECREF(result);
        Py_DECREF(cell_size);
        Py_DECREF(active_cells);
        return nullptr;
    }
    Py_DECREF(cell_size);
    Py_DECREF(active_cells);
    return result;
}

PyObject* py_extract_dual_contour_mesh(PyObject*, PyObject* args)
{
    PyObject* primitive_rows_obj = nullptr;
    PyObject* polygon_rows_obj = nullptr;
    PyObject* warp_rows_obj = nullptr;
    PyObject* instruction_rows_obj = nullptr;
    PyObject* bounds_min_obj = nullptr;
    PyObject* bounds_max_obj = nullptr;
    int resolution = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOi", &primitive_rows_obj, &polygon_rows_obj, &warp_rows_obj,
                          &instruction_rows_obj, &bounds_min_obj, &bounds_max_obj,
                          &resolution)) {
        return nullptr;
    }

    try {
        const auto primitive_rows = parse_rows(primitive_rows_obj, "primitive_rows");
        const auto polygon_rows = parse_rows(polygon_rows_obj, "polygon_rows");
        const auto warp_rows = parse_rows(warp_rows_obj, "warp_rows");
        const auto instruction_rows = parse_rows(instruction_rows_obj, "instruction_rows");
        const auto bounds_min = parse_bounds(bounds_min_obj, "bounds_min");
        const auto bounds_max = parse_bounds(bounds_max_obj, "bounds_max");

        PyThreadState* state = PyEval_SaveThread();
        MeshResult mesh;
        try {
            mesh = extract_dual_contour_impl(primitive_rows, polygon_rows, warp_rows, instruction_rows,
                                             bounds_min, bounds_max, resolution);
        } catch (...) {
            PyEval_RestoreThread(state);
            throw;
        }
        PyEval_RestoreThread(state);
        return mesh_result_to_python(mesh);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

PyObject* py_extract_iso_simplex_mesh(PyObject*, PyObject* args)
{
    PyObject* primitive_rows_obj = nullptr;
    PyObject* polygon_rows_obj = nullptr;
    PyObject* warp_rows_obj = nullptr;
    PyObject* instruction_rows_obj = nullptr;
    PyObject* bounds_min_obj = nullptr;
    PyObject* bounds_max_obj = nullptr;
    int resolution = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOi", &primitive_rows_obj, &polygon_rows_obj, &warp_rows_obj,
                          &instruction_rows_obj, &bounds_min_obj, &bounds_max_obj,
                          &resolution)) {
        return nullptr;
    }

    try {
        const auto primitive_rows = parse_rows(primitive_rows_obj, "primitive_rows");
        const auto polygon_rows = parse_rows(polygon_rows_obj, "polygon_rows");
        const auto warp_rows = parse_rows(warp_rows_obj, "warp_rows");
        const auto instruction_rows = parse_rows(instruction_rows_obj, "instruction_rows");
        const auto bounds_min = parse_bounds(bounds_min_obj, "bounds_min");
        const auto bounds_max = parse_bounds(bounds_max_obj, "bounds_max");

        PyThreadState* state = PyEval_SaveThread();
        MeshResult mesh;
        try {
            mesh = extract_iso_simplex_impl(primitive_rows, polygon_rows, warp_rows, instruction_rows,
                                            bounds_min, bounds_max, resolution);
        } catch (...) {
            PyEval_RestoreThread(state);
            throw;
        }
        PyEval_RestoreThread(state);
        return mesh_result_to_python(mesh);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

PyMethodDef kModuleMethods[] = {
    {"extract_dual_contour_mesh", py_extract_dual_contour_mesh, METH_VARARGS,
     "Extract a dual contour mesh from MathOPS compiled scene rows."},
    {"extract_iso_simplex_mesh", py_extract_iso_simplex_mesh, METH_VARARGS,
     "Extract an iso-simplex mesh from MathOPS compiled scene rows."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "mathops_meshing_native",
    "MathOPS CPU meshing helpers",
    -1,
    kModuleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit_mathops_meshing_native(void)
{
    return PyModule_Create(&kModuleDef);
}
