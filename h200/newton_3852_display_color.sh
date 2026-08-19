#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/workspace/.uv-bin:$PATH"

FORK_URL="https://github.com/Official-Space-AI/newton.git"
EXPECTED_BRANCH="kms8720/deformable-display-color"
EXPECTED_SHA="0ec11e6277f7f2282e5e2a4b87b8a05f18f92d50"

if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: EXPECTED_SHA must be a full 40-character lowercase commit SHA." >&2
  exit 2
fi

for command_name in git uv nvidia-smi sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "ERROR: required command is missing: $command_name" >&2
    exit 127
  fi
done

RUN_TAG="newton_3852_${EXPECTED_SHA:0:8}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/workspace/${RUN_TAG}"
SRC="${RUN_DIR}/src"
VENV="${RUN_DIR}/venv"
PY="${VENV}/bin/python"
WARP_CACHE_PATH="${RUN_DIR}/warp-cache"

LOG="${RUN_DIR}/h200.log"
PROBE="${RUN_DIR}/probe.py"
ARTIFACT="${RUN_DIR}/newton_3852_h200.npz"
RESULT_JSON="${RUN_DIR}/result.json"
PACKAGES="${RUN_DIR}/packages.txt"
GPU_INFO="${RUN_DIR}/gpu.txt"
GIT_INFO="${RUN_DIR}/git.txt"
HASHES="${RUN_DIR}/sha256.txt"

mkdir -p "$RUN_DIR"

export EXPECTED_SHA EXPECTED_BRANCH SRC VENV ARTIFACT RESULT_JSON
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export WARP_CACHE_PATH
unset PYTHONPATH || true

set +e
(
  set -Eeuo pipefail

  git init -q "$SRC"
  git -C "$SRC" remote add origin "$FORK_URL"
  git -C "$SRC" fetch --depth=1 origin \
    "refs/heads/${EXPECTED_BRANCH}:refs/remotes/origin/${EXPECTED_BRANCH}"

  REMOTE_SHA="$(git -C "$SRC" rev-parse "refs/remotes/origin/${EXPECTED_BRANCH}")"
  test "$REMOTE_SHA" = "$EXPECTED_SHA"

  git -C "$SRC" checkout --detach "$EXPECTED_SHA"
  test "$(git -C "$SRC" rev-parse HEAD)" = "$EXPECTED_SHA"
  test -z "$(git -C "$SRC" status --porcelain --untracked-files=all)"

  {
    echo "expected_branch=${EXPECTED_BRANCH}"
    echo "expected_sha=${EXPECTED_SHA}"
    echo "remote_sha=${REMOTE_SHA}"
    echo "head=$(git -C "$SRC" rev-parse HEAD)"
    echo "tree=$(git -C "$SRC" rev-parse HEAD^{tree})"
    echo "origin=$(git -C "$SRC" remote get-url origin)"
    git -C "$SRC" show -s --format='commit=%H%ncommit_time=%cI%nsubject=%s'
    echo "pre_probe_status=$(git -C "$SRC" status --porcelain --untracked-files=all)"
  } | tee "$GIT_INFO"

  {
    nvidia-smi -L
    nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap \
      --format=csv,noheader
  } | tee "$GPU_INFO"

  GPU0_NAME="$(
    nvidia-smi -i 0 --query-gpu=name --format=csv,noheader |
      tr -d '\r' |
      sed 's/[[:space:]]*$//'
  )"
  test "$GPU0_NAME" = "NVIDIA H200"

  uv venv --python 3.12 "$VENV"
  UV_PROJECT_ENVIRONMENT="$VENV" \
    uv sync \
      --project "$SRC" \
      --python "$PY" \
      --frozen \
      --no-dev

  uv pip freeze --python "$PY" | tee "$PACKAGES"

  tee "$PROBE" >/dev/null <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path

import numpy as np


expected_sha = os.environ["EXPECTED_SHA"]
expected_branch = os.environ["EXPECTED_BRANCH"]
source_root = Path(os.environ["SRC"]).resolve()
venv_root = Path(os.environ["VENV"]).resolve()
artifact_path = Path(os.environ["ARTIFACT"]).resolve()
result_path = Path(os.environ["RESULT_JSON"]).resolve()


def atomic_npz(path: Path, **values) -> None:
    tmp = path.with_name(path.name + ".tmp")
    prepared = {key: np.asarray(value) for key, value in values.items()}
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **prepared)
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


running = {
    "pass": False,
    "status": "running",
    "commit": expected_sha,
    "branch": expected_branch,
}
atomic_json(result_path, running)
atomic_npz(
    artifact_path,
    passed=False,
    status="running",
    commit=expected_sha,
    branch=expected_branch,
)

started = time.perf_counter()

try:
    import warp as wp

    import newton
    from newton._src.sensors.warp_raytrace.raytrace import (
        PARTICLES_SHAPE_ID,
        TRIANGLE_MESH_SHAPE_ID,
    )
    from newton._src.viewer.gl.opengl import RenderVertex, fill_vertex_data
    from newton.sensors import SensorTiledCamera
    from newton.viewer import ViewerNull

    def git_output(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source_root), *args],
            text=True,
        ).strip()

    commit = git_output("rev-parse", "HEAD")
    tree = git_output("rev-parse", "HEAD^{tree}")
    if commit != expected_sha:
        raise AssertionError(f"commit mismatch: {commit} != {expected_sha}")

    newton_origin = Path(newton.__file__).resolve()
    expected_newton_origin = (source_root / "newton" / "__init__.py").resolve()
    if newton_origin != expected_newton_origin:
        raise AssertionError(
            f"Newton import shadowed: {newton_origin} != {expected_newton_origin}"
        )

    warp_origin = Path(wp.__file__).resolve()
    warp_distribution_root = Path(
        metadata.distribution("warp-lang").locate_file("")
    ).resolve()
    if not warp_origin.is_relative_to(venv_root):
        raise AssertionError(f"Warp is outside isolated venv: {warp_origin}")
    if not warp_origin.is_relative_to(warp_distribution_root):
        raise AssertionError(
            f"Warp origin is not owned by its distribution: {warp_origin}"
        )

    with (source_root / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    def locked_version(package_name: str) -> str:
        versions = {
            package["version"]
            for package in lock["package"]
            if package.get("name") == package_name and "version" in package
        }
        if len(versions) != 1:
            raise AssertionError(
                f"expected one locked {package_name} version, got {versions}"
            )
        return versions.pop()

    loaded_newton_version = str(newton.__version__)
    metadata_newton_version = metadata.version("newton")
    locked_newton_version = locked_version("newton")
    if not (
        loaded_newton_version
        == metadata_newton_version
        == locked_newton_version
    ):
        raise AssertionError(
            "Newton version mismatch: "
            f"loaded={loaded_newton_version}, "
            f"metadata={metadata_newton_version}, "
            f"lock={locked_newton_version}"
        )

    loaded_warp_version = str(wp.__version__)
    metadata_warp_version = metadata.version("warp-lang")
    locked_warp_version = locked_version("warp-lang")
    if not (
        loaded_warp_version
        == metadata_warp_version
        == locked_warp_version
    ):
        raise AssertionError(
            "Warp version mismatch: "
            f"loaded={loaded_warp_version}, "
            f"metadata={metadata_warp_version}, "
            f"lock={locked_warp_version}"
        )

    wp.init()
    if not wp.is_cuda_available():
        raise AssertionError("Warp CUDA is unavailable")

    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise AssertionError(f"cuda:0 is not a CUDA device: {device}")
    if device.alias != "cuda:0" or device.ordinal != 0:
        raise AssertionError(
            f"unexpected CUDA alias/ordinal: {device.alias}/{device.ordinal}"
        )
    if device.name != "NVIDIA H200":
        raise AssertionError(f"expected exact NVIDIA H200, got {device.name!r}")
    if int(device.arch) != 90:
        raise AssertionError(
            f"expected H200 compute capability 9.0, got arch={device.arch}"
        )

    wp.set_device(device)

    checks: dict[str, bool] = {}
    observed: dict[str, object] = {}
    artifact_arrays: dict[str, np.ndarray] = {}
    launch_counter = {"count": 0}

    def check(name: str) -> None:
        checks[name] = True

    def require_device(value, label: str) -> None:
        if value is None:
            raise AssertionError(f"{label} is None")
        if value.device != device:
            raise AssertionError(
                f"{label} device mismatch: {value.device} != {device}"
            )

    def require_model_device(model, label: str) -> None:
        if model.device != device:
            raise AssertionError(
                f"{label}.device mismatch: {model.device} != {device}"
            )
        for attribute_name in (
            "particle_q",
            "particle_radius",
            "particle_flags",
            "particle_world",
            "particle_display_color",
            "tri_indices",
        ):
            value = getattr(model, attribute_name, None)
            if value is not None:
                require_device(value, f"{label}.{attribute_name}")

    def unpack_rgba(image) -> np.ndarray:
        packed = np.asarray(image.numpy(), dtype=np.uint32)
        return np.stack(
            (
                packed & 0xFF,
                (packed >> 8) & 0xFF,
                (packed >> 16) & 0xFF,
                (packed >> 24) & 0xFF,
            ),
            axis=-1,
        ).astype(np.uint8)

    def srgb_to_linear_literal(display_rgb) -> np.ndarray:
        rgb = np.asarray(display_rgb, dtype=np.float64)
        return np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)

    def render_center(
        model,
        config,
        *,
        with_normal: bool,
        camera_transforms=None,
    ) -> dict[str, np.ndarray | None]:
        sensor = SensorTiledCamera(
            model=model,
            default_render_config=config,
        )

        if camera_transforms is None:
            camera_transforms = wp.array(
                [[
                    wp.transformf(
                        wp.vec3f(0.0),
                        wp.quatf(0.0, 0.0, 0.0, 1.0),
                    )
                ]],
                dtype=wp.transformf,
                device=device,
            )

        camera_rays = sensor.utils.compute_camera_rays_pinhole(
            1,
            1,
            camera_fovs=math.radians(30.0),
        )
        albedo_image = sensor.utils.create_albedo_image_output(1, 1)
        shape_index_image = sensor.utils.create_shape_index_image_output(1, 1)
        normal_image = (
            sensor.utils.create_normal_image_output(1, 1)
            if with_normal
            else None
        )

        state = model.state()
        require_device(state.particle_q, "state.particle_q")
        require_device(camera_transforms, "camera_transforms")
        require_device(camera_rays, "camera_rays")
        require_device(albedo_image, "albedo_image")
        require_device(shape_index_image, "shape_index_image")
        if normal_image is not None:
            require_device(normal_image, "normal_image")

        sensor.update(
            state,
            camera_transforms,
            camera_rays,
            albedo_image=albedo_image,
            shape_index_image=shape_index_image,
            normal_image=normal_image,
        )
        wp.synchronize_device(device)
        launch_counter["count"] += 1

        return {
            "rgba": unpack_rgba(albedo_image),
            "shape": np.asarray(shape_index_image.numpy(), dtype=np.uint32),
            "normal": (
                np.asarray(normal_image.numpy(), dtype=np.float32)
                if normal_image is not None
                else None
            ),
        }

    def verify_pixel(
        rendered: dict[str, np.ndarray | None],
        index: tuple[int, int, int, int],
        expected_rgb,
        expected_shape: int,
        label: str,
    ) -> None:
        rgba = rendered["rgba"]
        shape = rendered["shape"]
        assert rgba is not None
        assert shape is not None

        pixel = rgba[index]
        np.testing.assert_allclose(
            pixel[:3],
            np.asarray(expected_rgb, dtype=np.float32) * 255.0,
            rtol=0.0,
            atol=1.1,
            err_msg=label,
        )
        if int(pixel[3]) != 255:
            raise AssertionError(f"{label}: alpha={pixel[3]}")
        if int(shape[index]) != expected_shape:
            raise AssertionError(
                f"{label}: shape={int(shape[index])}, expected={expected_shape}"
            )

        normal = rendered["normal"]
        if normal is not None:
            value = normal[index]
            if not np.all(np.isfinite(value)):
                raise AssertionError(f"{label}: non-finite normal {value}")
            if float(np.linalg.norm(value)) < 0.9:
                raise AssertionError(f"{label}: invalid normal {value}")

        observed[label] = {
            "rgba": pixel.tolist(),
            "shape": int(shape[index]),
        }
        check(label)

    # Builder/Model authoring and partial-white fill.
    author_builder = newton.ModelBuilder()
    author_builder.add_particle(
        wp.vec3(0.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
        color=(1.0, 0.0, 0.0),
    )
    author_builder.add_particle(
        wp.vec3(1.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
    )
    author_builder.add_particles(
        pos=[wp.vec3(2.0, 0.0, -2.0), wp.vec3(3.0, 0.0, -2.0)],
        vel=[wp.vec3(), wp.vec3()],
        mass=[1.0, 1.0],
        colors=[(0.0, 1.0, 0.0), None],
    )

    if author_builder.particle_display_color[1] is not None:
        raise AssertionError("unspecified single-particle color was not None")
    if author_builder.particle_display_color[3] is not None:
        raise AssertionError("unspecified bulk color was not None")

    author_model = author_builder.finalize(device=device)
    require_model_device(author_model, "author_model")
    expected_partial = np.asarray(
        [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float32,
    )
    actual_partial = author_model.particle_display_color.numpy()
    np.testing.assert_allclose(actual_partial, expected_partial)
    if author_model.particle_colors.dtype != wp.int32:
        raise AssertionError(
            "graph-color storage no longer uses int32 independently from display color"
        )
    artifact_arrays["builder_partial_colors"] = actual_partial
    check("builder_partial_white_fill")

    malformed_builder = newton.ModelBuilder()
    malformed_builder.add_particles(
        pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)],
        vel=[wp.vec3(), wp.vec3()],
        mass=[1.0, 1.0],
        colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )
    malformed_builder.particle_display_color.pop()
    try:
        malformed_builder.finalize(device=device)
    except ValueError as exc:
        validation_message = str(exc)
        expected_fragments = (
            "particle_display_color",
            "length 1",
            "expected 2",
            "particle_count",
        )
        if not all(fragment in validation_message for fragment in expected_fragments):
            raise AssertionError(
                f"unexpected structure-validation error: {validation_message}"
            ) from exc
    else:
        raise AssertionError(
            "finalize accepted mismatched particle_display_color length"
        )
    observed["builder_structure_validation"] = {
        "failure_type": "ValueError",
        "message": validation_message,
        "particle_count": 2,
        "display_color_count": 1,
        "requested_device": device.alias,
    }
    artifact_arrays["builder_structure_validation_counts"] = np.asarray(
        [2, 1], dtype=np.int32
    )
    check("builder_rejects_mismatched_display_color_length")

    # High-level deformable helper propagation.
    cloth_color = np.asarray((0.2, 0.4, 0.6), dtype=np.float32)
    cloth_builder = newton.ModelBuilder()
    cloth_builder.add_cloth_mesh(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(),
        vertices=[
            (-1.0, -1.0, -2.0),
            (1.0, -1.0, -2.0),
            (0.0, 1.0, -2.0),
        ],
        indices=[0, 1, 2],
        density=1.0,
        color=tuple(cloth_color),
    )
    cloth_model = cloth_builder.finalize(device=device)
    require_model_device(cloth_model, "cloth_model")
    cloth_colors = cloth_model.particle_display_color.numpy()
    np.testing.assert_allclose(cloth_colors, np.tile(cloth_color, (3, 1)))
    artifact_arrays["cloth_helper_colors"] = cloth_colors
    check("cloth_helper_color_authoring")

    # CUDA GL vertex packing: ABI layout, authored color, and None -> white.
    gl_points = wp.array(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        dtype=wp.vec3,
        device=device,
    )
    gl_normals = wp.array(
        [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        dtype=wp.vec3,
        device=device,
    )
    gl_uvs = wp.array(
        [(0.0, 0.0), (1.0, 0.0)],
        dtype=wp.vec2,
        device=device,
    )
    gl_colors = wp.array(
        [(1.0, 0.25, 0.0), (0.0, 0.5, 1.0)],
        dtype=wp.vec3,
        device=device,
    )
    gl_vertices = wp.zeros(2, dtype=RenderVertex, device=device)
    for value, label in (
        (gl_points, "gl_points"),
        (gl_normals, "gl_normals"),
        (gl_uvs, "gl_uvs"),
        (gl_colors, "gl_colors"),
        (gl_vertices, "gl_vertices"),
    ):
        require_device(value, label)

    wp.launch(
        fill_vertex_data,
        dim=2,
        inputs=[gl_points, gl_normals, gl_uvs, gl_colors],
        outputs=[gl_vertices],
        device=device,
    )
    wp.synchronize_device(device)
    # Copy immediately: CPU Warp arrays may expose a shared NumPy view, and the
    # fallback launch below must not overwrite the authored-color evidence.
    authored_vertices = gl_vertices.numpy().copy()
    vertex_dtype = authored_vertices.dtype
    vertex_stride_bytes = int(vertex_dtype.itemsize)
    vertex_color_offset_bytes = int(vertex_dtype.fields["color"][1])
    if vertex_stride_bytes != 44:
        raise AssertionError(
            f"RenderVertex stride changed: {vertex_stride_bytes} != 44"
        )
    if vertex_color_offset_bytes != 32:
        raise AssertionError(
            "RenderVertex color offset changed: "
            f"{vertex_color_offset_bytes} != 32"
        )
    np.testing.assert_allclose(
        authored_vertices["color"],
        gl_colors.numpy(),
        rtol=0.0,
        atol=1.0e-6,
    )
    check("gl_render_vertex_layout")
    check("gl_fill_vertex_data_authored_color")

    wp.launch(
        fill_vertex_data,
        dim=2,
        inputs=[gl_points, gl_normals, gl_uvs, None],
        outputs=[gl_vertices],
        device=device,
    )
    wp.synchronize_device(device)
    fallback_vertices = gl_vertices.numpy().copy()
    np.testing.assert_allclose(
        fallback_vertices["color"],
        np.ones((2, 3), dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    artifact_arrays["render_vertex_authored_colors"] = authored_vertices["color"]
    artifact_arrays["render_vertex_white_fallback"] = fallback_vertices["color"]
    artifact_arrays["render_vertex_stride_bytes"] = np.asarray(vertex_stride_bytes, dtype=np.int32)
    artifact_arrays["render_vertex_color_offset_bytes"] = np.asarray(
        vertex_color_offset_bytes,
        dtype=np.int32,
    )
    observed["render_vertex_cuda"] = {
        "authored_colors": authored_vertices["color"].tolist(),
        "white_fallback": fallback_vertices["color"].tolist(),
        "stride_bytes": vertex_stride_bytes,
        "color_offset_bytes": vertex_color_offset_bytes,
    }
    check("gl_fill_vertex_data_white_fallback")

    # Standalone particle: center ray must retain middle particle index 1.
    target_color = np.asarray((0.25, 0.5, 0.75), dtype=np.float32)
    particle_builder = newton.ModelBuilder()
    particle_builder.add_particle(
        wp.vec3(2.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
        radius=0.25,
        color=(1.0, 0.0, 0.0),
    )
    particle_builder.add_particle(
        wp.vec3(0.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
        radius=0.75,
        color=tuple(target_color),
    )
    particle_builder.add_particle(
        wp.vec3(-2.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
        radius=0.25,
        color=(0.0, 1.0, 0.0),
    )
    particle_model = particle_builder.finalize(device=device)
    require_model_device(particle_model, "particle_model")

    particle_config = SensorTiledCamera.RenderConfig(
        enable_particles=True,
        enable_textures=False,
        output_color_space=newton.utils.ColorSpace.SRGB,
    )
    for with_normal, path_name in ((False, "depth_only"), (True, "full")):
        rendered = render_center(
            particle_model,
            particle_config,
            with_normal=with_normal,
        )
        label = f"standalone_middle_index_{path_name}"
        verify_pixel(
            rendered,
            (0, 0, 0, 0),
            target_color,
            int(PARTICLES_SHAPE_ID),
            label,
        )
        artifact_arrays[f"{label}_rgba"] = rendered["rgba"]

    # Triangle: interpolate raw display values first, then decode once.
    # Scalene, so the sampled point lands on barycentric weights (0.2, 0.3, 0.5)
    # and every permutation of the three weights yields a different pixel.
    triangle_builder = newton.ModelBuilder()
    triangle_builder.add_particles(
        pos=[
            wp.vec3(-1.0, 0.0, -2.0),
            wp.vec3(0.0, -1.0, -2.0),
            wp.vec3(0.4, 0.6, -2.0),
        ],
        vel=[wp.vec3()] * 3,
        mass=[1.0] * 3,
        radius=[0.01] * 3,
        colors=[
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
    )
    triangle_builder.add_triangle(0, 1, 2)
    triangle_model = triangle_builder.finalize(device=device)
    require_model_device(triangle_model, "triangle_model")

    display_mix = np.asarray((0.2, 0.3, 0.5), dtype=np.float32)
    linear_mix = srgb_to_linear_literal(display_mix)

    for space_name, output_space, expected_rgb in (
        ("srgb", newton.utils.ColorSpace.SRGB, display_mix),
        ("linear", newton.utils.ColorSpace.LINEAR, linear_mix),
    ):
        triangle_config = SensorTiledCamera.RenderConfig(
            enable_backface_culling=False,
            enable_particles=False,
            enable_textures=False,
            output_color_space=output_space,
        )
        for with_normal, path_name in ((False, "depth_only"), (True, "full")):
            rendered = render_center(
                triangle_model,
                triangle_config,
                with_normal=with_normal,
            )
            label = f"triangle_display_interp_{space_name}_{path_name}"
            verify_pixel(
                rendered,
                (0, 0, 0, 0),
                expected_rgb,
                int(TRIANGLE_MESH_SHAPE_ID),
                label,
            )
            artifact_arrays[f"{label}_rgba"] = rendered["rgba"]

    artifact_arrays["triangle_display_mix"] = display_mix
    artifact_arrays["triangle_linear_mix"] = linear_mix

    # Existing no-authored-color white fallback: particle and triangle.
    uncolored_particle_builder = newton.ModelBuilder()
    uncolored_particle_builder.add_particle(
        wp.vec3(0.0, 0.0, -2.0),
        wp.vec3(),
        1.0,
        radius=0.5,
    )
    uncolored_particle_model = uncolored_particle_builder.finalize(device=device)
    require_model_device(uncolored_particle_model, "uncolored_particle_model")
    if uncolored_particle_model.particle_display_color is not None:
        raise AssertionError("uncolored particle model unexpectedly allocated colors")

    rendered = render_center(
        uncolored_particle_model,
        SensorTiledCamera.RenderConfig(
            enable_particles=True,
            enable_textures=False,
        ),
        with_normal=False,
    )
    verify_pixel(
        rendered,
        (0, 0, 0, 0),
        (1.0, 1.0, 1.0),
        int(PARTICLES_SHAPE_ID),
        "uncolored_particle_white_fallback",
    )
    artifact_arrays["uncolored_particle_rgba"] = rendered["rgba"]

    uncolored_triangle_builder = newton.ModelBuilder()
    uncolored_triangle_builder.add_particles(
        pos=[
            wp.vec3(-1.0, -1.0, -2.0),
            wp.vec3(1.0, -1.0, -2.0),
            wp.vec3(0.0, 1.0, -2.0),
        ],
        vel=[wp.vec3()] * 3,
        mass=[1.0] * 3,
        radius=[0.01] * 3,
    )
    uncolored_triangle_builder.add_triangle(0, 1, 2)
    uncolored_triangle_model = uncolored_triangle_builder.finalize(device=device)
    require_model_device(uncolored_triangle_model, "uncolored_triangle_model")
    if uncolored_triangle_model.particle_display_color is not None:
        raise AssertionError("uncolored triangle model unexpectedly allocated colors")

    rendered = render_center(
        uncolored_triangle_model,
        SensorTiledCamera.RenderConfig(
            enable_backface_culling=False,
            enable_particles=False,
            enable_textures=False,
        ),
        with_normal=False,
    )
    verify_pixel(
        rendered,
        (0, 0, 0, 0),
        (1.0, 1.0, 1.0),
        int(TRIANGLE_MESH_SHAPE_ID),
        "uncolored_triangle_white_fallback",
    )
    artifact_arrays["uncolored_triangle_rgba"] = rendered["rgba"]

    # Multiworld/global face-index isolation.
    world_colors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    spacing = 10.0
    multiworld_builder = newton.ModelBuilder()

    for world_index, color in enumerate(world_colors):
        blueprint = newton.ModelBuilder()
        blueprint.add_particles(
            pos=[
                wp.vec3(-1.0, -1.0, -2.0),
                wp.vec3(1.0, -1.0, -2.0),
                wp.vec3(0.0, 1.0, -2.0),
            ],
            vel=[wp.vec3()] * 3,
            mass=[1.0] * 3,
            radius=[0.01] * 3,
            colors=[color] * 3,
        )
        blueprint.add_triangle(0, 1, 2)
        multiworld_builder.add_world(
            blueprint,
            xform=wp.transform(
                wp.vec3(float(world_index) * spacing, 0.0, 0.0),
                wp.quat_identity(),
            ),
        )

    multiworld_model = multiworld_builder.finalize(device=device)
    require_model_device(multiworld_model, "multiworld_model")
    if multiworld_model.world_count != 2:
        raise AssertionError(f"expected 2 worlds, got {multiworld_model.world_count}")

    global_tri_indices = multiworld_model.tri_indices.numpy()
    np.testing.assert_array_equal(
        global_tri_indices,
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )

    multiworld_camera_transforms = wp.array(
        [[
            wp.transformf(
                wp.vec3f(float(world_index) * spacing, 0.0, 0.0),
                wp.quatf(0.0, 0.0, 0.0, 1.0),
            )
            for world_index in range(2)
        ]],
        dtype=wp.transformf,
        device=device,
    )

    rendered = render_center(
        multiworld_model,
        SensorTiledCamera.RenderConfig(
            enable_backface_culling=False,
            enable_particles=False,
            enable_textures=False,
            max_distance=5.0,
        ),
        with_normal=False,
        camera_transforms=multiworld_camera_transforms,
    )
    for world_index, expected_color in enumerate(world_colors):
        verify_pixel(
            rendered,
            (world_index, 0, 0, 0),
            expected_color,
            int(TRIANGLE_MESH_SHAPE_ID),
            f"multiworld_global_face_world_{world_index}",
        )

    artifact_arrays["multiworld_rgba"] = rendered["rgba"]
    artifact_arrays["multiworld_shape"] = rendered["shape"]
    artifact_arrays["multiworld_global_tri_indices"] = global_tri_indices

    # ViewerBase inactive-particle CUDA scan/compaction path.
    class CaptureViewer(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.data = None

        def log_points(
            self,
            name,
            points,
            radii=None,
            colors=None,
            hidden=False,
        ):
            self.data = points, radii, colors, hidden

    active = int(newton.ParticleFlags.ACTIVE)
    compact_builder = newton.ModelBuilder()
    compact_builder.add_particles(
        pos=[wp.vec3(float(index), 0.0, 0.0) for index in range(5)],
        vel=[wp.vec3()] * 5,
        mass=[1.0] * 5,
        radius=[0.1, 0.2, 0.3, 0.4, 0.5],
        flags=[active, 0, active, 0, active],
        colors=[
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        ],
    )
    compact_model = compact_builder.finalize(device=device)
    require_model_device(compact_model, "compact_model")

    viewer = CaptureViewer()
    viewer.set_model(compact_model)
    viewer._log_particles(compact_model.state())
    wp.synchronize_device(device)

    if viewer.data is None:
        raise AssertionError("ViewerNull did not receive compacted particles")

    compact_points, compact_radii, compact_colors, _hidden = viewer.data
    require_device(compact_points, "compact_points")
    require_device(compact_radii, "compact_radii")
    require_device(compact_colors, "compact_colors")

    compact_points_np = compact_points.numpy()
    compact_radii_np = compact_radii.numpy()
    compact_colors_np = compact_colors.numpy()

    np.testing.assert_allclose(compact_points_np[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(compact_radii_np, [0.1, 0.3, 0.5])
    np.testing.assert_allclose(
        compact_colors_np,
        [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        ],
    )

    artifact_arrays["compacted_points"] = compact_points_np
    artifact_arrays["compacted_radii"] = compact_radii_np
    artifact_arrays["compacted_colors"] = compact_colors_np
    check("viewer_inactive_particle_color_compaction")

    if launch_counter["count"] != 9:
        raise AssertionError(
            f"expected 9 camera launches, got {launch_counter['count']}"
        )
    if len(checks) != 17 or not all(checks.values()):
        raise AssertionError(f"incomplete checks: {checks}")

    artifact_values = {
        **artifact_arrays,
        "passed": True,
        "status": "completed",
        "commit": commit,
        "branch": expected_branch,
        "device": device.alias,
        "device_name": device.name,
        "device_arch": int(device.arch),
        "newton_version": loaded_newton_version,
        "warp_version": loaded_warp_version,
        "camera_launch_count": launch_counter["count"],
        "check_names": np.asarray(sorted(checks)),
    }
    atomic_npz(artifact_path, **artifact_values)

    with np.load(artifact_path, allow_pickle=False) as artifact:
        if not bool(artifact["passed"]):
            raise AssertionError("completed artifact does not pass")
        if str(artifact["status"]) != "completed":
            raise AssertionError(f"unexpected artifact status: {artifact['status']}")
        for key in artifact.files:
            value = artifact[key]
            if value.dtype.hasobject:
                raise AssertionError(f"object dtype in artifact key {key!r}")

    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    report = {
        "pass": True,
        "status": "completed",
        "issue": 3852,
        "commit": commit,
        "branch": expected_branch,
        "git_tree": tree,
        "device": device.alias,
        "device_name": device.name,
        "device_arch": int(device.arch),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "python": sys.version.split()[0],
            "newton": loaded_newton_version,
            "warp-lang": loaded_warp_version,
            "numpy": metadata.version("numpy"),
        },
        "origins": {
            "newton": str(newton_origin),
            "warp": str(warp_origin),
            "warp_distribution_root": str(warp_distribution_root),
            "python": sys.executable,
        },
        "checks": checks,
        "observed": observed,
        "render_vertex_layout": {
            "stride_bytes": vertex_stride_bytes,
            "color_offset_bytes": vertex_color_offset_bytes,
        },
        "camera_launch_count": launch_counter["count"],
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "artifact_readable_without_pickle": True,
        "elapsed_s": time.perf_counter() - started,
    }
    atomic_json(result_path, report)
    print("RESULT " + json.dumps(report, sort_keys=True))

except BaseException as exc:
    failure = {
        "pass": False,
        "status": "exception",
        "commit": expected_sha,
        "branch": expected_branch,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "elapsed_s": time.perf_counter() - started,
    }
    atomic_json(result_path, failure)
    atomic_npz(
        artifact_path,
        passed=False,
        status="exception",
        commit=expected_sha,
        branch=expected_branch,
        failure_type=type(exc).__name__,
        failure_message=str(exc),
    )
    print("RESULT " + json.dumps(failure, sort_keys=True))
    traceback.print_exc()
    raise
PY

  "$PY" "$PROBE"

  POST_STATUS="$(git -C "$SRC" status --porcelain --untracked-files=all)"
  {
    echo "post_probe_status=${POST_STATUS}"
    echo "post_probe_head=$(git -C "$SRC" rev-parse HEAD)"
  } >> "$GIT_INFO"
  test -z "$POST_STATUS"
) 2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
set -e

HASH_INPUTS=()
for path in \
  "$PROBE" \
  "$ARTIFACT" \
  "$RESULT_JSON" \
  "$PACKAGES" \
  "$GPU_INFO" \
  "$GIT_INFO" \
  "$LOG"
do
  if test -f "$path"; then
    HASH_INPUTS+=("$path")
  fi
done

sha256sum "${HASH_INPUTS[@]}" | tee "$HASHES"

echo "RESULT_LOG=$LOG"
echo "RESULT_PROBE=$PROBE"
echo "RESULT_ARTIFACT=$ARTIFACT"
echo "RESULT_JSON=$RESULT_JSON"
echo "RESULT_PACKAGES=$PACKAGES"
echo "RESULT_GPU=$GPU_INFO"
echo "RESULT_GIT=$GIT_INFO"
echo "RESULT_SHA256=$HASHES"

exit "$RC"
