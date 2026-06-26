#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the packaged G1 + Omnipicker URDF into a MuJoCo-loadable MJCF.

The released G1 package stores arm visuals as binary FBX files and the
Omnipicker visuals as DAE files. MuJoCo does not reliably compile that mixture
directly from URDF, so this script first extracts the package, converts all
referenced visual meshes to OBJ, rewrites the URDF mesh paths, and asks MuJoCo
to compile/save the final MJCF.

The default output is visual-only. That is intentional for Phantom/Masquerade
style video editing: we need high-quality rendered robot pixels, depth, and
masks, not contact dynamics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_G1_ZIP = PROJECT_ROOT / "G1" / "G1_URDF_Omnipicker.zip"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "g1_mjcf"
DEFAULT_URDF_IN_ZIP = "G1_URDF_Omnipicker/urdf/G1/G1_omnipicker_omnipicker.urdf"

sys.path.insert(0, str(SCRIPT_DIR))

from render_g1_arm_mesh_on_serve_bread import parse_fbx_mesh  # noqa: E402
from render_g1_gripper_mesh_on_serve_bread import parse_dae_mesh, rpy_to_R  # noqa: E402


def write_sleeve_obj(
    dst: Path,
    rings: list[tuple[np.ndarray, float]],
    axis: np.ndarray,
    label: str,
    segments: int = 48,
    cap_ends: bool = False,
) -> None:
    """Write a render-only tapered sleeve in the target body local frame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, up))) > 0.95:
        up = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(axis, up)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    vertices: list[tuple[float, float, float]] = []
    for center, radius in rings:
        center = np.asarray(center, dtype=np.float64)
        for i in range(segments):
            theta = 2.0 * np.pi * float(i) / float(segments)
            point = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
            vertices.append((float(point[0]), float(point[1]), float(point[2])))

    faces: list[tuple[int, int, int]] = []
    for ring_idx in range(len(rings) - 1):
        base0 = ring_idx * segments
        base1 = (ring_idx + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((base0 + i + 1, base1 + i + 1, base1 + j + 1))
            faces.append((base0 + i + 1, base1 + j + 1, base0 + j + 1))
    if cap_ends:
        start_center_idx = len(vertices) + 1
        vertices.append(tuple(float(v) for v in rings[0][0]))
        end_center_idx = len(vertices) + 1
        vertices.append(tuple(float(v) for v in rings[-1][0]))
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((start_center_idx, j + 1, i + 1))
            faces.append((end_center_idx, (len(rings) - 1) * segments + i + 1, (len(rings) - 1) * segments + j + 1))

    with dst.open("w", encoding="utf-8") as f:
        f.write(f"# Render-only URDF-collision-derived sleeve for {label}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def wrist_sleeve_rings() -> tuple[list[tuple[np.ndarray, float]], np.ndarray]:
    collision_origin = np.asarray([-1.7782549414354032e-05, -0.05507596972928094, -0.0017447156928267265])
    collision_rpy = [0.0, -1.4835107730827122, -1.56897465223053]
    collision_radius = 0.04857149291994264
    collision_axis = rpy_to_R(*collision_rpy) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    collision_axis /= np.linalg.norm(collision_axis)

    def center_at_y(y: float) -> np.ndarray:
        t = (float(y) - collision_origin[1]) / collision_axis[1]
        return collision_origin + t * collision_axis

    return (
        [
            (center_at_y(-0.070), collision_radius * 0.58),
            (center_at_y(-0.096), collision_radius * 0.70),
            (center_at_y(-0.160), collision_radius * 0.64),
            (center_at_y(-0.187), collision_radius * 0.50),
        ],
        collision_axis,
    )


def render_sleeve_specs() -> list[dict]:
    wrist_rings, wrist_axis = wrist_sleeve_rings()
    return [
        {
            "name": "arm_r_shoulder_sleeve",
            "body": "arm_r_link1",
            "rgba": "0.82 0.83 0.82 1",
            "rings": [
                (np.asarray([0.0002, -0.0020, -0.1580]), 0.032),
                (np.asarray([0.0002, -0.0020, -0.1180]), 0.040),
                (np.asarray([0.0002, -0.0020, 0.0320]), 0.044),
            ],
            "axis": np.asarray([0.0, 0.0, 1.0]),
        },
        {
            "name": "arm_r_upper_sleeve",
            "body": "arm_r_link2",
            "rgba": "0.82 0.83 0.82 1",
            "rings": [
                (np.asarray([0.0, -0.060, 0.0010]), 0.038),
                (np.asarray([0.0, -0.160, 0.0010]), 0.046),
                (np.asarray([0.0, -0.300, 0.0010]), 0.036),
            ],
            "axis": np.asarray([0.0, -1.0, 0.0]),
        },
        {
            "name": "arm_r_elbow_sleeve",
            "body": "arm_r_link4",
            "rgba": "0.84 0.85 0.84 1",
            "rings": [
                (np.asarray([0.0, -0.074, -0.0044]), 0.036),
                (np.asarray([0.0, -0.132, -0.0044]), 0.042),
                (np.asarray([0.0, -0.198, -0.0044]), 0.034),
            ],
            "axis": np.asarray([0.0, -1.0, 0.0]),
        },
        {
            "name": "arm_r_mid_sleeve",
            "body": "arm_r_link3",
            "rgba": "0.84 0.85 0.84 1",
            "rings": [
                (np.asarray([0.0007, -0.0000, -0.100]), 0.034),
                (np.asarray([0.0007, -0.0000, -0.040]), 0.043),
                (np.asarray([0.0007, -0.0000, 0.022]), 0.038),
            ],
            "axis": np.asarray([0.0, 0.0, 1.0]),
        },
        {
            "name": "arm_r_wrist_sleeve",
            "body": "arm_r_link6",
            "rgba": "0.82 0.83 0.82 1",
            "rings": wrist_rings,
            "axis": wrist_axis,
        },
    ]


def package_mesh_path_to_relative(filename: str) -> Path:
    prefix = "package://genie_robot_description/"
    if filename.startswith(prefix):
        return Path(filename[len(prefix) :])
    return Path(filename)


def export_mesh_as_obj(src: Path, dst: Path) -> dict:
    import trimesh

    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if suffix == ".fbx":
        part = parse_fbx_mesh(src, (180, 180, 180))
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
    elif suffix == ".dae":
        part = parse_dae_mesh(src, (180, 180, 180))
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
    else:
        loaded = trimesh.load(src, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded

    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }


def rewrite_urdf_meshes(
    urdf_path: Path,
    package_root: Path,
    mjcf_root: Path,
    visual_only: bool,
    keep_links: set[str] | None,
) -> tuple[Path, list[dict]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if keep_links is not None:
        for link in list(root.findall("link")):
            if link.attrib.get("name") not in keep_links:
                root.remove(link)
        for joint in list(root.findall("joint")):
            parent = joint.find("parent")
            child = joint.find("child")
            parent_name = parent.attrib.get("link") if parent is not None else None
            child_name = child.attrib.get("link") if child is not None else None
            if parent_name not in keep_links or child_name not in keep_links:
                root.remove(joint)
        for loop_joint in list(root.findall("loop_joint")):
            root.remove(loop_joint)

    if visual_only:
        for link in root.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)

    conversions: list[dict] = []
    converted: dict[Path, Path] = {}
    for mesh_el in root.findall(".//mesh"):
        filename = mesh_el.attrib.get("filename")
        if not filename:
            continue
        rel = package_mesh_path_to_relative(filename)
        src = package_root / rel
        dst_rel = rel.with_suffix(".obj")
        dst = mjcf_root / dst_rel
        if src not in converted:
            conversions.append(export_mesh_as_obj(src, dst))
            converted[src] = dst_rel
        mesh_el.set("filename", str(converted[src]))

    # MuJoCo's URDF importer compiles collision geometry into geoms. It may drop
    # pure visual tags, so for rendering we mirror every visual mesh as a
    # collision mesh after path conversion. The output MJCF is visual-only by
    # default, so these geoms are meant for pixels / depth / segmentation.
    if visual_only:
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                collision = ET.Element("collision")
                origin = visual.find("origin")
                geometry = visual.find("geometry")
                if origin is not None:
                    collision.append(ET.fromstring(ET.tostring(origin, encoding="unicode")))
                if geometry is not None:
                    collision.append(ET.fromstring(ET.tostring(geometry, encoding="unicode")))
                link.append(collision)

    # MuJoCo's URDF importer supports normal links/joints and ignores many
    # ROS-only concepts. The Omnipicker loop_joint tags are custom, so remove
    # them before compilation and keep the visible tree joints.
    for loop_joint in list(root.findall("loop_joint")):
        root.remove(loop_joint)

    rewritten = mjcf_root / "g1_omnipicker_rewritten.urdf"
    tree.write(rewritten, encoding="utf-8", xml_declaration=True)
    return rewritten, conversions


def patch_render_mjcf(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {xml_path}")

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    sleeve_specs = render_sleeve_specs()
    for spec in sleeve_specs:
        mesh_name = f"{spec['name']}_mesh"
        mesh_rel = Path("meshes") / "generated" / f"{spec['name']}.obj"
        write_sleeve_obj(xml_path.parent / mesh_rel, spec["rings"], spec["axis"], spec["name"])
        if asset.find(f"mesh[@name='{mesh_name}']") is None:
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_rel)})

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(1, visual)
    if visual.find("quality") is None:
        ET.SubElement(visual, "quality", {"shadowsize": "2048", "offsamples": "4"})
    if visual.find("map") is None:
        ET.SubElement(visual, "map", {"znear": "0.02", "zfar": "3.0"})

    rgba_by_mesh = {
        "arm_r_base_link": "0.70 0.72 0.72 1",
        "arm_r_link1": "0.86 0.87 0.86 1",
        "arm_r_link2": "0.62 0.64 0.64 1",
        "arm_r_link3": "0.84 0.85 0.84 1",
        "arm_r_link4": "0.58 0.60 0.60 1",
        "arm_r_link5": "0.84 0.85 0.84 1",
        "arm_r_link6": "0.58 0.61 0.62 1",
        "arm_r_link7": "0.82 0.83 0.82 1",
        "gripper_base_link": "0.42 0.44 0.45 1",
        "inner_link1": "0.18 0.20 0.22 1",
        "inner_link2": "0.18 0.20 0.22 1",
        "inner_link3": "0.36 0.38 0.38 1",
        "inner_link4": "0.76 0.77 0.75 1",
        "outer_link1": "0.18 0.20 0.22 1",
        "outer_link2": "0.18 0.20 0.22 1",
        "outer_link3": "0.36 0.38 0.38 1",
        "outer_link4": "0.76 0.77 0.75 1",
    }
    used_names: set[str] = set()
    for geom in root.findall(".//geom"):
        mesh = geom.attrib.get("mesh", "geom")
        name = mesh
        n = 1
        while name in used_names:
            n += 1
            name = f"{mesh}_{n}"
        used_names.add(name)
        geom.set("name", name)
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("rgba", rgba_by_mesh.get(mesh, "0.72 0.73 0.72 1"))

    if worldbody.find("camera[@name='aria']") is None:
        ET.SubElement(worldbody, "camera", {"name": "aria", "pos": "0 0 0", "quat": "1 0 0 0", "mode": "fixed", "fovy": "60"})
    if worldbody.find("light[@name='key_light']") is None:
        ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0 -1 1", "dir": "0 0 -1", "directional": "true", "diffuse": "0.8 0.8 0.8"})
    if worldbody.find("light[@name='fill_light']") is None:
        ET.SubElement(worldbody, "light", {"name": "fill_light", "pos": "-1 0 0.5", "dir": "1 0 -0.3", "directional": "true", "diffuse": "0.35 0.35 0.35"})

    if worldbody.find("body[@name='arm_r_base_link']") is None:
        movable = [child for child in list(worldbody) if child.tag not in {"camera", "light"}]
        for child in movable:
            worldbody.remove(child)
        root_body = ET.Element("body", {"name": "arm_r_base_link", "pos": "0 0 0", "quat": "1 0 0 0"})
        root_body.append(ET.Element("freejoint", {"name": "g1_root_freejoint"}))
        for child in movable:
            root_body.append(child)
        worldbody.insert(0, root_body)

    # The released visual meshes have a few visible inter-link gaps even though
    # the kinematic joints are continuous. Add render-only sleeves derived from
    # URDF collision hull centerlines rather than changing the robot FK.
    for spec in sleeve_specs:
        body = worldbody.find(f".//body[@name='{spec['body']}']")
        if body is None or body.find(f"geom[@name='{spec['name']}']") is not None:
            continue
        ET.SubElement(
            body,
            "geom",
            {
                "name": spec["name"],
                "type": "mesh",
                "mesh": f"{spec['name']}_mesh",
                "rgba": spec["rgba"],
                "contype": "0",
                "conaffinity": "0",
            },
        )

    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def save_mjcf_from_urdf(rewritten_urdf: Path, out_xml: Path) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(rewritten_urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(out_xml), model)
    patch_render_mjcf(out_xml)
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "ncam": int(model.ncam),
    }


def right_arm_with_parent_chain_link_set() -> set[str]:
    return {
        "base_link",
        "body_link1",
        "body_link2",
        "arm_base_link",
        "arm_r_base_link",
        "arm_r_link1",
        "arm_r_link2",
        "arm_r_link3",
        "arm_r_link4",
        "arm_r_link5",
        "arm_r_link6",
        "arm_r_end_link",
        "gripper_r_base_link",
        "gripper_r_inner_link1",
        "gripper_r_inner_link2",
        "gripper_r_inner_link3",
        "gripper_r_inner_link4",
        "gripper_r_outer_link1",
        "gripper_r_outer_link2",
        "gripper_r_outer_link3",
        "gripper_r_outer_link4",
        "gripper_r_center_link",
    }


def descendant_link_set(root: ET.Element, root_link: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.attrib["link"], []).append(child.attrib["link"])
    keep = {root_link}
    stack = [root_link]
    while stack:
        link = stack.pop()
        for child in children.get(link, []):
            if child not in keep:
                keep.add(child)
                stack.append(child)
    return keep


def right_arm_local_link_set(urdf_path: Path) -> set[str]:
    root = ET.parse(urdf_path).getroot()
    return descendant_link_set(root, "arm_r_base_link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-zip", type=Path, default=DEFAULT_G1_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--urdf-in-zip", default=DEFAULT_URDF_IN_ZIP)
    parser.add_argument("--scope", choices=["right-arm-local", "right-arm", "full"], default="right-arm-local")
    parser.add_argument("--with-collision", action="store_true", help="Keep URDF collision geometry in the MJCF.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = Path(tempfile.mkdtemp(prefix="g1_urdf_extract_"))
    try:
        with zipfile.ZipFile(args.g1_zip) as zf:
            zf.extractall(extract_dir)
        package_root = extract_dir / "G1_URDF_Omnipicker"
        urdf_path = extract_dir / args.urdf_in_zip
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found inside zip extraction: {urdf_path}")

        if args.scope == "right-arm-local":
            keep_links = right_arm_local_link_set(urdf_path)
        elif args.scope == "right-arm":
            keep_links = right_arm_with_parent_chain_link_set()
        else:
            keep_links = None
        rewritten_urdf, conversions = rewrite_urdf_meshes(
            urdf_path=urdf_path,
            package_root=package_root,
            mjcf_root=args.out_dir,
            visual_only=not args.with_collision,
            keep_links=keep_links,
        )
        out_xml = args.out_dir / f"g1_omnipicker_{args.scope.replace('-', '_')}.xml"
        model_info = save_mjcf_from_urdf(rewritten_urdf, out_xml)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    report = {
        "g1_zip": str(args.g1_zip),
        "scope": args.scope,
        "visual_only": not args.with_collision,
        "rewritten_urdf": str(args.out_dir / "g1_omnipicker_rewritten.urdf"),
        "mjcf": str(out_xml),
        "mesh_conversions": len(conversions),
        "model": model_info,
    }
    (args.out_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
