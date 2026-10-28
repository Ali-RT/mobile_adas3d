"""CPU coverage fixtures: no real downloads, inference or GPU."""
import bz2
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts import audit_nuscenes_coverage_0_6m as audit
from scripts import nuscenes_coverage_data as data
from scripts import parking_near_field_geometry as geometry
from scripts import parking_edge_common as common
from scripts import prepare_parking_lidar_depth as old


def archive_fixture(root, name, members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for key, raw in members:
            info = tarfile.TarInfo(key); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    decoded = stream.getvalue()
    raw = gzip.compress(decoded, mtime=0) if name.endswith(".tgz") else bz2.compress(decoded)
    path = root / name; path.write_bytes(raw)
    return path, {"file": name, "sha256": common.sha(raw), "size_bytes": len(raw),
                  "decoded_bytes_checked_not_saved": len(decoded)}


def tiny_dataset(root):
    c = audit.load_config(audit.CONFIG)
    c.update(expected_scenes=1, expected_keyframes=2, expected_scenes_per_part=1,
             local_disk_reserve_gb=0, keyframes_per_chunk=1)
    c["classes"] = {"Vehicle": ["vehicle.car"], "Pedestrian": ["human.pedestrian.adult"]}
    roles = {"scenes": {"scene-fixture": {"log_token": "log", "role": "train"}},
             "previously_inspected_evaluation_logs": []}
    pose = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
    tables = {name: [] for name in data.META_TABLES}
    tables["scene"] = [{"token": "scene", "name": "scene-fixture", "log_token": "log"}]
    tables["category"] = [{"token": "cat", "name": "vehicle.car"}]
    tables["instance"] = [{"token": "track", "category_token": "cat"}]
    for ch in sorted(data.CHANNELS):
        tables["sensor"].append({"token": ch, "channel": ch, "modality": "lidar" if ch == "LIDAR_TOP" else "camera"})
        tables["calibrated_sensor"].append({**pose, "token": ch, "sensor_token": ch,
            "camera_intrinsic": [[100, 0, 50], [0, 100, 30], [0, 0, 1]]})
    labels = []; sensor_members = []; label_members = []
    image = io.BytesIO(); Image.new("RGB", (100, 60), (30, 40, 50)).save(image, format="JPEG")
    for i, z in enumerate([.5, 5.]):
        token = f"frame{i}"; p = f"pose{i}"
        tables["sample"].append({"token": token, "scene_token": "scene"})
        tables["ego_pose"].append({**pose, "token": p})
        tables["sample_annotation"].append({**pose, "token": f"ann{i}", "sample_token": token,
            "instance_token": "track", "size": [1, 1, 1], "translation": [0, 0, z], "visibility_token": "4"})
        for ch in sorted(data.CHANNELS):
            lidar = ch == "LIDAR_TOP"; sd = f"{ch}_{i}"
            filename = f"samples/{ch}/{i}.{'bin' if lidar else 'jpg'}"
            tables["sample_data"].append({"token": sd, "filename": filename,
                "sample_token": token, "ego_pose_token": p, "calibrated_sensor_token": ch,
                "is_key_frame": True, "timestamp": i * 1000000, "width": 100, "height": 60})
            points = np.zeros((5, 5), dtype="<f4"); points[:, 2] = z
            points[:, 0] = np.linspace(-.08 * z, .08 * z, 5)
            sensor_members.append((filename, points.tobytes() if lidar else image.getvalue()))
            if lidar:
                label = f"lidarseg/v1.0-trainval/{sd}_lidarseg.bin"
                labels.append({"token": sd, "filename": label})
                label_members.append((label, bytes([17] * 5)))
    meta_members = [(f"v1.0-trainval/{name}.json", common.encoded(value)) for name, value in tables.items()]
    roles["metadata_table_sha256"] = {Path(name).stem: common.sha(raw) for name, raw in meta_members}
    roles = audit.signed(roles)
    semantic = [{"name": "vehicle.car", "index": 17}, {"name": "human.pedestrian.adult", "index": 2}]
    label_members += [("v1.0-trainval/lidarseg.json", common.encoded(labels)),
                      ("v1.0-trainval/category.json", common.encoded(semantic))]
    meta, mr = archive_fixture(root, "metadata.tgz", meta_members)
    label, lr = archive_fixture(root, "labels.tar.bz2", label_members)
    sensor, sr = archive_fixture(root, "sensors.tgz", sensor_members)
    local = root / "local"; local.mkdir()
    data.extract_support(meta, local / "metadata", mr, c, roles, "metadata")
    data.extract_support(label, local / "lidarseg", lr, c, roles, "lidarseg")
    database = data.build_index(local, local / "metadata", local / "lidarseg", c, roles, {"fixture": True})
    contract = {"part": 1, "archive_sha256": sr["sha256"], "run_sha256": "fixture"}
    part = local / "parts/part01"
    staged = data.stage_part(sensor, sr, part, database, c, contract)
    return c, roles, local, database, part, staged, sensor, sr, contract


class GeometryTests(unittest.TestCase):
    def setUp(self):
            self.config = json.loads(audit.CONFIG.read_text())
            self.guard = self.config["boundary_guard"]

    def test_no_absolute_accuracy_threshold_or_training(self):
            self.assertIsNone(self.config["accuracy_threshold_m"])
            self.assertFalse(self.config["accuracy_pass_fail_enabled"])
            self.assertFalse(self.config["training_authorized"])
            self.assertIn([0, 1], self.config["range_bands_m"])
            self.assertIn([1, 2], self.config["range_bands_m"])

    def test_exact_band_edges_and_positive_depth(self):
            z = np.array([-1., 0., .5, 1., 2., 4., 6., 6.01])
            np.testing.assert_array_equal(geometry.band_mask(z, 0, 2), [False, False, True, True, False, False, False, False])
            np.testing.assert_array_equal(geometry.band_mask(z, 2, 4), [False, False, False, False, True, False, False, False])
            np.testing.assert_array_equal(geometry.band_mask(z, 4, 6), [False, False, False, False, False, True, True, False])

    def test_positive_projection_retains_submetre_and_matches_old_above_one(self):
            pose = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
            cs = {**pose, "camera_intrinsic": [[10, 0, 50], [0, 10, 30], [0, 0, 1]]}
            xyz = np.array([[0, 0, .5], [0, 0, 1.5], [.4, 0, 5], [0, 0, 0], [0, 0, -1]], dtype=np.float32)
            uv, z, ids, _ = geometry.project_positive(xyz, pose, pose, pose, cs, 100, 60)
            np.testing.assert_allclose(z, [.5, 1.5, 5])
            np.testing.assert_array_equal(ids, [0, 1, 2])
            expected_uv, expected_z, expected_ids, _ = old.camera_projection(xyz, pose, pose, pose, cs, 100, 60)
            np.testing.assert_array_equal(uv[z > 1], expected_uv)
            np.testing.assert_array_equal(z[z > 1], expected_z)
            np.testing.assert_array_equal(ids[z > 1], expected_ids)

    def test_sparse_guard_matches_dense_reference_on_random_inputs(self):
            rng = np.random.default_rng(123)
            uv = rng.uniform([2, 2], [97, 57], (3000, 2))
            z = rng.uniform(.05, 12, len(uv)).astype(np.float32)
            sem = rng.choice([0, 2, 17, 31], len(uv)).astype(np.uint8)
            actual = geometry.sparse_near_raster(uv, z, sem, np.arange(len(z)), 100, 60, self.guard)
            config = json.loads(old.CONFIG.read_text())
            config.update(depth_min_m=0, depth_max_m=6)
            expected = old.rasterize(uv, z, sem, np.zeros(len(z), bool), 100, 60, config)
            for field in ["xy", "z", "semantic", "conservative"]:
                np.testing.assert_array_equal(actual[field], expected[field])

    def test_near_invalid_return_occludes_far_valid_return(self):
            uv = np.array([[10, 10], [10, 10], [40, 40]], dtype=float)
            z = np.array([.5, 3., 1.5], dtype=np.float32)
            p = geometry.sparse_near_raster(uv, z, np.array([0, 17, 2]), np.arange(3), 100, 60, self.guard)
            np.testing.assert_array_equal(p["source_indices"], [2])
            np.testing.assert_array_equal(p["z"], [1.5])

    def test_guard_uses_far_returns_before_range_filter(self):
            uv = np.array([[10, 10], [11, 10]], dtype=float)
            p = geometry.sparse_near_raster(uv, np.array([5.8, 7.]), np.full(2, 17), np.arange(2), 100, 60, self.guard)
            np.testing.assert_array_equal(p["z"], [5.8])
            self.assertFalse(p["conservative"][0])

    def test_empty_projection(self):
            p = geometry.sparse_near_raster(np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0, dtype=int), 100, 60, self.guard)
            self.assertEqual(len(p["z"]), 0)

    def test_box_axes_semantics_and_ambiguous_ownership(self):
            ann = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0], "size": [2, 4, 2], "semantic_index": 17}
            world = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 0], [5, 5, 5]], dtype=float).T
            sem = np.array([17, 17, 2, 17])
            np.testing.assert_array_equal(geometry.associate_objects(world, sem, [ann]), [0, -1, -1, -1])
            np.testing.assert_array_equal(geometry.associate_objects(world, sem, [ann, ann]), [-2, -1, -1, -1])


class FullCoverageTests(unittest.TestCase):
    def test_frozen_roles_are_log_disjoint_and_unchanged(self):
        c = audit.load_config(audit.CONFIG)
        roles = data.verified_roles(audit.ROOT / c["frozen_roles_file"], c["expected_scenes"])
        counts = Counter(r["role"] for r in roles["scenes"].values())
        self.assertEqual(counts, {"train": 477, "dev": 150, "heldout_eval": 113, "excluded_previously_used_mini_log": 110})
        self.assertEqual(len(roles["previously_inspected_evaluation_logs"]), 3)

    def test_paths_and_special_members_rejected(self):
        for name in ["../x", "/x", "x/../../y", "a\\b", ""]:
            with self.assertRaises(ValueError): data.safe_name(name)
        m = tarfile.TarInfo("link"); m.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError): data.member_info(m, set(), 20)
        m = tarfile.TarInfo("."); m.type = tarfile.DIRTYPE
        self.assertEqual(data.member_info(m, set(), 20), ".")
        m = tarfile.TarInfo("file"); m.size = 5
        seen = set(); data.member_info(m, seen, 20)
        with self.assertRaises(ValueError): data.member_info(m, seen, 20)

    def test_archive_crc_size_and_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, receipt = archive_fixture(root, "bad.tgz", [("../escape", b"x")])
            with data.bounded_tar(path, receipt) as archive:
                with self.assertRaises(ValueError):
                    for m in archive: data.member_info(m, set(), 20)
            receipt["decoded_bytes_checked_not_saved"] -= 1
            with self.assertRaises(ValueError):
                with data.bounded_tar(path, receipt) as archive: list(archive)
            raw = bytearray(path.read_bytes()); raw[-8] ^= 1; path.write_bytes(raw)
            with self.assertRaises((OSError, ValueError)):
                with data.bounded_tar(path, {**receipt, "decoded_bytes_checked_not_saved": 10240}) as archive: list(archive)

    def test_copy_resumes_owned_prefix_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset = root / "drive"; local = root / "local"
            (dataset / "archives").mkdir(parents=True)
            raw = b"fixture" * 100
            (dataset / "archives/fixture.tgz").write_bytes(raw)
            receipt = {"file": "fixture.tgz", "sha256": common.sha(raw), "size_bytes": len(raw)}
            destination = local / "archives/fixture.tgz"
            common.immutable(destination.with_name("fixture.tgz.copy.json"), common.encoded(receipt))
            partial = destination.with_name("fixture.tgz.partial"); partial.write_bytes(raw[:128])
            self.assertEqual(data.copy_archive(dataset, local, receipt, 0).read_bytes(), raw)
            self.assertFalse(partial.exists())
            destination.write_bytes(b"different")
            with self.assertRaises(ValueError): data.copy_archive(dataset, local, receipt, 0)

    def test_metadata_to_projection_to_chunk_resume_and_union(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, roles, local, db, part, staged, sensor, sr, contract = tiny_dataset(root)
            # All cameras checked, but no extracted RGB files.
            self.assertEqual(staged["rgb_headers_checked"], 12)
            self.assertFalse(list(local.rglob("*.jpg")))
            self.assertEqual(data.stage_part(sensor, sr, part, db, c, contract), staged)
            records = [audit.audit_frame(f, part, local / "lidarseg", db, c, roles) for f in staged["keyframes"]]
            tokens = [f["sample_token"] for f in staged["keyframes"]]
            audit.validate_records(records, tokens, roles)
            census = audit.Census(c)
            for record in records: census.add(record, roles["scenes"]["scene-fixture"])
            cell = census.result()["cells"]["Vehicle:0-6m"]
            self.assertEqual(cell["surface_supported_object_camera_appearances"], 12)
            self.assertEqual(cell["surface_supported_annotation_observations"], 2)
            self.assertEqual(cell["surface_supported_instance_tracks"], 1)
            self.assertEqual(census.result()["cells"]["Vehicle:0-1m"]["surface_supported_annotation_observations"], 1)
            path = root / "result/chunks/00000.json.gz"
            raw = gzip.compress(common.encoded(audit.signed({"part_identity": contract, "records": records})), mtime=0)
            common.immutable(path, raw)
            self.assertEqual(audit.read_chunk(path, contract, tokens, roles, common.sha(raw)), records)
            with self.assertRaises(ValueError): audit.read_chunk(path, contract, list(reversed(tokens)), roles)
            changed = json.loads(gzip.decompress(raw)); changed["records"][0]["scene"] = "other"
            path.write_bytes(gzip.compress(common.encoded(changed)))
            with self.assertRaises(ValueError): audit.read_chunk(path, contract, tokens, roles)

    def test_full_part_resume_skips_source_staging_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, roles, local, db, part, staged, sensor, sr, _ = tiny_dataset(root)
            identity = {"manifest_sha256": "fixture"}; output = root / "output"
            # tiny_dataset already staged this exact part contract.
            with mock.patch.object(data, "copy_archive", return_value=sensor):
                result = audit.process_part(1, sr, root, output, local, db, c, roles, identity, 2, keep_local=True)
            with mock.patch.object(data, "copy_archive", side_effect=AssertionError("should skip copy")):
                self.assertEqual(audit.process_part(1, sr, root, output, local, db, c, roles, identity, 2), result)
            chunk = output / "parts/part01" / result["chunks"][0]["file"]
            chunk.write_bytes(b"changed")
            with self.assertRaises(ValueError): audit.completed_part(output / "parts/part01", result["part_identity"], roles)

    def test_missing_part_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            c = audit.load_config(audit.CONFIG)
            receipts = {"v1.0-trainval01_keyframes.tgz": {"sha256": "fixture"}}
            with self.assertRaises(FileNotFoundError):
                audit.aggregate(Path(folder), c, {}, receipts, {"manifest_sha256": "fixture"})

    def test_interrupted_part_reuses_its_durable_first_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, roles, local, db, part, staged, sensor, sr, _ = tiny_dataset(root)
            output = root / "output"; identity = {"manifest_sha256": "fixture"}
            original = audit.audit_frame
            def fail_second(frame, *args):
                if frame["sample_token"] == "frame1":
                    raise OSError("simulated runtime interruption")
                return original(frame, *args)
            with mock.patch.object(data, "copy_archive", return_value=sensor), mock.patch.object(audit, "audit_frame", side_effect=fail_second):
                with self.assertRaises(OSError):
                    audit.process_part(1, sr, root, output, local, db, c, roles, identity, 1, keep_local=True)
            first = output / "parts/part01/chunks/00000.json.gz"
            before = first.read_bytes()
            self.assertFalse((output / "parts/part01/result.json").exists())
            recomputed = []
            def resume(frame, *args):
                recomputed.append(frame["sample_token"])
                return original(frame, *args)
            with mock.patch.object(data, "copy_archive", return_value=sensor), mock.patch.object(audit, "audit_frame", side_effect=resume):
                result = audit.process_part(1, sr, root, output, local, db, c, roles, identity, 1, keep_local=True)
            self.assertTrue(result["complete"])
            self.assertEqual(recomputed, ["frame1"])
            self.assertEqual(first.read_bytes(), before)

    def test_subthreshold_support_and_timing_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            c, roles, local, db, part, staged, *_ = tiny_dataset(Path(folder))
            frame = staged["keyframes"][0]
            frame["sensors"]["CAM_FRONT"]["timestamp"] += 51000
            record = audit.audit_frame(frame, part, local / "lidarseg", db, c, roles)
            self.assertEqual(len(record["camera_rows"]), 5)
            self.assertEqual(record["excluded"][0]["reason"], "camera_lidar_gap_over_limit")
            audit.validate_records([record], [frame["sample_token"]], roles)
            for row in record["camera_rows"]:
                row["objects"][0]["surface_points"] = {k: 4 for k in row["objects"][0]["surface_points"]}
            census = audit.Census(c); census.add(record, roles["scenes"]["scene-fixture"])
            cell = census.result()["cells"]["Vehicle:0-6m"]
            self.assertEqual(cell["surface_supported_annotation_observations"], 0)
            self.assertEqual(cell["centre_in_image_annotation_observations"], 1)


    def test_all_ten_part_summary_and_deterministic_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, original_roles, local, db, part, staged, *_ = tiny_dataset(root)
            template = [audit.audit_frame(f, part, local / "lidarseg", db, c, original_roles) for f in staged["keyframes"]]
            c.update(expected_scenes=10, expected_keyframes=20)
            output = root / "output"; identity = {"manifest_sha256": "fixture"}
            roles = {"scenes": {}, "previously_inspected_evaluation_logs": ["log4"]}
            receipts = {}
            for i in range(1, 11):
                scene = f"scene{i}"; log = f"log{i}"
                role = {1: "train", 2: "dev", 3: "excluded_previously_used_mini_log"}.get(i, "heldout_eval")
                roles["scenes"][scene] = {"role": role, "log_token": log}
                receipt = {"sha256": common.sha(str(i).encode())}
                receipts[f"v1.0-trainval{i:02d}_keyframes.tgz"] = receipt
                contract = audit.part_contract(i, receipt, identity)
                records = json.loads(json.dumps(template))
                for record in records:
                    record["sample_token"] = f"{i}_{record['sample_token']}"
                    record["lidar_token"] = f"{i}_{record['lidar_token']}"
                    record["scene"] = scene
                    for row in record["camera_rows"]:
                        row.update(scene=scene, recording_log=log, role=role, sample_token=record["sample_token"],
                                   camera_token=f"{i}_{row['camera_token']}")
                        for obj in row["objects"]:
                            obj["annotation_token"] = f"{i}_{obj['annotation_token']}"
                            obj["instance_token"] = f"{i}_{obj['instance_token']}"
                    record.pop("record_sha256"); record.update(audit.signed(record, "record_sha256"))
                tokens = [r["sample_token"] for r in records]
                raw = gzip.compress(common.encoded(audit.signed({"part_identity": contract, "records": records})), mtime=0)
                folder = output / f"parts/part{i:02d}"
                common.immutable(folder / "chunks/00000.json.gz", raw)
                result = audit.signed({"complete": True, "part_identity": contract, "sample_tokens": tokens,
                    "scenes": [scene], "rgb_headers_checked": 12, "chunks": [{"file": "chunks/00000.json.gz",
                    "sha256": common.sha(raw), "sample_tokens": tokens}]})
                common.immutable(folder / "result.json", common.encoded(result))
            report = audit.aggregate(output, c, roles, receipts, identity)
            self.assertTrue(report["complete"])
            self.assertEqual(report["unique_keyframes"], 20)
            self.assertEqual(report["rgb_headers_checked"], 120)
            self.assertEqual(report["groups"]["heldout_eval"]["keyframes"], 14)
            self.assertEqual(report["groups"]["heldout_uninspected_logs"]["keyframes"], 12)
            self.assertFalse(report["training_authorized"])
            common.immutable(output / "run_identity.json", common.encoded(identity))
            audit.bundle(output)
            first = common.file_sha(output / "nuscenes_0_6m_coverage_results.zip")
            audit.bundle(output)
            self.assertEqual(first, common.file_sha(output / "nuscenes_0_6m_coverage_results.zip"))

    def test_cleanup_only_owned_immediate_local_part(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); local = root / "local"; part = local / "parts/part01"
            archive = local / "archives/part.tgz"; contract = {"archive_sha256": "fixture"}
            common.immutable(part / "owner.json", common.encoded(contract))
            common.immutable(archive, b"fixture")
            owner = archive.with_name(archive.name + ".copy.json")
            common.immutable(owner, common.encoded({"sha256": "fixture"}))
            unrelated = root / "historical"; unrelated.write_bytes(b"preserve")
            with self.assertRaises(ValueError): data.release_part(local, local / "parts", archive, contract)
            self.assertTrue(archive.exists())
            data.release_part(local, part, archive, contract)
            self.assertFalse(part.exists()); self.assertFalse(archive.exists())
            self.assertEqual(unrelated.read_bytes(), b"preserve")

    def test_notebook_is_syntax_valid_cpu_only_and_self_contained(self):
        import ast
        notebook = json.loads((audit.ROOT / "notebooks/NuScenes_0_6m_Coverage_CPU_Colab.ipynb").read_text())
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]); ast.parse(source); sources.append(source)
                self.assertEqual(cell["outputs"], []); self.assertIsNone(cell["execution_count"])
        source = "\n".join(sources)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("--no-deps", source)
        self.assertNotIn("git reset", source)
        self.assertLess(source.index("def run("), source.index("run(['git'"))
        for name in ("prepare", "run", "bundle"):
            self.assertIn(f"stage('{name}')", source)

if __name__ == "__main__":
    unittest.main()
