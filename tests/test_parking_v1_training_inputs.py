"""CPU synthetic V1 data preparation tests; no models or real downloads."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import prepare_parking_v1_training as prepare
from scripts import parking_edge_common as common
from scripts import audit_nuscenes_coverage_0_6m as coverage

spec = importlib.util.spec_from_file_location("v1_coverage_fixture", prepare.ROOT/"tests/test_nuscenes_full_coverage.py")
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def raw_fixture(root):
    cov, roles, source, database, part, staged, archive, receipt, contract = fixture.tiny_dataset(root)
    c = prepare.configuration(); c.update(maximum_train_images=16, images_per_shard=2,
        readiness_minimum_tracks_per_cell=1, readiness_minimum_logs_per_cell=1)
    p = prepare.data_protocol(c)
    rows = []
    for frame in staged["keyframes"]:
        for row in coverage.audit_frame(frame, part, source/"lidarseg", database, cov, roles)["camera_rows"]:
            rows.append({**{k:row[k] for k in ["camera_token","sample_token","scene","role","recording_log"]},
                "source_part":1,"coverage_row":row,"coverage_row_sha256":common.identity(row),
                "supported_counts":prepare.reference.supported_counts(row,p)})
    identity = coverage.signed({"test":"V1 inputs"})
    selected = prepare.select_train(rows,c)
    selection = coverage.signed({"preparation_identity_sha256":identity["manifest_sha256"],"samples":selected,
        "available_pool":prepare.support(rows,c,p),"selected_pool":prepare.support(selected,c,p)})
    dataset = root/"drive_dataset"; (dataset/"archives").mkdir(parents=True)
    shutil.copy2(archive,dataset/"archives"/receipt["file"])
    local = root/"new_local"; shutil.copytree(source/"lidarseg",local/"lidarseg")
    a = SimpleNamespace(dataset=dataset,local=local,output=root/"remote",workers=2)
    common.immutable(local/"preparation_identity.json",common.encoded(identity))
    common.immutable(local/"selection_manifest.json",common.encoded(selection))
    return a,c,p,roles,identity,selection,cov,database,{"v1.0-trainval01_keyframes.tgz":receipt}


class ContractTests(unittest.TestCase):
    def test_preparation_only_contract(self):
        c = prepare.configuration()
        self.assertEqual(c["candidate_role"],"train")
        self.assertEqual(c["maximum_train_images"],4096)
        self.assertIn("Vehicle:0-2m",c["selection_cells"])
        for key in ["training_authorized","model_inference_authorized","heldout_evaluation_authorized","phone_app_changed"]:
            self.assertFalse(c[key])
        self.assertIsNone(c["accuracy_threshold_m"])

    def test_no_model_imports(self):
        tree = ast.parse(Path(prepare.__file__).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): names.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom): names.append(node.module or "")
        self.assertFalse(any(n.startswith(("torch","ultralytics","models")) for n in names))

    def test_notebook_compiles_cpu_revision_and_local_logs(self):
        n = json.loads((prepare.ROOT/"notebooks/Parking_V1_Training_Inputs_CPU_Colab.ipynb").read_text())
        all_code = ""
        for i,cell in enumerate(n["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"]); compile(code,f"cell{i}","exec"); all_code += code
        self.assertIn(prepare.configuration()["revision"],all_code)
        self.assertIn("ALLOW_ONE_TIME_SMALL_DRIVE_READS = False",all_code)
        self.assertIn("LOGS = Path('/content')",all_code)
        self.assertNotIn("reset', '--hard",all_code)

    def test_owner_reentry_and_unowned_scratch(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); a=SimpleNamespace(dataset=root/"data",coverage=root/"coverage",output=root/"out",local=root/"local")
            prepare.initialize_owner(a,prepare.configuration()); prepare.initialize_owner(a,prepare.configuration())
            a.local=root/"unowned"; a.local.mkdir(); (a.local/"keep.txt").write_bytes(b"keep")
            with self.assertRaises(ValueError): prepare.initialize_owner(a,prepare.configuration())


class TransferTests(unittest.TestCase):
    def snapshot_fixture(self, root):
        c=prepare.configuration();local=root/"local";local.mkdir()
        receipt={"sha256":"sensor_source"}
        cov={"manifest_sha256":"coverage","config":{"expected_keyframes":1},"download_manifest_sha256":"downloads"}
        contract=coverage.part_contract(1,receipt,cov)
        chunk_raw=b"reviewed opaque compressed chunk"
        chunk={"file":"chunks/00000.json.gz","sample_tokens":["frame"],"sha256":common.sha(chunk_raw)}
        parts={1:coverage.signed({"complete":True,"part_identity":contract,"chunks":[chunk],"sample_tokens":["frame"]})}
        path=root/"folder.zip"
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("folder/"+prepare.SUMMARY_NAME,b"reviewed_summary")
            z.writestr("folder/parts/part01/chunks/00000.json.gz",chunk_raw)
        manifest=root/"download.json";manifest.write_bytes(b"reviewed_downloads")
        a=SimpleNamespace(local=local,output=root/"remote",coverage=root/"unused_coverage",dataset=root/"unused_dataset",
            coverage_archive=[path],download_manifest=manifest,allow_one_time_small_drive_reads=False)
        receipts={"v1.0-trainval01_keyframes.tgz":receipt}
        return a,c,cov,parts,receipts

    def test_snapshot_upload_then_same_vm_no_remote_reads(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,cov,parts,receipts=self.snapshot_fixture(Path(f))
            with mock.patch.object(prepare,"summary_catalog",return_value=(cov,{},parts)),mock.patch.object(prepare,"download_catalog",return_value=({"manifest_sha256":"downloads"},receipts)):
                path,_=prepare.snapshot(a,c)
                self.assertTrue(path.exists())
                with mock.patch.object(prepare,"UploadedCoverage",side_effect=AssertionError("No source reopens")),mock.patch.object(prepare,"atomic_copy",side_effect=AssertionError("No remote transfer")):
                    self.assertEqual(prepare.snapshot(a,c)[0],path)

    def test_new_vm_snapshot_one_archive_read_only(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,cov,parts,receipts=self.snapshot_fixture(Path(f))
            with mock.patch.object(prepare,"summary_catalog",return_value=(cov,{},parts)),mock.patch.object(prepare,"download_catalog",return_value=({"manifest_sha256":"downloads"},receipts)):
                prepare.snapshot(a,c);a.local=Path(f)/"new_vm"
                original=prepare.atomic_copy
                calls=[]
                def copy_once(source,target,*args,**kwargs):
                    calls.append((source,target));return original(source,target,*args,**kwargs)
                with mock.patch.object(prepare,"atomic_copy",side_effect=copy_once),mock.patch.object(prepare,"UploadedCoverage",side_effect=AssertionError("No upload needed")):
                    prepare.snapshot(a,c)
                self.assertEqual(len(calls),1)

    def test_snapshot_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);a,c,cov,parts,receipts=self.snapshot_fixture(root)
            with mock.patch.object(prepare,"summary_catalog",return_value=(cov,{},parts)),mock.patch.object(prepare,"download_catalog",return_value=({"manifest_sha256":"downloads"},receipts)):
                path,_=prepare.snapshot(a,c);before=path.read_bytes()
                a.local=root/"independent_local";a.local.mkdir();a.output=root/"independent_remote"
                path,_=prepare.snapshot(a,c)
                self.assertEqual(path.read_bytes(),before)

    def test_repacked_changed_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,cov,parts,receipts=self.snapshot_fixture(Path(f))
            with mock.patch.object(prepare,"summary_catalog",return_value=(cov,{},parts)),mock.patch.object(prepare,"download_catalog",return_value=({"manifest_sha256":"downloads"},receipts)):
                path,_=prepare.snapshot(a,c)
                with zipfile.ZipFile(path) as z:members={n:z.read(n) for n in z.namelist()}
                name="parts/part01/chunks/00000.json.gz";members[name]=b"changed"
                index=json.loads(members["snapshot_inventory.json"]);index["files"][name]=common.sha(members[name])
                members["snapshot_inventory.json"]=common.encoded(index)
                bad=Path(f)/"bad.zip";bad.write_bytes(prepare.reference.deterministic_zip(members))
                with self.assertRaises(ValueError):prepare.inspect_snapshot(bad,c)

    def test_atomic_copy_preserves_complete_conflicts(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); src=root/"src"; dst=root/"dst"; src.write_bytes(b"data")
            prepare.atomic_copy(src,dst,common.sha(b"data")); self.assertEqual(dst.read_bytes(),b"data")
            src.write_bytes(b"changed")
            with self.assertRaises(ValueError): prepare.atomic_copy(src,dst)
            self.assertEqual(dst.read_bytes(),b"data")

    def test_acknowledged_publish_does_not_reopen_remote(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); src=root/"src"; src.write_bytes(b"data"); target=root/"remote"/"out.zip"
            prepare.publish(src,target,root/"local")
            with mock.patch.object(prepare,"atomic_copy",side_effect=AssertionError("No repeat transfer")):
                prepare.publish(src,target,root/"local")

    def test_oversized_transfer_fails_before_target(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); src=root/"src"; src.write_bytes(b"large")
            with self.assertRaises(ValueError): prepare.atomic_copy(src,root/"dst",maximum=1)
            self.assertFalse((root/"dst").exists())

    def test_uploaded_folder_suffixes_and_duplicates(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); path=root/"upload.zip"
            with zipfile.ZipFile(path,"w") as z: z.writestr("folder/parts/p/chunk.gz",b"raw")
            upload=prepare.UploadedCoverage([path],1000)
            try:self.assertEqual(upload.read("parts/p/chunk.gz",10),b"raw")
            finally:upload.close()
            upload=prepare.UploadedCoverage([path,path],1000)
            try:
                with self.assertRaises(ValueError):upload.read("parts/p/chunk.gz",10)
            finally:upload.close()

    def test_uploaded_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/"upload.zip"
            with zipfile.ZipFile(path,"w") as z:z.writestr("../bad",b"x")
            with self.assertRaises(ValueError):prepare.UploadedCoverage([path],1000)

    def test_missing_uploaded_snapshot_stops_without_small_reads(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f); root.joinpath("local").mkdir()
            a=SimpleNamespace(local=root/"local",output=root/"remote",coverage=root/"coverage",coverage_archive=[],allow_one_time_small_drive_reads=False)
            with mock.patch.object(prepare,"read_once",side_effect=AssertionError("No Drive read")):
                with self.assertRaises(FileNotFoundError):prepare.snapshot(a,prepare.configuration())


class PreparationTests(unittest.TestCase):
    def test_train_only_rare_views_and_deterministic_selection(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,*_=raw_fixture(Path(f))
            prepare.validate_selection(selection,identity,c,roles,p)
            self.assertEqual(selection["samples"],prepare.select_train(list(reversed(selection["samples"])),c))
            rare=[s for s in selection["samples"] if s["supported_counts"]["Vehicle:0-1m"]]
            self.assertEqual(len(rare),6)
            bad=copy.deepcopy(selection);bad["samples"][0]["role"]="dev"
            with self.assertRaises(ValueError):prepare.validate_selection(bad,identity,c,roles,p)

    def test_changed_log_and_counts_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,*_=raw_fixture(Path(f))
            row=copy.deepcopy(selection["samples"][0]);row["recording_log"]="heldout_log"
            with self.assertRaises(ValueError):prepare.validate_row(row,c,roles,p)
            row=copy.deepcopy(selection["samples"][0]);row["supported_counts"]["Vehicle:0-1m"]+=1
            with self.assertRaises(ValueError):prepare.validate_row(row,c,roles,p)

    def test_rare_budget_overflow_fails(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,*_=raw_fixture(Path(f));c["maximum_train_images"]=2
            with self.assertRaises(ValueError):prepare.select_train(selection["samples"],c)

    def test_missing_cells_are_not_declared_ready(self):
        with tempfile.TemporaryDirectory() as f:
            *_,selection,cov,db,receipts=raw_fixture(Path(f))
            self.assertFalse(selection["selected_pool"]["six_cell_data_ready"])

    def test_broad_range_and_duplicate_pixels_rejected(self):
        sample={"width":100,"height":60}
        points={"xy":np.array([[1,1]],np.int32),"z":np.array([3],np.float32)}
        prepare.validate_broad(points,sample); points["z"][0]=1
        with self.assertRaises(ValueError):prepare.validate_broad(points,sample)
        points={"xy":np.array([[1,1],[1,1]],np.int32),"z":np.array([3,4],np.float32)}
        with self.assertRaises(ValueError):prepare.validate_broad(points,sample)

    def test_sensor_archives_to_train_parts_and_final_report(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,cov,db,receipts=raw_fixture(Path(f))
            value=prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,db)
            final=a.local/"train_parts/train_part01.zip"
            with zipfile.ZipFile(final) as z:
                self.assertTrue(all(i.date_time==(2026,10,29,0,0,0) for i in z.infolist()))
                first=value["shards"][0]; inner=zipfile.ZipFile(io.BytesIO(z.read(first["file"])))
                sample=first["samples"][0]
                with np.load(io.BytesIO(inner.read(sample["ground_truth_member"])),allow_pickle=False) as gt:
                    self.assertTrue((gt["z"]<2).any())
                with np.load(io.BytesIO(inner.read(sample["broad_member"])),allow_pickle=False) as gt:
                    self.assertTrue((gt["z"]>=2).all())
            report=prepare.finalize(a,selection,c,identity,p)
            self.assertTrue(report["complete"]);self.assertFalse(report["training_performed"])
            self.assertFalse(report["training_data_ready"])
            self.assertTrue((a.output/"parking_v1_training_inputs_results.zip").exists())
            final.unlink()
            with mock.patch.object(prepare.data,"copy_archive",side_effect=AssertionError("No sensor restage")):
                self.assertEqual(prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,None),value)

    def test_interrupted_part_reuses_complete_local_shard(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,cov,db,receipts=raw_fixture(Path(f))
            original=prepare.broad_reference;calls=[0]
            def fail(*args,**kwargs):
                calls[0]+=1
                if calls[0]==3:raise RuntimeError("interruption")
                return original(*args,**kwargs)
            a.workers=1
            with mock.patch.object(prepare,"broad_reference",side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,"interruption"):
                    prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,db)
            first=a.local/"parts/part01/shards/train/part01_000.zip";before=first.read_bytes()
            prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,db)
            with zipfile.ZipFile(a.local/"train_parts/train_part01.zip") as z:
                self.assertEqual(z.read("shards/train/part01_000.zip"),before)

    def test_changed_completed_part_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,cov,db,receipts=raw_fixture(Path(f))
            prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,db)
            final=a.local/"train_parts/train_part01.zip";final.write_bytes(b"preserve conflict")
            with self.assertRaises((ValueError,zipfile.BadZipFile)):
                prepare.prepare_part(a,1,selection,c,identity,p,receipts,cov,None)
            self.assertEqual(final.read_bytes(),b"preserve conflict")

    def test_existing_selection_repaired_without_chunk_rescan(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,*_=raw_fixture(Path(f))
            with mock.patch.object(prepare.zipfile,"ZipFile",side_effect=AssertionError("No coverage rescan")):
                # The local plan exists but its archive publication was interrupted.
                # deterministic_zip itself needs ZipFile, so mock the writer only here.
                with mock.patch.object(prepare.reference,"deterministic_zip",return_value=b"portable selection"):
                    actual=prepare.plan(a,c,Path(f)/"absent",None,identity,roles,p)
            self.assertEqual(actual,selection)
            self.assertTrue((a.output/"selection_state.zip").exists())

    def test_insufficient_plan_stops_before_sensor_work(self):
        with tempfile.TemporaryDirectory() as f:
            a,c,p,roles,identity,selection,*_=raw_fixture(Path(f))
            self.assertFalse(selection["selected_pool"]["six_cell_data_ready"])
            for row in selection["samples"]:
                self.assertEqual(row["role"],"train")
                self.assertEqual(roles["scenes"][row["scene"]]["role"],"train")
            argv=["prepare_v1","prepare","--dataset",str(a.dataset),"--coverage",str(Path(f)/"coverage"),
                "--output",str(a.output),"--local",str(a.local)]
            with mock.patch.object(prepare,"configuration",return_value=c),mock.patch.object(prepare,"initialize_owner"),mock.patch.object(prepare,"initialize",return_value=(None,None,identity,roles,p)),mock.patch.object(prepare,"plan",return_value=selection),mock.patch.object(prepare.coverage,"prepare_support",side_effect=AssertionError("No sensor work")),mock.patch.object(sys,"argv",argv),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"minimum six-cell diversity"):
                    prepare.main()


if __name__=="__main__":unittest.main()
