"""Transfer-only tests: no Drive, network, real models or held-out data."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from scripts import parking_v0_local_io as local
from scripts import parking_edge_common as common

spec = importlib.util.spec_from_file_location("local_io_v0_fixture", local.ROOT/"tests/test_parking_v0_dev.py")
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def data_fixture(root, count=4):
    m, c, _, points = fixture.make_fixture(root/"benchmark", count)
    common.immutable(root/"benchmark/benchmark_manifest.json", common.encoded(m))
    bundle = b"synthetic frozen bundle bytes"; c["adapter_bundle_sha256"] = common.sha(bundle)
    common.immutable(root/"model.zip", bundle)
    return m, c, points


def args_for_stage(root):
    return SimpleNamespace(inputs=root/"inputs", local_archive=root/"local_inputs.zip",
        archive=root/"remote/input_capsule.zip", benchmark=root/"benchmark", bundle=root/"model.zip")


def fake_process(command, **kwargs):
    """Execute the real scoring/packaging entry point with a synthetic predictor."""
    names = ["benchmark", "bundle", "output", "local", "source", "weights"]
    args = SimpleNamespace(**{name: Path(command[command.index("--"+name)+1]) for name in names}, device="cpu")
    predictor = fixture.Predictor(); predictor.registration = {"artifact_sha256": local.v0.configuration()["checkpoint_sha256"]}
    output = io.StringIO()
    with mock.patch.object(local.v0, "read_v0_bundle", return_value=({}, {})), mock.patch.object(local.v0, "V0Predictor", return_value=predictor):
        with contextlib.redirect_stdout(output): local.v0.evaluate(args)
    return SimpleNamespace(stdout=io.StringIO(output.getvalue()), wait=lambda: 0, terminate=lambda: None, poll=lambda: 0)


class LocalIOTests(unittest.TestCase):
    def test_numerical_evaluator_unchanged(self):
        self.assertEqual(common.file_sha(Path(local.v0.__file__)), local.EVALUATOR_SHA)

    def test_capsule_contains_only_dev_bodies_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _ = data_fixture(root); opened = []
            def read(name): opened.append(name); return (root/"benchmark"/name).read_bytes()
            members = local.capsule_members(m, (root/"model.zip").read_bytes(), read, c)
            self.assertEqual(len(opened), 4)
            self.assertTrue(all(n.startswith("shards/dev/") for n in opened))
            self.assertFalse(any("heldout_eval/" in n for n in members))
            capsule = root/"capsule.zip"; capsule.write_bytes(local.stored_zip(members))
            actual, receipt = local.inspect_capsule(capsule, c)
            self.assertEqual(actual, members); self.assertEqual(receipt["dev_images"], 4)

    def test_pack_once_then_local_resume_never_touches_remote(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _ = data_fixture(root); args = args_for_stage(root)
            with mock.patch.object(local.v0, "configuration", return_value=c): local.stage(args)
            self.assertTrue(args.archive.exists())
            before = args.local_archive.read_bytes()
            original_read = local.read_once
            def local_only(path):
                if Path(path).is_relative_to(root/"benchmark") or Path(path).is_relative_to(root/"remote"):
                    raise AssertionError("Do not read immutable remote inputs again")
                return original_read(path)
            with mock.patch.object(local.v0, "configuration", return_value=c), mock.patch.object(local, "read_once", side_effect=local_only), mock.patch.object(local, "copy_once", side_effect=AssertionError("No transfers on local resume")):
                local.stage(args)
            self.assertEqual(args.local_archive.read_bytes(), before)
            real_resolve=Path.resolve
            def resolve_only_local(path,*a,**kw):
                if path.is_relative_to(root/"remote") or path.is_relative_to(root/"benchmark") or path==root/"model.zip":
                    raise AssertionError("No remote metadata/symlink probes on a verified local capsule resume")
                return real_resolve(path,*a,**kw)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(Path,"resolve",new=resolve_only_local):local.stage(args)

    def test_new_vm_copies_only_one_saved_capsule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); _, c, _ = data_fixture(root); args = args_for_stage(root)
            with mock.patch.object(local.v0, "configuration", return_value=c): local.stage(args)
            args.local_archive = root/"new_vm/capsule.zip"; args.inputs = root/"new_vm/inputs"
            args.benchmark = args.bundle = None
            with mock.patch.object(local.v0, "configuration", return_value=c), mock.patch.object(local, "capsule_members", side_effect=AssertionError("Never rebuild from source")):
                local.stage(args)
            self.assertTrue((args.inputs/"benchmark/benchmark_manifest.json").is_file())

    def test_offline_upload_mode_never_requires_drive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _ = data_fixture(root)
            members = local.capsule_members(m, (root/"model.zip").read_bytes(), lambda n:(root/"benchmark"/n).read_bytes(), c)
            capsule = root/"uploaded.zip"; capsule.write_bytes(local.stored_zip(members))
            args = SimpleNamespace(inputs=root/"offline_inputs", local_archive=capsule, archive=None, benchmark=None, bundle=None)
            with mock.patch.object(local.v0, "configuration", return_value=c), mock.patch.object(local, "copy_once", side_effect=AssertionError("Offline means no transfer")):
                local.stage(args)
            self.assertTrue((args.inputs/"local_input_receipt.json").is_file())

    def test_altered_data_rejected_even_if_capsule_index_is_rehashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _ = data_fixture(root)
            members = local.capsule_members(m, (root/"model.zip").read_bytes(), lambda n:(root/"benchmark"/n).read_bytes(), c)
            name = next(n for n in members if n.startswith("benchmark/shards/")); members[name] = b"tampered"
            index = json.loads(members["input_inventory.json"]); index["files"][name] = {"sha256":common.sha(members[name]),"size_bytes":len(members[name])}
            members["input_inventory.json"] = common.encoded(index)
            capsule = root/"bad.zip"; capsule.write_bytes(local.stored_zip(members))
            with self.assertRaises(ValueError): local.inspect_capsule(capsule, c)

    def test_unexpected_heldout_body_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); m, c, _ = data_fixture(root)
            members = local.capsule_members(m, (root/"model.zip").read_bytes(), lambda n:(root/"benchmark"/n).read_bytes(), c)
            members["benchmark/shards/heldout_eval/secret.zip"] = b"do not use"
            capsule = root/"bad.zip"; capsule.write_bytes(local.stored_zip(members))
            with self.assertRaises(ValueError): local.inspect_capsule(capsule, c)

    def test_archive_traversal_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/"bad.zip"
            with zipfile.ZipFile(path,"w") as z:z.writestr("../escape",b"bad")
            with self.assertRaises(ValueError):local.read_archive(path)
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("same",b"one")
                with self.assertWarns(UserWarning):z.writestr("same",b"two")
            with self.assertRaises(ValueError):local.read_archive(path)

    def test_copy_cache_and_hash_failure_are_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source=root/"source.zip"; source.write_bytes(b"payload"); target=root/"copy.zip"
            with self.assertRaises(ValueError):local.copy_once(source,target,"wrong")
            self.assertFalse(target.exists()); self.assertFalse(list(root.glob(".v0_transfer_*")))
            digest=common.file_sha(source);self.assertEqual(local.copy_once(source,target,digest),digest)
            source.unlink();self.assertEqual(local.copy_once(source,target,digest),digest)

    def test_accidental_full_dataset_transfer_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"full_dataset.tgz";source.write_bytes(b"placeholder")
            actual_stat=Path.stat
            def stat(path,*a,**kw):
                if path==source:return SimpleNamespace(st_size=local.MAX_CAPSULE_BYTES+1)
                return actual_stat(path,*a,**kw)
            with mock.patch.object(Path,"stat",new=stat),mock.patch.object(Path,"read_bytes",side_effect=AssertionError("Must reject before reading")):
                with self.assertRaises(ValueError):local.copy_once(source,root/"destination.zip")
                with self.assertRaises(ValueError):local.read_archive(source)
            self.assertFalse((root/"destination.zip").exists())

    def test_drive_read_error_is_not_retried_or_claimed_as_confirmed_quota(self):
        with mock.patch.object(Path,"read_bytes",side_effect=OSError(5,"I/O error")) as read:
            with self.assertRaisesRegex(RuntimeError,"not confirmed"):local.read_once(Path("unreadable"))
            self.assertEqual(read.call_count,1)

    def test_input_publish_failure_keeps_local_inputs_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,c,_=data_fixture(root);args=args_for_stage(root)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local,"copy_once",side_effect=OSError(5,"Drive quota or mount")):
                local.stage(args)
            self.assertTrue((args.inputs/"benchmark/benchmark_manifest.json").exists())

    def prepare_run(self,root,count=4):
        m,c,_=data_fixture(root,count);args=args_for_stage(root)
        with mock.patch.object(local.v0,"configuration",return_value=c):local.stage(args)
        return m,c,SimpleNamespace(inputs=args.inputs,output=root/"results",local=root/"scratch",
            source=root/"source",weights=root/"weights",checkpoints=root/"remote_progress",checkpoint_every=4)

    def test_full_local_wrapper_and_archived_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m,c,args=self.prepare_run(root)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",side_effect=fake_process):local.run_local(args)
            self.assertEqual({p.name for p in args.checkpoints.glob("*.zip")},{"v0_progress_004.zip","v0_complete.zip"})
            restored=root/"restored"
            self.assertEqual(local.restore_checkpoint(args.checkpoints/"v0_complete.zip",restored,m,c),4)
            self.assertFalse((restored/"v0_dev_results.json").exists())  # Recompute, never trust cached metrics.
            self.assertEqual(len(list((restored/"predictions").glob("*.zip"))),4)
            args.output=restored
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",side_effect=fake_process):local.run_local(args)
            self.assertTrue((restored/"v0_dev_results.json").is_file())

    def test_child_running_ahead_cannot_change_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m,c,args=self.prepare_run(root,count=8)
            # fake_process produces all eight predictions before the parent
            # sees the first progress line, like a child running ahead.
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",side_effect=fake_process):local.run_local(args)
            with zipfile.ZipFile(args.checkpoints/"v0_progress_004.zip") as z:
                self.assertEqual(json.loads(z.read("checkpoint_inventory.json"))["prediction_shards"],4)
            with zipfile.ZipFile(args.checkpoints/"v0_progress_008.zip") as z:
                self.assertEqual(json.loads(z.read("checkpoint_inventory.json"))["prediction_shards"],8)
            self.assertEqual(local.latest_checkpoint(args.checkpoints).name,"v0_complete.zip")

    def test_checkpoint_drive_failure_does_not_abort_local_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,c,args=self.prepare_run(root)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",side_effect=fake_process),mock.patch.object(local,"copy_once",side_effect=OSError(5,"Drive I/O")) as save:
                local.run_local(args)
                self.assertEqual(save.call_count,1)
            self.assertTrue((args.output/"parking_v0_dev_results.zip").is_file())
            self.assertTrue((args.output/"portable_progress/v0_complete.zip").is_file())

    def test_wrapper_interruption_terminates_child_and_restores_signal_handler(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,c,args=self.prepare_run(root)
            class InterruptingStream:
                def __iter__(self):raise KeyboardInterrupt("Cell stopped")
                def close(self):pass
            stopped=[]
            process=SimpleNamespace(stdout=InterruptingStream(),poll=lambda:None,
                terminate=lambda:stopped.append("terminate"),wait=lambda:stopped.append("wait"))
            original=local.signal.getsignal(local.signal.SIGTERM)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",return_value=process):
                with self.assertRaises(KeyboardInterrupt):local.run_local(args)
            self.assertEqual(stopped,["terminate","wait"])
            self.assertEqual(local.signal.getsignal(local.signal.SIGTERM),original)

    def test_other_registration_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m,c,args=self.prepare_run(root)
            with mock.patch.object(local.v0,"configuration",return_value=c),mock.patch.object(local.subprocess,"Popen",side_effect=fake_process):local.run_local(args)
            bad=copy.deepcopy(c);bad["checkpoint_sha256"]="different model"
            with self.assertRaises(ValueError):local.restore_checkpoint(args.checkpoints/"v0_complete.zip",root/"other",m,bad)

    def test_local_first_notebook_syntax_and_io_guards(self):
        notebook=json.loads((local.ROOT/"notebooks/Parking_V0_Development_Baseline_CPU_Colab.ipynb").read_text());source=""
        for cell in notebook["cells"]:
            if cell["cell_type"]=="code":ast.parse(cell["source"]);source+=cell["source"]
        self.assertIn("USE_DRIVE = True",source);self.assertIn("LOCAL log:",source)
        self.assertIn("'stage-inputs'",source);self.assertIn("'evaluate-local'",source)
        self.assertNotIn("force_remount",source);self.assertNotIn("nvidia-smi",source)


if __name__=="__main__":unittest.main()
