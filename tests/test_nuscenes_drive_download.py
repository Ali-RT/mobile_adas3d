"""Local HTTP fixtures only; never download real datasets in tests."""
from contextlib import contextmanager
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import random
import tempfile
import threading
from pathlib import Path
import unittest
from unittest import mock

from scripts import download_nuscenes_to_drive as download
from scripts import parking_edge_common as common


@contextmanager
def server_fixture(data):
    state = {"data": data, "etag": '"fixture"', "ranges": [], "mode": "normal", "dropped": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["data"])))
            self.send_header("ETag", state["etag"])
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            header = self.headers.get("Range")
            state["ranges"].append(header)
            start = int(header.split("=")[1].split("-")[0]) if header else 0
            if state["mode"] == "ignore_range":
                start = 0
                self.send_response(200)
            else:
                self.send_response(206)
                returned_start = start + 1 if state["mode"] == "wrong_range" else start
                self.send_header("Content-Range", f"bytes {returned_start}-{len(state['data']) - 1}/{len(state['data'])}")
            raw = state["data"][start:]
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("ETag", state["etag"])
            self.end_headers()
            if state["mode"] == "drop_once" and not state["dropped"]:
                state["dropped"] = True
                self.wfile.write(raw[:192]); self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/fixture.tgz"
    try:
        yield state, url
    finally:
        server.shutdown(); server.server_close(); worker.join()


class DriveDownloadTests(unittest.TestCase):
    def setUp(self):
        self.raw = random.Random(12).randbytes(2048)
        self.payload = gzip.compress(self.raw, mtime=0)

    @contextmanager
    def fixture(self, payload=None):
        with tempfile.TemporaryDirectory() as folder, server_fixture(self.payload if payload is None else payload) as (state, url):
            source = {"name": "fixture.tgz", "url": url, "size_bytes": len(state["data"]),
                      "etag": state["etag"], "observed_sha256": None}
            # Allow localhost only inside this isolated HTTP test fixture.
            with mock.patch.object(download, "validate_url", lambda url: None), mock.patch.object(download, "CHUNK_BYTES", 128):
                yield Path(folder), source, state

    def test_all_ten_keyframes_and_full_variants_have_distinct_sizes(self):
        _, keyframes = download.load_sources(download.CONFIG)
        _, full = download.load_sources(download.CONFIG, "full")
        self.assertEqual(len(keyframes), 12)
        self.assertEqual(sum(s["size_bytes"] for s in keyframes), 45527157484)
        self.assertEqual(sum(s["size_bytes"] for s in full), 315049392354)
        self.assertEqual(sum(s["name"].endswith("_keyframes.tgz") for s in keyframes), 10)
        self.assertEqual(len(download.load_sources(download.CONFIG, parts=[2, 9])[1]), 4)
        for parts in [[0], [11], [1, 1]]:
            with self.assertRaises(ValueError):
                download.load_sources(download.CONFIG, parts=parts)

    def test_only_public_https_sources_accepted(self):
        for url in ["http://motional-nuscenes.s3.amazonaws.com/a", "https://example.com/a",
                    "https://user:password@motional-nuscenes.s3.amazonaws.com/a",
                    "https://motional-nuscenes.s3.amazonaws.com/a?token=secret"]:
            with self.assertRaises(ValueError):
                download.validate_url(url)

    def test_download_crc_receipt_and_verified_reuse(self):
        with self.fixture() as (root, source, state):
            receipt = download.download_one(root, source, retry_seconds=0)
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["decoded_bytes_checked_not_saved"], len(self.raw))
            self.assertEqual(receipt["sha256"], common.sha(self.payload))
            second = download.download_one(root, source)
            self.assertEqual(second, receipt)
            self.assertEqual(len(state["ranges"]), 1)

    def test_network_interruption_resumes_committed_prefix(self):
        with self.fixture() as (root, source, state):
            state["mode"] = "drop_once"
            with self.assertRaisesRegex(RuntimeError, "rerun this cell"):
                download.download_one(root, source, retries=0)
            p = download.owned_paths(root, source["name"])
            self.assertEqual(p["partial"].stat().st_size, 128)
            receipt = download.download_one(root, source, retry_seconds=0)
            self.assertEqual(state["ranges"], ["bytes=0-", "bytes=128-"])
            self.assertEqual(receipt["sha256"], common.sha(self.payload))

    def prepare_partial(self, root, source, tail=b""):
        p = download.owned_paths(root, source["name"])
        identity = download.source_identity(source)
        common.immutable(p["owner"], common.encoded(identity))
        chunk = self.payload[:128]
        state = {"owner_sha256": common.identity(identity), "chunks": [{"size_bytes": 128, "sha256": common.sha(chunk)}]}
        download.replace_state(p["state"], state)
        p["partial"].parent.mkdir(parents=True, exist_ok=True)
        p["partial"].write_bytes(chunk + tail)
        return p

    def test_only_uncommitted_owned_tail_is_discarded(self):
        with self.fixture() as (root, source, state):
            self.prepare_partial(root, source, b"interrupted-write")
            receipt = download.download_one(root, source)
            self.assertEqual(state["ranges"], ["bytes=128-"])
            self.assertEqual(receipt["sha256"], common.sha(self.payload))

    def test_ignored_or_wrong_range_never_appended(self):
        for mode in ["ignore_range", "wrong_range"]:
            with self.subTest(mode=mode), self.fixture() as (root, source, state):
                p = self.prepare_partial(root, source)
                before = p["partial"].read_bytes()
                state["mode"] = mode
                with self.assertRaises(RuntimeError):
                    download.download_one(root, source)
                self.assertEqual(p["partial"].read_bytes(), before)
                self.assertFalse(p["archive"].exists())

    def test_changed_committed_chunk_is_preserved_and_rejected(self):
        with self.fixture() as (root, source, state):
            p = self.prepare_partial(root, source)
            p["partial"].write_bytes(b"x" * 128)
            with self.assertRaisesRegex(RuntimeError, "Partial chunk 0 changed"):
                download.download_one(root, source)
            self.assertEqual(state["ranges"], [])
            self.assertEqual(p["partial"].read_bytes(), b"x" * 128)

    def test_changed_remote_etag_is_rejected_before_transfer(self):
        with self.fixture() as (root, source, state):
            state["etag"] = '"changed"'
            with self.assertRaisesRegex(RuntimeError, "Remote archive identity changed"):
                download.download_one(root, source)
            self.assertFalse((root / "archives").exists())

    def test_unowned_existing_archive_never_overwritten(self):
        with self.fixture() as (root, source, state):
            p = download.owned_paths(root, source["name"])
            p["archive"].parent.mkdir()
            p["archive"].write_bytes(b"user-data")
            with self.assertRaisesRegex(RuntimeError, "Unowned existing file"):
                download.download_one(root, source)
            self.assertEqual(p["archive"].read_bytes(), b"user-data")

    def test_compression_crc_failure_does_not_publish_archive(self):
        damaged = bytearray(self.payload); damaged[-8] ^= 1
        with self.fixture(bytes(damaged)) as (root, source, state):
            with self.assertRaises((gzip.BadGzipFile, EOFError)):
                download.download_one(root, source)
            p = download.owned_paths(root, source["name"])
            self.assertFalse(p["archive"].exists())
            self.assertFalse(p["receipt"].exists())
            self.assertTrue(p["partial"].exists())

    def test_previously_observed_sha_is_checked(self):
        with self.fixture() as (root, source, state):
            source["observed_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                download.download_one(root, source)
            self.assertFalse(download.owned_paths(root, source["name"])["receipt"].exists())

    def test_low_space_plan_creates_no_dataset_files(self):
        with self.fixture() as (root, source, state):
            plan = download.make_plan(root, [source], 0, 20)
            self.assertFalse(plan["space_check_passed"])
            self.assertEqual(list(root.iterdir()), [])

    def test_valid_receipt_but_modified_archive_rejected(self):
        with self.fixture() as (root, source, state):
            download.download_one(root, source)
            p = download.owned_paths(root, source["name"])
            p["archive"].write_bytes(b"x" * source["size_bytes"])
            with self.assertRaisesRegex(RuntimeError, "Existing complete archive changed"):
                download.download_one(root, source)

    def test_bzip2_crc_is_checked_without_extraction(self):
        import bz2
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "labels.tar.bz2"
            p.write_bytes(bz2.compress(self.raw))
            s = {"name": p.name, "size_bytes": p.stat().st_size, "observed_sha256": None}
            r = download.validate_payload(p, s)
            self.assertEqual(r["decoded_bytes_checked_not_saved"], len(self.raw))
            self.assertTrue(r["compression_crc_validated"])

    def test_killed_between_archive_and_receipt_can_recover(self):
        with self.fixture() as (root, source, state):
            original = common.immutable
            def interrupt_receipt(path, raw):
                if path.parent.name == "receipts":
                    raise RuntimeError("fixture receipt interruption")
                return original(path, raw)
            with mock.patch.object(common, "immutable", interrupt_receipt):
                with self.assertRaisesRegex(RuntimeError, "fixture receipt interruption"):
                    download.download_one(root, source)
            p = download.owned_paths(root, source["name"])
            self.assertTrue(p["archive"].exists())
            self.assertFalse(p["receipt"].exists())
            receipt = download.download_one(root, source)
            self.assertEqual(receipt["sha256"], common.sha(self.payload))
            self.assertEqual(len(state["ranges"]), 1)

    def test_symlinked_download_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "elsewhere").mkdir()
            (root / "archives").symlink_to(root / "elsewhere", target_is_directory=True)
            with self.assertRaises(RuntimeError):
                download.owned_paths(root, "fixture.tgz")

    def test_malformed_chunk_record_is_rejected_without_overwrite(self):
        with self.fixture() as (root, source, state):
            p = self.prepare_partial(root, source)
            value = json.loads(p["state"].read_text())
            value["chunks"][0]["size_bytes"] = 127
            p["state"].write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "Invalid partial chunk receipt"):
                download.download_one(root, source)
            self.assertEqual(p["partial"].read_bytes(), self.payload[:128])

    def test_invalid_storage_budget_is_rejected(self):
        with self.fixture() as (root, source, state):
            for free in [-1, float("nan"), float("inf")]:
                with self.assertRaises(ValueError):
                    download.make_plan(root, [source], free, 20)


if __name__ == "__main__":
    unittest.main()
