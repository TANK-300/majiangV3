"""验证 /debug/engine_status 暴露 model_version + bundle_hash（spec §6.2）。"""
from __future__ import annotations

from pathlib import Path

from backend.app.services.v3_service import V3SearchService


def test_status_returns_model_version_and_bundle_hash_keys():
    svc = V3SearchService()
    status = svc.status()
    # 不论 V3 是否加载成功，键都必须存在（值可能为 None）
    assert "model_version" in status
    assert "bundle_hash" in status


def test_compute_bundle_hash_stable(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "v3" / "agari_prob").mkdir(parents=True)
    (bundle / "v3" / "agari_prob" / "model.json").write_text('{"task":"agari_prob"}')
    h1 = V3SearchService._compute_bundle_hash(bundle)
    h2 = V3SearchService._compute_bundle_hash(bundle)
    assert h1 == h2  # deterministic
    assert len(h1) == 16


def test_compute_bundle_hash_changes_when_content_changes(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "v3" / "agari_prob").mkdir(parents=True)
    target = bundle / "v3" / "agari_prob" / "model.json"
    target.write_text('{"task":"agari_prob"}')
    h1 = V3SearchService._compute_bundle_hash(bundle)
    target.write_text('{"task":"agari_prob","extra":1}')
    h2 = V3SearchService._compute_bundle_hash(bundle)
    assert h1 != h2


def test_compute_bundle_hash_empty_dir_returns_hash(tmp_path):
    bundle = tmp_path / "empty"
    bundle.mkdir()
    h = V3SearchService._compute_bundle_hash(bundle)
    assert isinstance(h, str)
    assert len(h) == 16


def test_resolve_model_version_reads_version_file(tmp_path):
    bundle = tmp_path / "R3"
    bundle.mkdir()
    (bundle / "VERSION").write_text("3.0.0-r3\n")
    assert V3SearchService._resolve_model_version(bundle) == "3.0.0-r3"


def test_resolve_model_version_falls_back_to_dir_name(tmp_path):
    bundle = tmp_path / "R2"
    bundle.mkdir()
    assert V3SearchService._resolve_model_version(bundle) == "R2"
