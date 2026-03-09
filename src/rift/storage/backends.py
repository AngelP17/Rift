from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from rift.utils.config import RiftPaths


@dataclass(frozen=True)
class StorageStatus:
    backend: str
    available: bool
    bucket: str | None
    endpoint: str | None
    root_path: str
    details: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StorageBackend(Protocol):
    def save_parquet(self, frame: pl.DataFrame, object_name: str) -> str: ...
    def load_parquet(self, object_name: str) -> pl.DataFrame: ...
    def exists(self, object_name: str) -> bool: ...
    def status(self) -> StorageStatus: ...


class LocalStorageBackend:
    def __init__(self, paths: RiftPaths) -> None:
        self.paths = paths

    def _target(self, object_name: str) -> Path:
        target = self.paths.storage_dir / object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_parquet(self, frame: pl.DataFrame, object_name: str) -> str:
        target = self._target(object_name)
        frame.write_parquet(target)
        return str(target)

    def load_parquet(self, object_name: str) -> pl.DataFrame:
        return pl.read_parquet(self._target(object_name))

    def exists(self, object_name: str) -> bool:
        return self._target(object_name).exists()

    def status(self) -> StorageStatus:
        return StorageStatus(
            backend="local",
            available=True,
            bucket=None,
            endpoint=None,
            root_path=str(self.paths.storage_dir),
            details="Local filesystem storage is active.",
        )


class MinioStorageBackend:
    def __init__(self, paths: RiftPaths) -> None:
        from minio import Minio

        self.paths = paths
        self.endpoint = os.getenv("RIFT_MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("RIFT_MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("RIFT_MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("RIFT_MINIO_BUCKET", "rift-data")
        self.secure = os.getenv("RIFT_MINIO_SECURE", "false").lower() == "true"
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def _temp_path(self, object_name: str) -> Path:
        temp = self.paths.storage_dir / "tmp" / object_name
        temp.parent.mkdir(parents=True, exist_ok=True)
        return temp

    def save_parquet(self, frame: pl.DataFrame, object_name: str) -> str:
        self.ensure_bucket()
        temp_path = self._temp_path(object_name)
        frame.write_parquet(temp_path)
        self.client.fput_object(self.bucket, object_name, str(temp_path))
        temp_path.unlink(missing_ok=True)
        return f"s3://{self.bucket}/{object_name}"

    def load_parquet(self, object_name: str) -> pl.DataFrame:
        self.ensure_bucket()
        temp_path = self._temp_path(object_name)
        self.client.fget_object(self.bucket, object_name, str(temp_path))
        frame = pl.read_parquet(temp_path)
        temp_path.unlink(missing_ok=True)
        return frame

    def exists(self, object_name: str) -> bool:
        self.ensure_bucket()
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except Exception:
            return False

    def status(self) -> StorageStatus:
        try:
            self.ensure_bucket()
            return StorageStatus(
                backend="minio",
                available=True,
                bucket=self.bucket,
                endpoint=self.endpoint,
                root_path=str(self.paths.storage_dir),
                details="MinIO-compatible object storage is active.",
            )
        except Exception as exc:  # pragma: no cover - depends on external service
            return StorageStatus(
                backend="minio",
                available=False,
                bucket=self.bucket,
                endpoint=self.endpoint,
                root_path=str(self.paths.storage_dir),
                details=f"MinIO unavailable: {exc}",
            )


def get_storage_backend(paths: RiftPaths) -> StorageBackend:
    backend = os.getenv("RIFT_STORAGE_BACKEND", "local").strip().lower()
    if backend == "minio":
        return MinioStorageBackend(paths)
    return LocalStorageBackend(paths)
