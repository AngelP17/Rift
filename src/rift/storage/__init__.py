"""Storage abstractions for local and S3-compatible backends."""

from rift.storage.backends import StorageStatus, get_storage_backend

__all__ = ["StorageStatus", "get_storage_backend"]
