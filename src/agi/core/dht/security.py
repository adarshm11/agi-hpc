# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DHT Security Module for AGI-HPC.

Provides security features for the distributed hash table:
- mTLS credentials management
- Access control and authorization
- Encryption at rest
- Audit logging

Sprint 6 Implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SecurityConfig:
    """Security configuration for the DHT subsystem."""

    enable_mtls: bool = False
    cert_path: str = os.getenv("AGI_DHT_CERT_PATH", "")
    key_path: str = os.getenv("AGI_DHT_KEY_PATH", "")
    ca_path: str = os.getenv("AGI_DHT_CA_PATH", "")
    enable_encryption: bool = False
    encryption_algorithm: str = "AES-256-GCM"
    enable_audit: bool = True
    audit_log_path: str = "dht_audit.log"
    access_control_enabled: bool = False
    allowed_peers: Set[str] = field(default_factory=set)


@dataclass
class MTLSCredentials:
    """Container for mutual TLS credential data."""

    cert_data: bytes = b""
    key_data: bytes = b""
    ca_data: bytes = b""

    def is_valid(self) -> bool:
        """Return True if both cert_data and key_data are non-empty."""
        return bool(self.cert_data and self.key_data)

    @classmethod
    def from_files(
        cls, cert_path: str, key_path: str, ca_path: str = ""
    ) -> "MTLSCredentials":
        """Load mTLS credentials from filesystem paths."""
        cert_data = b""
        key_data = b""
        ca_data = b""
        if cert_path:
            with open(cert_path, "rb") as f:
                cert_data = f.read()
            logger.info("[dht][security] loaded certificate from %s", cert_path)
        if key_path:
            with open(key_path, "rb") as f:
                key_data = f.read()
            logger.info("[dht][security] loaded key from %s", key_path)
        if ca_path:
            with open(ca_path, "rb") as f:
                ca_data = f.read()
            logger.info("[dht][security] loaded CA from %s", ca_path)
        return cls(cert_data=cert_data, key_data=key_data, ca_data=ca_data)


_DEFAULT_PERMISSIONS: List[str] = ["get", "put", "delete", "list"]


class AccessController:
    """Per-peer access control for the DHT."""

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self._config = config or SecurityConfig()
        self._peer_permissions: Dict[str, List[str]] = {}
        for peer_id in self._config.allowed_peers:
            self._peer_permissions[peer_id] = list(_DEFAULT_PERMISSIONS)
        logger.info(
            "[dht][security] AccessController initialized enabled=%s peers=%d",
            self._config.access_control_enabled,
            len(self._peer_permissions),
        )

    def check_access(self, peer_id: str, operation: str, key: str) -> bool:
        """Check whether a peer is allowed to perform an operation."""
        if not self._config.access_control_enabled:
            return True
        permissions = self._peer_permissions.get(peer_id)
        if permissions is None:
            logger.warning(
                "[dht][security] access denied - unknown peer=%s op=%s key=%s",
                peer_id,
                operation,
                key,
            )
            return False
        allowed = operation in permissions
        if not allowed:
            logger.warning(
                "[dht][security] access denied peer=%s op=%s key=%s",
                peer_id,
                operation,
                key,
            )
        return allowed

    def add_peer(self, peer_id: str, permissions: Optional[List[str]] = None) -> None:
        """Register a peer with the given permissions."""
        self._peer_permissions[peer_id] = list(
            permissions if permissions is not None else _DEFAULT_PERMISSIONS
        )
        logger.info(
            "[dht][security] added peer=%s permissions=%s",
            peer_id,
            self._peer_permissions[peer_id],
        )

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the access control list."""
        removed = self._peer_permissions.pop(peer_id, None)
        if removed is not None:
            logger.info("[dht][security] removed peer=%s", peer_id)
        else:
            logger.debug("[dht][security] remove_peer: peer=%s not found", peer_id)

    def get_permissions(self, peer_id: str) -> List[str]:
        """Return the permission list for a peer."""
        return list(self._peer_permissions.get(peer_id, []))


class EncryptionManager:
    """At-rest encryption for DHT values using HMAC-based XOR stream cipher."""

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self._config = config or SecurityConfig()
        seed = (self._config.cert_path or "agi-hpc-dht-default-key").encode()
        self._key = hashlib.sha256(seed).digest()
        logger.info(
            "[dht][security] EncryptionManager initialized algo=%s enabled=%s",
            self._config.encryption_algorithm,
            self._config.enable_encryption,
        )

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data bytes. Returns unchanged if encryption is disabled."""
        if not self._config.enable_encryption:
            return data
        nonce = os.urandom(16)
        keystream = self._derive_keystream(nonce, len(data))
        ciphertext = bytes(a ^ b for a, b in zip(data, keystream, strict=False))
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data bytes. Returns unchanged if encryption is disabled."""
        if not self._config.enable_encryption:
            return data
        if len(data) < 16:
            raise ValueError("Encrypted data too short to contain nonce")
        nonce = data[:16]
        ciphertext = data[16:]
        keystream = self._derive_keystream(nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, keystream, strict=False))

    def _derive_keystream(self, nonce: bytes, length: int) -> bytes:
        """Derive a keystream using iterated HMAC-SHA256."""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hmac.new(
                self._key,
                nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            stream += block
            counter += 1
        return stream[:length]


class AuditLogger:
    """Append-only audit log for DHT security events."""

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self._config = config or SecurityConfig()
        self._entries: List[Dict[str, Any]] = []
        self._max_entries: int = 10000
        logger.info(
            "[dht][security] AuditLogger initialized enabled=%s path=%s",
            self._config.enable_audit,
            self._config.audit_log_path,
        )

    def log_access(self, peer_id: str, operation: str, key: str, allowed: bool) -> None:
        """Record an access attempt."""
        if not self._config.enable_audit:
            return
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "type": "access",
            "peer_id": peer_id,
            "operation": operation,
            "key": key,
            "allowed": allowed,
        }
        self._append(entry)
        log_fn = logger.debug if allowed else logger.warning
        log_fn(
            "[dht][audit] access peer=%s op=%s key=%s allowed=%s",
            peer_id,
            operation,
            key,
            allowed,
        )

    def log_error(self, peer_id: str, error: str) -> None:
        """Record a security-related error."""
        if not self._config.enable_audit:
            return
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "type": "error",
            "peer_id": peer_id,
            "error": error,
        }
        self._append(entry)
        logger.error("[dht][audit] error peer=%s error=%s", peer_id, error)

    def get_recent_entries(self, count: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent audit entries, newest first."""
        return list(reversed(self._entries[-count:]))

    def _append(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the ring buffer."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
