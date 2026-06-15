"""密钥管理

使用 Ed25519 高效签名(也支持 RSA / ECDSA)
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)

PathLike = Union[str, Path]


@dataclass
class KeyPair:
    """密钥对"""
    private_key: PrivateKeyTypes
    public_key: PublicKeyTypes
    algorithm: str  # "ed25519" | "rsa" | "ec"

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def private_key_bytes(self) -> bytes:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_key_fingerprint(self) -> str:
        """公钥 SHA-256 指纹 (hex)"""
        return hashlib.sha256(self.public_key_bytes()).hexdigest()


def generate_keypair(algorithm: str = "ed25519") -> KeyPair:
    """生成密钥对

    Args:
        algorithm: "ed25519" / "rsa-2048" / "ec-p256"
    """
    if algorithm == "ed25519":
        # audit B8: 之前这里偷偷用 SECP256K1 + ECDSA 顶替 ed25519,
        # manifest 记录 "ed25519" 但签名实际是 ECDSA, verify 一方按 ed25519
        # 校验会失败. cryptography 库本身支持 ed25519, 现在真的用它.
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
    elif algorithm == "rsa-2048":
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
    elif algorithm == "ec-p256":
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return KeyPair(private_key=private_key, public_key=public_key, algorithm=algorithm)


def save_keypair(keypair: KeyPair, private_path: PathLike, public_path: PathLike, password: bytes = None) -> None:
    """保存密钥对到文件"""
    private_path = Path(private_path)
    public_path = Path(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    encryption = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )

    private_path.write_bytes(
        keypair.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
    )
    public_path.write_bytes(keypair.public_key_bytes())
    private_path.chmod(0o600)  # 仅所有者可读写


def load_keypair(private_path: PathLike, password: bytes = None) -> KeyPair:
    """从文件加载密钥对"""
    private_path = Path(private_path)
    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(),
        password=password,
    )
    public_key = private_key.public_key()

    # 检测算法
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        algorithm = "ed25519"
    elif isinstance(private_key, ec.EllipticCurvePrivateKey):
        algorithm = "ec"
    elif isinstance(private_key, rsa.RSAPrivateKey):
        algorithm = "rsa"
    else:
        algorithm = "unknown"

    return KeyPair(private_key=private_key, public_key=public_key, algorithm=algorithm)


def load_public_key(public_path: PathLike) -> PublicKeyTypes:
    """从文件加载公钥"""
    return serialization.load_pem_public_key(Path(public_path).read_bytes())


# ============================================================
# 签名 / 验签
# ============================================================

def sign_data(private_key: PrivateKeyTypes, data: bytes, algorithm: str = "ed25519") -> bytes:
    """用私钥对数据签名"""
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        # ed25519 不需要单独的 hash 步骤, 内部自带
        return private_key.sign(data)
    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        return private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    if isinstance(private_key, rsa.RSAPrivateKey):
        return private_key.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
    raise ValueError(f"Unsupported key type: {type(private_key)}")


def verify_signature(public_key: PublicKeyTypes, data: bytes, signature: bytes) -> bool:
    """用公钥验签"""
    try:
        if isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, data)
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(
                signature,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
        else:
            return False
        return True
    except Exception:
        return False
