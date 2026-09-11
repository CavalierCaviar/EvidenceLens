import pytest
from app.security import hash_password, verify_password, create_access_token, decode_access_token

def test_hash_password_creates_different_hash():
    password = "secretpassword"
    hash1 = hash_password(password)
    hash2 = hash_password(password)
    assert hash1 != hash2
    assert hash1 != password

def test_verify_password_correct():
    password = "secretpassword"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True

def test_verify_password_wrong():
    password = "secretpassword"
    hashed = hash_password(password)
    assert verify_password("wrongpassword", hashed) is False

def test_create_and_decode_access_token():
    user_id = 1
    token = create_access_token({"sub": str(user_id)})
    payload = decode_access_token(token)
    assert payload is not None
    assert payload.get("sub") == str(user_id)

def test_decode_access_token_invalid():
    invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
    payload = decode_access_token(invalid_token)
    assert payload is None
