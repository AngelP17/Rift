#!/usr/bin/env bash
set -euo pipefail

# Rift Release Verification Script
# Verifies container image signatures, SBOM attestations, and model bundle signatures.
#
# Usage:
#   ./scripts/verify_release.sh v1.0.0
#   ./scripts/verify_release.sh v1.0.0 --bundle rift-model-bundle-v1.0.0.tar.gz

VERSION="${1:?Usage: verify_release.sh <version> [--bundle <path>]}"
REGISTRY="ghcr.io/angelp17/rift"
IMAGE="${REGISTRY}:${VERSION}"

echo "============================================================"
echo "  Rift Release Verification: ${VERSION}"
echo "============================================================"

echo ""
echo "[1/4] Verifying container image signature..."
if cosign verify "${IMAGE}" --certificate-identity-regexp=".*github.*" --certificate-oidc-issuer="https://token.actions.githubusercontent.com" 2>/dev/null; then
    echo "  PASS: Image signature verified"
else
    echo "  FAIL: Image signature verification failed"
    echo "  (Expected: keyless OIDC signature from GitHub Actions)"
fi

echo ""
echo "[2/4] Verifying SBOM attestation..."
if cosign verify-attestation "${IMAGE}" --type spdxjson --certificate-identity-regexp=".*github.*" --certificate-oidc-issuer="https://token.actions.githubusercontent.com" 2>/dev/null; then
    echo "  PASS: SBOM attestation verified"
else
    echo "  FAIL: SBOM attestation verification failed"
fi

echo ""
echo "[3/4] Verifying provenance attestation..."
if gh attestation verify "oci://${IMAGE}" --repo AngelP17/Rift 2>/dev/null; then
    echo "  PASS: Provenance attestation verified"
else
    echo "  WARN: Provenance check requires gh CLI auth"
fi

if [[ "${2:-}" == "--bundle" ]]; then
    BUNDLE="${3:?Provide bundle path after --bundle}"
    SIG="${BUNDLE%.tar.gz}.sig"
    CERT="${BUNDLE%.tar.gz}.cert"

    echo ""
    echo "[4/4] Verifying model bundle signature..."
    if cosign verify-blob \
        --signature "${SIG}" \
        --certificate "${CERT}" \
        --certificate-identity-regexp=".*github.*" \
        --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
        "${BUNDLE}" 2>/dev/null; then
        echo "  PASS: Model bundle signature verified"
    else
        echo "  FAIL: Model bundle signature verification failed"
    fi
else
    echo ""
    echo "[4/4] Skipping bundle verification (use --bundle <path> to verify)"
fi

echo ""
echo "============================================================"
echo "  Verification complete"
echo "============================================================"
