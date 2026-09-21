#!/usr/bin/env bash
# Unsigned device IPA for SideStore / LiveContainer. Run on macOS with Xcode 16+.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen is required. Install with: brew install xcodegen" >&2
  exit 1
fi

xcodegen generate

DERIVED="${ROOT}/build/DerivedData"
rm -rf "${ROOT}/build/Payload" "${ROOT}/build/Volocal.ipa"
mkdir -p "${DERIVED}" "${ROOT}/build"

xcodebuild \
  -project Volocal.xcodeproj \
  -scheme Volocal \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "${DERIVED}" \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  DEVELOPMENT_TEAM="" \
  ONLY_ACTIVE_ARCH=NO \
  build

APP="$(find "${DERIVED}/Build/Products" -path '*iphoneos*' -name 'Volocal.app' | head -n 1)"
if [[ -z "${APP}" || ! -d "${APP}" ]]; then
  echo "Volocal.app was not produced. xcodebuild output is under ${DERIVED}" >&2
  exit 1
fi

mkdir -p "${ROOT}/build/Payload"
cp -R "${APP}" "${ROOT}/build/Payload/Volocal.app"

# SideStore re-signs; keep the entitlement so GetMoreRAM / increased-memory-limit survives.
if [[ -f "${ROOT}/Volocal/Volocal.entitlements" ]]; then
  cp "${ROOT}/Volocal/Volocal.entitlements" "${ROOT}/build/Payload/Volocal.app/" || true
fi

(
  cd "${ROOT}/build"
  rm -f Volocal.ipa
  zip -qry Volocal.ipa Payload
)

echo "IPA: ${ROOT}/build/Volocal.ipa"
