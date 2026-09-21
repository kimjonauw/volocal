#!/usr/bin/env bash
# Ad-hoc-signed device IPA for SideStore. LiveContainer guests are unsupported
# (iOS 26 crash / broken mic AEC). Run on macOS with Xcode 16+.
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
  ALWAYS_EMBED_SWIFT_STANDARD_LIBRARIES=YES \
  build

APP="$(find "${DERIVED}/Build/Products" -path '*iphoneos*' -name 'Volocal.app' | head -n 1)"
if [[ -z "${APP}" || ! -d "${APP}" ]]; then
  echo "Volocal.app was not produced. xcodebuild output is under ${DERIVED}" >&2
  exit 1
fi

FRAMEWORKS="${APP}/Frameworks"
mkdir -p "${FRAMEWORKS}"

# Transitive SPM binary targets (llama, NemoTextProcessing / libtext_processing_rs)
# are skipped when CODE_SIGNING_ALLOWED=NO. Copy ios-arm64 slices into the app.
if [[ -d "${DERIVED}/SourcePackages/artifacts" ]]; then
  while IFS= read -r -d '' src; do
    name="$(basename "${src}")"
    dest="${FRAMEWORKS}/${name}"
    if [[ -e "${dest}" ]]; then
      continue
    fi
    echo "Embedding ${name} from ${src}"
    cp -R "${src}" "${dest}"
  done < <(find "${DERIVED}/SourcePackages/artifacts" \
    \( -path '*ios-arm64/*' -o -path '*ios-arm64_arm64e/*' \) \
    \( -name '*.framework' -o -name '*.dylib' \) -print0 2>/dev/null || true)
fi

while IFS= read -r -d '' dylib; do
  dest="${FRAMEWORKS}/$(basename "${dylib}")"
  if [[ -e "${dest}" ]]; then
    continue
  fi
  echo "Embedding $(basename "${dylib}")"
  cp "${dylib}" "${dest}"
done < <(find "${DERIVED}/Build/Products/Release-iphoneos" -maxdepth 3 -name '*.dylib' -print0 2>/dev/null || true)

chmod +x "${APP}/Volocal" || true

# iOS 26 dyld will not map a Mach-O with no signature. Ad-hoc sign so SideStore
# can import; SideStore re-signs with the user's cert. GetMoreRAM injects
# increased-memory-limit at that step — do not copy the entitlements plist
# into the .app (it is not a real entitlement blob).
sign_adhoc() {
  local path="$1"
  codesign --force --sign - --timestamp=none --generate-entitlement-der "${path}" 2>/dev/null \
    || codesign --force --sign - --timestamp=none "${path}"
}

if [[ -d "${FRAMEWORKS}" ]]; then
  find "${FRAMEWORKS}" -name '*.dylib' -print0 2>/dev/null | while IFS= read -r -d '' lib; do
    sign_adhoc "${lib}"
  done
  find "${FRAMEWORKS}" -name '*.framework' -print0 2>/dev/null | while IFS= read -r -d '' fw; do
    sign_adhoc "${fw}"
  done
fi
sign_adhoc "${APP}"

echo "=== Volocal.app ==="
ls -la "${APP}"
echo "=== Frameworks ==="
ls -la "${FRAMEWORKS}" || echo "(none)"
echo "=== file ==="
file "${APP}/Volocal"
echo "=== otool -L ==="
otool -L "${APP}/Volocal"
echo "=== codesign ==="
codesign -dv "${APP}" 2>&1 || true

missing=0
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  if [[ -e "${FRAMEWORKS}/${rel}" ]]; then
    continue
  fi
  top="${rel%%/*}"
  if [[ -e "${FRAMEWORKS}/${top}" ]]; then
    continue
  fi
  echo "MISSING @rpath/${rel}" >&2
  missing=1
done < <(otool -L "${APP}/Volocal" | sed -n 's/.*@rpath\/\([^ ]*\).*/\1/p')
if [[ "${missing}" -ne 0 ]]; then
  echo "IPA is missing linked libraries; refusing to publish." >&2
  exit 1
fi

PAYLOAD="${ROOT}/build/Payload"
rm -rf "${PAYLOAD}"
mkdir -p "${PAYLOAD}"
cp -R "${APP}" "${PAYLOAD}/Volocal.app"
rm -f "${PAYLOAD}/Volocal.app/Volocal.entitlements" || true

(
  cd "${ROOT}/build"
  rm -f Volocal.ipa
  ditto -c -k --keepParent Payload Volocal.ipa
)

echo "IPA: ${ROOT}/build/Volocal.ipa"
ls -lh "${ROOT}/build/Volocal.ipa"
