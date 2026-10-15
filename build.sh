#!/usr/bin/env bash
# Gigatoken.cs build driver: cargo (native C ABI, release) + dotnet (pack).
# Usage: ./build.sh [--no-native] [--pack] [extra dotnet build args...]
#   Cross-compile: TARGETS="x86_64-unknown-linux-gnu aarch64-apple-darwin" ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

DO_NATIVE=1
DO_PACK=0
DOTNET_ARGS=()
for a in "$@"; do
  case "$a" in
    --no-native) DO_NATIVE=0 ;;
    --pack) DO_PACK=1 ;;
    *) DOTNET_ARGS+=("$a") ;;
  esac
done

triple_to_rid() {
  case "$1" in
    x86_64-unknown-linux-gnu) echo "linux-x64" ;;
    aarch64-unknown-linux-gnu) echo "linux-arm64" ;;
    x86_64-apple-darwin) echo "osx-x64" ;;
    aarch64-apple-darwin) echo "osx-arm64" ;;
    x86_64-pc-windows-msvc) echo "win-x64" ;;
    aarch64-pc-windows-msvc) echo "win-arm64" ;;
    *) echo "unknown-$1" ;;
  esac
}

host_rid() {
  local os arch
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os-$arch" in
    Linux-x86_64) echo "linux-x64" ;;
    Linux-aarch64) echo "linux-arm64" ;;
    Darwin-x86_64) echo "osx-x64" ;;
    Darwin-arm64) echo "osx-arm64" ;;
    MINGW*-x86_64|MSYS*-x86_64|CYGWIN*-x86_64) echo "win-x64" ;;
    MINGW*-aarch64|MSYS*-aarch64|CYGWIN*-aarch64) echo "win-arm64" ;;
    Windows_NT-x86_64|Windows_NT-AMD64) echo "win-x64" ;;
    Windows_NT-ARM64|Windows_NT-aarch64) echo "win-arm64" ;;
    *) echo "unknown-${os}-${arch}" ;;
  esac
}

if [ "$DO_NATIVE" -eq 1 ]; then
  TARGETS="${TARGETS:-host}"
  if [ "$TARGETS" = "host" ]; then
    echo "==> cargo build --release (host)"
    cargo build --release
    rid="$(host_rid)"
    case "$rid" in
      linux-*) lib="libgigatoken_cabi.so" ;;
      osx-*) lib="libgigatoken_cabi.dylib" ;;
      win-*) lib="gigatoken_cabi.dll" ;;
      *) echo "error: unsupported host RID '$rid'; cross-compile with TARGETS=\"...\" ./build.sh" >&2; exit 1 ;;
    esac
    dest="src/Gigatoken/runtimes/$rid/native"
    mkdir -p "$dest"
    cp -f "target/release/$lib" "$dest/"
    echo "==> staged $dest/$lib"
  else
    for t in $TARGETS; do
      echo "==> cargo build --release --target $t"
      cargo build --release --target "$t"
      rid="$(triple_to_rid "$t")"
      case "$rid" in
        linux-*) lib="libgigatoken_cabi.so" ;;
        osx-*) lib="libgigatoken_cabi.dylib" ;;
        win-*) lib="gigatoken_cabi.dll" ;;
        *) echo "error: unsupported target triple '$t'" >&2; exit 1 ;;
      esac
      dest="src/Gigatoken/runtimes/$rid/native"
      mkdir -p "$dest"
      cp -f "target/$t/release/$lib" "$dest/"
      echo "==> staged $dest/$lib"
    done
  fi
fi

if [ "$DO_PACK" -eq 1 ]; then
  echo "==> dotnet pack -c Release"
  HAS_CONFIG=0
  for a in ${DOTNET_ARGS[@]+"${DOTNET_ARGS[@]}"}; do
    case "$a" in
      -c|--configuration|/p:Configuration*) HAS_CONFIG=1 ;;
    esac
  done
  if [ "$HAS_CONFIG" -eq 1 ]; then
    dotnet pack Gigatoken.slnx "${DOTNET_ARGS[@]}"
  else
    dotnet pack Gigatoken.slnx -c Release "${DOTNET_ARGS[@]}"
  fi
else
  echo "==> dotnet build -c Release"
  HAS_CONFIG=0
  for a in ${DOTNET_ARGS[@]+"${DOTNET_ARGS[@]}"}; do
    case "$a" in
      -c|--configuration|/p:Configuration*) HAS_CONFIG=1 ;;
    esac
  done
  if [ "$HAS_CONFIG" -eq 1 ]; then
    dotnet build Gigatoken.slnx "${DOTNET_ARGS[@]}"
  else
    dotnet build Gigatoken.slnx -c Release "${DOTNET_ARGS[@]}"
  fi
fi
