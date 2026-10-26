#!/usr/bin/env bash
# Rebuild rocBLAS with gfx906 (Vega 20: Instinct MI50 / MI60, Radeon VII) in its GPU_TARGETS, into a user-local prefix.
#
# ROCm 7.x / 10.x no longer builds gfx906's rocBLAS kernels (Tensile library + HIP kernels), so hipBLAS aborts there with
# "rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906". The rocBLAS source still has gfx906 (the
# vega20 logic files); this builds it from the rocm-libraries tag of the installed ROCm. Nothing under ROCM_PATH changes:
# the rebuilt library goes to $PREFIX and is loaded with LD_PRELOAD (ROCm's libhipblas has an RPATH to its own folder,
# which LD_LIBRARY_PATH cannot override; hipBLAS, the engine and everything else keep using the system ROCm; docs/AMD_MAYA.md, "gfx906 (Vega 20)").
#
# The rebuilt rocBLAS also carries every other architecture in OTHER_TARGETS (default: the other AMD GPUs the machine has),
# because it replaces the system one for the whole process: a gfx906 + gfx1200 PC needs both in it.
#
#   tools/hip/build_rocblas_gfx906.sh                      # gfx906 + the machine's other GPUs
#   OTHER_TARGETS="gfx1200;gfx1100" tools/hip/build_rocblas_gfx906.sh
#   TAG=therock-10.1 PREFIX=$HOME/local/rocm-gfx906 JOBS=10 tools/hip/build_rocblas_gfx906.sh
#
# Needs: git, cmake 3.24+, python3 with venv, curl, ~15 GB of disk, network access. Takes 1-3 hours (Tensile generating
# kernels for each architecture; gfx906 alone is the shortest).
set -euo pipefail

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PREFIX=${PREFIX:-$HOME/local/rocm-gfx906}
WORK=${WORK:-$HOME/gfx906-work}
JOBS=${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
BOOST_VERSION=${BOOST_VERSION:-1.86.0}
MSGPACK_TAG=${MSGPACK_TAG:-cpp-6.1.1}

# the rocm-libraries tag of the installed ROCm (10.1.0 -> therock-10.1); TAG= overrides
if [ -z "${TAG:-}" ]; then
  ver=$(cat "$ROCM_PATH"/.info/version 2>/dev/null || true)
  TAG="therock-$(echo "$ver" | cut -d. -f1,2)"
fi
echo "ROCm: $ROCM_PATH   rocm-libraries tag: $TAG   prefix: $PREFIX   work: $WORK"

if [ -z "${OTHER_TARGETS:-}" ]; then
  OTHER_TARGETS=$("$ROCM_PATH"/bin/rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | grep -vE '^gfx(000|906)$' | sort -u | tr '\n' ';' | sed 's/;$//' || true)
fi
TARGETS="gfx906:xnack-${OTHER_TARGETS:+;$OTHER_TARGETS}"
echo "rocBLAS GPU_TARGETS: $TARGETS"

mkdir -p "$WORK"
cd "$WORK"

[ -d venv ] || python3 -m venv venv
# shellcheck disable=SC1091
. venv/bin/activate
pip -q install ninja pyyaml msgpack joblib packaging six wheel virtualenv psutil

# rocBLAS source at the installed ROCm's tag (sparse: only rocBLAS and the shared Tensile)
if [ ! -d rl ]; then
  git clone -q --depth 1 --branch "$TAG" --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git rl
  git -C rl sparse-checkout set projects/rocblas shared/tensile
fi

# Tensile's host library reads its kernel index with msgpack-cxx (header-only), which pulls in Boost headers
if [ ! -f deps/lib/cmake/msgpack-cxx/msgpack-cxx-config.cmake ]; then
  [ -d msgpack-c ] || git clone -q --depth 1 --branch "$MSGPACK_TAG" https://github.com/msgpack/msgpack-c.git msgpack-c
  cmake -S msgpack-c -B msgpack-build -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DMSGPACK_USE_BOOST=OFF \
        -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$WORK/deps" >/dev/null
  cmake --build msgpack-build >/dev/null
  cmake --install msgpack-build >/dev/null
fi
if [ ! -d deps/include/boost/assert ]; then
  b=boost_${BOOST_VERSION//./_}
  curl -sSL -o boost.tar.gz "https://archives.boost.io/release/$BOOST_VERSION/source/$b.tar.gz"
  tar -xzf boost.tar.gz --strip-components=1 -C deps/include "$b/boost"
  rm boost.tar.gz
fi

export PATH="$ROCM_PATH/bin:$PATH"
cmake -S rl/projects/rocblas -B build-rocblas -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$ROCM_PATH/bin/amdclang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/amdclang++" \
      -DCMAKE_PREFIX_PATH="$ROCM_PATH;$WORK/deps" -DROCM_PATH="$ROCM_PATH" -DGPU_TARGETS="$TARGETS" \
      -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build build-rocblas -j"$JOBS"
cmake --install build-rocblas

lib=$(ls -d "$PREFIX"/lib*/ 2>/dev/null | head -1)
echo
echo "Done. Load the rebuilt rocBLAS ahead of the system one (it finds its own Tensile library next to itself):"
echo "  export LD_PRELOAD=${lib%/}/librocblas.so.5"
echo "(Maya: ./maya.sh --env LD_PRELOAD=${lib%/}/librocblas.so.5, or export it before ./maya.sh)"
