#!/bin/bash
set -e

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake ../..
make -j4
make install
cd -
