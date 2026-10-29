#!/usr/bin/env bash
# Upload the DEEPX runtimes and the conversion toolchain to the OAAX bucket.
#
#   scripts/publish-artifacts.sh release <version> <artifacts-dir>   # on a v* tag
#   scripts/publish-artifacts.sh branch  <branch>  <artifacts-dir>   # on a pull request, for testing
#   scripts/publish-artifacts.sh dry     <name>    <artifacts-dir>   # only check the files and print the keys
#
# <artifacts-dir> holds the CI artifacts, one directory per artifact name (what actions/download-artifact creates):
#   runtime-<arch>-ubuntu<version>/library-dxrt-v<dxrt version>.tar.gz   (6: x86_64 and aarch64 x Ubuntu 20.04, 22.04, 24.04)
#   toolchain-image/oaax-deepx-toolchain.tar
#   runtime-x86_64-windows/library-dxrt-*.tar.gz                          (optional: the Windows CI job is disabled)
# S3_ACCESS_KEY, S3_SECRET_KEY and S3_ENDPOINT_URL must be set for release and branch.
#
# release: keys under <version>/ are written once (if any exists nothing is uploaded: a published version is
#          immutable, oaax-manager pins its sha256); latest/ follows the newest release.
# branch:  keys under <branch>/ are overwritten on every run, latest/ is never touched, and a name that could be a
#          release (latest, or starting with a digit or v<digit>) is refused so a branch can never replace one.
set -euo pipefail

mode=${1:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}
name=${2:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}
dir=${3:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}

case $mode in
  release | dry) ;;
  branch)
    if [[ $name =~ ^(latest|v?[0-9]) ]]; then
      echo "branch name '$name' looks like a release, rename the branch to upload its artifacts" >&2
      exit 1
    fi
    ;;
  *) echo "unknown mode: $mode" >&2; exit 1 ;;
esac

# The single archive of an artifact directory; its file name (library-dxrt-v<dxrt version>.tar.gz) is part of the key.
single_file() { # <dir> <glob>
  local found=()
  shopt -s nullglob
  # shellcheck disable=SC2206  # the glob must expand
  found=("$1"/$2)
  shopt -u nullglob
  if [ "${#found[@]}" -ne 1 ] || [ ! -s "${found[0]}" ]; then
    echo "expected exactly one non-empty $2 in $1" >&2
    return 1
  fi
  echo "${found[0]}"
}

# local file -> key, "@" stands for the version / branch / latest
declare -A files=()
for arch in x86_64 aarch64; do
  for ubuntu in 20.04 22.04 24.04; do
    f=$(single_file "$dir/runtime-$arch-ubuntu$ubuntu" 'library-*.tar.gz')
    files[$f]="runtimes/@/DEEPX/$arch/Ubuntu/$ubuntu/$(basename "$f")"
  done
done
f=$(single_file "$dir/toolchain-image" 'oaax-deepx-toolchain.tar')
files[$f]="conversion-toolchain/@/DEEPX/oaax-deepx-toolchain.tar"
# Windows is optional: its CI job is disabled until dx_rt_windows is reachable, so today there is no Windows key.
if [ -d "$dir/runtime-x86_64-windows" ]; then
  f=$(single_file "$dir/runtime-x86_64-windows" 'library-*.tar.gz')
  files[$f]="runtimes/@/DEEPX/x86_64/Windows/$(basename "$f")"
fi

if [ "$mode" = dry ]; then
  for f in "${!files[@]}"; do
    echo "would upload $f -> s3://oaax/${files[$f]//@/$name}"
  done
  exit 0
fi

s3() {
  s3cmd --access_key="$S3_ACCESS_KEY" --secret_key="$S3_SECRET_KEY" \
    --host="$S3_ENDPOINT_URL" --host-bucket="%(bucket)s.$S3_ENDPOINT_URL" --bucket-location=us-east-1 --ssl "$@"
}

if [ "$mode" = release ]; then
  # s3cmd info: 0 = exists, 12 = not found, anything else = a real error (credentials, network)
  for f in "${!files[@]}"; do
    url="s3://oaax/${files[$f]//@/$name}"
    rc=0
    s3 info "$url" >/dev/null 2>&1 || rc=$?
    case $rc in
      0) echo "$url already exists: versions are immutable, bump VERSION for a new release" >&2; exit 1 ;;
      12) ;;
      *) echo "could not check $url (s3cmd exit $rc)" >&2; exit 1 ;;
    esac
  done
fi

for f in "${!files[@]}"; do
  s3 put "$f" "s3://oaax/${files[$f]//@/$name}"
  if [ "$mode" = release ]; then s3 put "$f" "s3://oaax/${files[$f]//@/latest}"; fi
done
