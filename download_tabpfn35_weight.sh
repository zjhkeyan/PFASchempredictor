#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${package_root}/models"
filename="tabpfn-v3.5-20260909.safetensors"
expected_sha256="ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"

mkdir -p "${model_dir}"
python - "${model_dir}" "${filename}" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

destination = Path(sys.argv[1])
filename = sys.argv[2]
cached = Path(
    hf_hub_download(
        repo_id="Prior-Labs/tabpfn_3_5",
        filename=filename,
    )
)
target = destination / filename
if not target.exists() or target.stat().st_size != cached.stat().st_size:
    shutil.copyfile(cached, target)
print(target)
PY

actual_sha256="$(sha256sum "${model_dir}/${filename}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    printf 'SHA256 mismatch: expected %s, got %s\n' "${expected_sha256}" "${actual_sha256}" >&2
    exit 1
fi
printf 'Verified TabPFN 3.5 checkpoint: %s\n' "${model_dir}/${filename}"
