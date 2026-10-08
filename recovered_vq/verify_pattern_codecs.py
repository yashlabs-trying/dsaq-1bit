"""Fresh-process decoder verification for saved checkpoint-pattern codecs."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")
import waverel_checkpoint_patterns as experiment


root = Path("/workspace/waverel_patterns")
report = {}
for path in sorted(root.glob("*.bin")):
    if path.name == "strict_fisher_vq_layer0.bin":
        continue  # This comparison payload uses the recovered VQ decoder.
    weights, metadata = experiment.decode(path)
    digest = hashlib.sha256()
    for name in metadata["names"]:
        digest.update(weights[name].contiguous().numpy().tobytes())
    report[path.name] = {
        "bytes": path.stat().st_size,
        "bpw": path.stat().st_size * 8 / metadata["weights"],
        "decoded_weight_sha256": digest.hexdigest(),
        "matrices": {name: list(weights[name].shape) for name in metadata["names"]},
    }
(root / "roundtrip_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
