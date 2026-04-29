# Lean + Mathlib Verifier Configuration

This project supports two Lean verification backends:

- `kimina` (recommended): remote/local `kimina-lean-server` for high-concurrency proof checking.
- `lean_interact`: local `lean-interact` server as fallback.

The implementation is in `src/prover_finetune/experiments/lean_checker.py`.

## Recommended: Kimina Backend

Set `lean.project_config_path` to a yaml file (for example `configs/lean_project.example.yaml`) and use:

```yaml
lean:
  project_config_path: configs/lean_project.example.yaml
```

Example verifier config:

```yaml
mathlib_path: external/mathlib4-v4.27.0
timeout_sec: 120
verifier_backend: kimina

kimina_api_url: http://localhost:8000
kimina_api_key: null
kimina_http_timeout: 600
kimina_reuse: true
kimina_debug: false

use_lean_interact: true
```

### Kimina setup

Follow the official repository: [project-numina/kimina-lean-server](https://github.com/project-numina/kimina-lean-server).

Typical local startup:

```bash
docker run -d \
  --name kimina-server \
  --restart unless-stopped \
  -p 8000:8000 \
  projectnumina/kimina-lean-server:2.0.0
```

Then verify server health via client or API before running experiments.

## Fallback: lean-interact Backend

If Kimina is unavailable, set:

```yaml
verifier_backend: lean_interact
```

Or:

```yaml
verifier_backend: auto
```

`auto` first tries Kimina and falls back to `lean-interact` when Kimina client is unavailable.

## Notes

- `mathlib_path` should point to a prepared mathlib4 directory.
- Keep Lean / mathlib versions consistent between your local project and the Kimina server image.
- `kimina_reuse: true` usually improves throughput by reusing warm REPL state.
