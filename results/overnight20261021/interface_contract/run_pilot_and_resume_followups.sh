#!/usr/bin/env bash
# Explicit execution only: main verifies and stops the existing waiting followup first.
set -u
.venv/bin/python scripts/run_overnight_parallel.py --execute --phase all --manifest results/next_experiments/interface_pilot_manifest.json --state results/next_experiments/interface_pilot_state.json --ports 18000
interface_pilot_status=$?
.venv/bin/python - "$interface_pilot_status" <<'PY'
import datetime,json,sys
from pathlib import Path
Path('results/overnight20261021/interface_contract/pilot_controller_exit.json').write_text(json.dumps({
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'exit_code':int(sys.argv[1]),
    'next_action':'Restart followups; its source and rank terminal checks remain required'
},indent=2)+'\n')
PY
exec .venv/bin/python scripts/overnight_followups.py --execute --reserve-transfer-minutes 45 > results/overnight20261021/followups.log 2>&1
