"""Verify and stop the specified pod from outside its disposable container.

The pod-injected credential travels over the existing SSH connection, remains
only in this process's memory, and is never written to output or an artifact.
Use --execute only after backing up the experiment outputs.
"""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--expected-id", required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # Capture this output privately. Never include stdout/stderr in exceptions:
    # stdout contains the credential on a successful SSH read.
    remote = """python - <<'PY'
import json
from pathlib import Path
env = dict(x.decode().split('=',1) for x in Path('/proc/1/environ').read_bytes().split(b'\\0') if b'=' in x)
print(json.dumps({'id': env['RUNPOD_POD_ID'], 'key': env['RUNPOD_API_KEY']}))
PY"""
    proc = subprocess.run(["ssh", "-i", args.identity, "-p", str(args.port), "-o", "BatchMode=yes",
                           "-o", "ConnectTimeout=15", args.host, remote],
                          capture_output=True, text=True, timeout=45)
    if proc.returncode:
        raise SystemExit("Could not read the pod's management credential over SSH")
    credential = json.loads(proc.stdout)
    if credential["id"] != args.expected_id:
        raise SystemExit("Remote container identity does not match the expected pod")

    def request(query):
        req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + credential["key"]})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"Runpod management request returned HTTP {exc.code}") from None
        if result.get("errors"):
            raise SystemExit(json.dumps({"errors": [e.get("message") for e in result["errors"]]}))
        return result["data"]

    query = 'query { pod(input: {podId: ' + json.dumps(args.expected_id) + '}) { id name desiredStatus costPerHr runtime { uptimeInSeconds } } }'
    pod = request(query)["pod"]
    if not pod or pod["id"] != args.expected_id or pod["name"] != args.expected_name:
        raise SystemExit("Pod identity verification failed")
    print(json.dumps({"verified_current_pod": pod}), flush=True)
    if args.execute:
        result = request('mutation { podStop(input: {podId: ' + json.dumps(args.expected_id) + '}) { id desiredStatus } }')
        print(json.dumps(result), flush=True)
        if result["podStop"]["desiredStatus"] != "EXITED":
            raise SystemExit("Stop response did not confirm EXITED")
        for _ in range(40):
            final = request(query)["pod"]
            if final["desiredStatus"] == "EXITED" and final.get("runtime") is None:
                print(json.dumps({"stopped_pod": final, "gpu_runtime_released": True}), flush=True)
                return
            time.sleep(3)
        raise SystemExit("Stop was requested but GPU runtime release was not confirmed")


if __name__ == "__main__":
    main()
