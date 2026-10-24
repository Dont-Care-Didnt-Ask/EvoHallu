"""Read or stop only the current, explicitly identified Runpod pod.

The injected credential stays on the pod and is never included in output.
Use --execute only after experiment artifacts have been backed up elsewhere.
"""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-id", required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    values = dict(item.decode().split("=", 1) for item in Path("/proc/1/environ").read_bytes().split(b"\0") if b"=" in item)
    if values.get("RUNPOD_POD_ID") != args.expected_id:
        raise SystemExit("Current container does not match the expected pod")

    def request(query):
        req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + values["RUNPOD_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"Runpod management request returned HTTP {exc.code}") from None
        if result.get("errors"):
            # Only report server messages, never the authenticated request.
            raise SystemExit(json.dumps({"errors": [e.get("message") for e in result["errors"]]}))
        return result["data"]

    pod = request('query { pod(input: {podId: ' + json.dumps(args.expected_id) + '}) { id name desiredStatus costPerHr } }')["pod"]
    if not pod or pod["id"] != args.expected_id or pod["name"] != args.expected_name:
        raise SystemExit("Pod identity verification failed")
    print(json.dumps({"verified_current_pod": pod}), flush=True)
    if args.execute:
        result = request('mutation { podStop(input: {podId: ' + json.dumps(args.expected_id) + '}) { id desiredStatus } }')
        print(json.dumps(result), flush=True)
        if result["podStop"]["desiredStatus"] != "EXITED":
            raise SystemExit("Stop response did not confirm EXITED")


if __name__ == "__main__":
    main()
