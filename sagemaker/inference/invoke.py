import argparse, json, base64
import boto3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint_name", required=True)
    ap.add_argument("--audio", required=True, help="Path to .wav (16 kHz recommended)")
    args = ap.parse_args()

    # Read wav bytes and base64-encode them
    with open(args.audio, "rb") as f:
        b = f.read()
    payload = {"audio_base64": base64.b64encode(b).decode("utf-8")}

    rt = boto3.client("sagemaker-runtime")
    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
        Accept="application/json",
    )
    out = json.loads(resp["Body"].read().decode("utf-8"))
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
