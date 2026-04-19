"""Render the Kubernetes Deployment manifest for the GLM-4.1V vision pool.

Separate pool from the CPU workers — each pod owns one GPU (prefer
cheap non-A100), loads GLM-4.1V-9B-Thinking once at startup, and pulls
tasks on subject ``erebus.tasks.solve_task_vision``.

NRP policy: stay under 40% sustained GPU util per pod by keeping
concurrency at 1 per pod. Scale horizontally via more replicas.

Usage:
    python scripts/render_erebus_vision_pool.py --replicas 2 > vision.yaml
    kubectl apply -f vision.yaml
"""

from __future__ import annotations

import argparse

from nats_bursting import PoolDescriptor, pool_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="erebus-vision-workers")
    ap.add_argument("--namespace", default="ssu-atlas-ai")
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=1)
    # NRP non-A100 GPUs that fit a 9B bf16 model (~18GB):
    #   L40 48GB, L40S 48GB, V100 32GB, A10 24GB (tight), 3090 24GB (tight),
    #   H100 80GB (needs form), H200 (needs form)
    ap.add_argument("--cpu", default="2")
    ap.add_argument("--memory", default="24Gi")
    ap.add_argument("--model-id", default="THUDM/GLM-4.1V-9B-Thinking")
    ap.add_argument("--bundle-repo", default="ahb-sjsu/neurogolf-bundle")
    args = ap.parse_args()

    desc = PoolDescriptor(
        name=args.name,
        namespace=args.namespace,
        replicas=args.replicas,
        cpu=args.cpu,
        memory=args.memory,
        gpu=args.gpu,
        consumer_group="erebus-vision-workers",
        stream="EREBUS_TASKS",
        subjects=["erebus.tasks.solve_task_vision"],
        env={
            "TASK_DIR": "/work/tasks",
            "COMPILER_DIR": "/work/bundle/src/compiler",
            "PYTHONPATH": "/work/agi-hpc/src",
            "NATS_RESULT_PREFIX": "erebus.results.",
            "NATS_DURABLE": "0",
            "RESULT_WEBHOOK_URL": "https://atlas-sjsu.duckdns.org/api/erebus/result",
            "VISION_MODEL_ID": args.model_id,
            # HF cache on ephemeral storage — for real we'd mount a PVC
            "HF_HOME": "/work/hf-cache",
            "TRANSFORMERS_CACHE": "/work/hf-cache",
        },
        env_from_secrets={
            "NRP_LLM_TOKEN": ("erebus-worker-secrets", "nrp-llm-token"),
            "HF_TOKEN": ("erebus-worker-secrets", "hf-token"),
        },
        pre_install=[
            # Base deps (nats-bursting worker + Pillow for image rendering
            # + accelerate for device_map=auto).
            "pip install --quiet nats-py openai numpy pillow "
            "'accelerate>=0.33' 'transformers>=4.49' "
            "'git+https://github.com/ahb-sjsu/nats-bursting.git@main#subdirectory=python'",
            "mkdir -p /work && cd /work && "
            f"git clone --depth 1 https://github.com/{args.bundle_repo}.git bundle && "
            "git clone --depth 1 https://github.com/ahb-sjsu/agi-hpc.git && "
            "mkdir -p /work/tasks && "
            "tar -C /work/tasks -xzf bundle/data/tasks.tar.gz",
        ],
        entry=["python3", "-u", "-m", "agi.autonomous.erebus_worker_main"],
    )
    print(pool_manifest(desc))


if __name__ == "__main__":
    main()
