#!/usr/bin/env python3
"""Print hybrid MoE scale plans for 2T, 20T, 200T (and optional custom).

  python examples/16_scale_plan.py
  python examples/16_scale_plan.py --total 20e12 --experts 128 --top-k 2
  python examples/16_scale_plan.py --optical-speedup 10 --tflops 500
"""

from __future__ import annotations

import argparse
import json

from atom.capacity import CapacityParams
from atom.scale_plan import (
    ThroughputAssumptions,
    compare_ladder,
    format_plan,
    plan_ladder,
    plan_scale,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="ATOM scale training planner")
    ap.add_argument("--total", type=float, default=None, help="single total param target")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--token-mult", type=float, default=20.0)
    ap.add_argument("--m-number", type=float, default=2.0)
    ap.add_argument("--tflops", type=float, default=100.0, help="digital active TFLOP/s assumption")
    ap.add_argument("--optical-speedup", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cap = CapacityParams(m_number=args.m_number)
    thr = ThroughputAssumptions(
        digital_active_tflops=args.tflops,
        optical_speedup_on_scores=args.optical_speedup,
    )
    kw = dict(
        n_experts=args.experts,
        top_k=args.top_k,
        token_multiplier=args.token_mult,
        capacity=cap,
        thr=thr,
    )

    if args.total is not None:
        plans = [plan_scale(args.total, **kw)]
    else:
        plans = plan_ladder(**kw)

    if args.json:
        print(json.dumps([p.to_dict() for p in plans], indent=2))
        return

    print(compare_ladder(plans))
    print()
    for p in plans:
        print(format_plan(p))
        print()


if __name__ == "__main__":
    main()
