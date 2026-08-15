"""CLI: python -m atom.claim_audit"""

from __future__ import annotations

import argparse
import json

from atom.claims import build_claims, format_audit
from atom.calibrate import synthetic_calibration_demo
from atom.physical_score import PhysicalScoreParams, predict_score_observables, AcceptanceGate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ATOM physical claim audit")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--calibrate-demo", action="store_true", help="run synthetic inverse calibration")
    ap.add_argument("--phase-noise", type=float, default=0.05)
    ap.add_argument("--crosstalk", type=float, default=0.05)
    args = ap.parse_args(argv)

    params = PhysicalScoreParams(
        phase_noise_sigma=args.phase_noise,
        amplitude_error_sigma=0.03,
        angular_mix=args.crosstalk,
        detector_noise_sigma=0.02,
        adc_bits=8,
    )
    claims = build_claims(score_params=params)

    if args.json:
        payload = {
            "claims": [c.to_dict() for c in claims],
            "score_prediction": predict_score_observables(params).to_dict(),
        }
        if args.calibrate_demo:
            payload["calibration_demo"] = synthetic_calibration_demo().to_dict()
        print(json.dumps(payload, indent=2))
        return

    print(format_audit(claims))
    pred = predict_score_observables(params)
    gate = AcceptanceGate()
    print("Score twin prediction (Monte Carlo mean):")
    print(f"  RMS={pred.metrics.rms_error:.4f} cosine={pred.metrics.cosine:.4f} "
          f"top1={pred.metrics.top1_agreement:.3f} SNR_dB={pred.metrics.logit_snr_db:.2f}")
    print(f"  gate_pass={gate.passes(pred.metrics)} details={gate.evaluate(pred.metrics)}")
    if args.calibrate_demo:
        cal = synthetic_calibration_demo()
        print("\nSynthetic inverse calibration:")
        print(f"  loss={cal.loss:.6f} params={cal.params.to_dict()}")


if __name__ == "__main__":
    main()
