# TODO — pending before submission (deadline 2026-11-17)

## Reproducibility

- [ ] Publish the champion weights as GitHub release assets (113 MB each — over git's
      100 MB limit) and put the release URL in `REPRODUCING.md`:
      - `weights/moncuniserver/ball_yolo26x_1024_clu.pt`
        `f69ff921f8ae32eeba75aa6dc8730eefe3b4389232e0f8b0948482ea7f412053`
      - `weights/players_weights/players_yolo26x_1024_clu.pt`
        `03031012298e134eb13f4dd03c25651cc3ffc231c27f93f16fb38c4b2de04865`
- [ ] Fill the training-provenance TODO in `REPRODUCING.md`: external dataset name(s), source
      link + license, training config/log for both YOLO26x detectors.

## Rules & organisers

- [ ] Send the clarification email to the organisers (three boundary questions: scoring the
      public window as a correctness check only; inspecting the video to understand possession;
      what counts as "uploading provided data to AI services").
- [ ] Working rule from now on: no parameter choice justified by the public `spellF1` or by
      frames inspected against the video — only a-priori physical/semantic justifications and
      automatic checks on provided data. Document each threshold with its provenance.

## Documentation

- [ ] Rewrite `README.md` and `possession_lab/TECHNICAL.md` to describe the real champion
      chain (perception → fusion poll → transition layer with dead-ball locks, tag revival,
      possession revival → CSV), not the legacy tracker-follow pipeline; include limitations.

## Submission

- [ ] Produce `<team_name>.csv` from `possession_lab/outputs/transition_predictions.csv` and
      run `python evaluation/validate_submission.py <team_name>.csv` on the **exact file to be
      uploaded**.
- [ ] Re-check the repo regenerates the identical CSV (`REPRODUCING.md` checksums) before
      uploading via the Google form (team name exactly as registered).
