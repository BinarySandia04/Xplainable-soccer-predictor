# TODO — pending before submission (deadline 2026-11-17)

## Reproducibility

- [x] Publish the champion weights as GitHub release assets (113 MB each — over git's
      100 MB limit) and put the release URL in `REPRODUCING.md`:
      [`champion-run` release](https://github.com/BinarySandia04/Xplainable-soccer-predictor/releases/tag/champion-run)
      — also carries the full-match detections (`perception_new_fixed.npz`, `fusion.npz`) and
      the small-outputs bundle so the results can be viewed on any PC without the model.
- [x] Fill the training-provenance TODO in `REPRODUCING.md`: external dataset name(s), source
      link + license, training config/log for both YOLO26x detectors. (Scripts
      `ball_training/train_cluster.py` / `train_players_cluster.py` are committed; dataset
      sources + licenses are listed in the *Model weights* section.)

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
