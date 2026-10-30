GPU FARM PERCEPTION — QUICK SHEET
=================================

Bundle contents
---------------
  farm_perceive.py       two-stage inference (ball first, players second, then merge)
  farm_perceive.sbatch   SLURM wrapper for one GPU job
  FARM_README.txt        this file
  possession_lab/        only config.py and perception.py are needed

Create the transfer bundle from the repository root:

  mkdir -p dist
  tar czf dist/mtah_farm_perceive.tar.gz \
      farm_perceive.py farm_perceive.sbatch FARM_README.txt \
      possession_lab/config.py possession_lab/perception.py

Paths expected by the script
----------------------------
The homography and ATD sidecars must have the same stem as the tactical video.
For example, for `BAR-ATH_tactical-001.mp4` the script expects:

  BAR-ATH_tactical-001_homography.csv
  BAR-ATH_tactical-001_atd.txt
  BAR-ATH_tactical-001_atd.xml

The default paths below are the ones used on the Merit GPU setup. Override them
with command-line arguments or environment variables if yours are different:

  videos:   /home/alumnes/joan.ignasi.cid/dataset/tactical
  ball:     ~/ball_data/weights/ball_yolo26x_1024_clu.pt
  players:  ~/ball_data/player_data/weights/players_yolo26x_1024_clu.pt
  output:   ~/perception_runs

1. Upload from the laptop
-------------------------
Replace `GPU_USER@GPU_HOST` with your own SSH target.

  scp dist/mtah_farm_perceive.tar.gz GPU_USER@GPU_HOST:~/

2. Unpack and inspect on the GPU server
---------------------------------------
  ssh GPU_USER@GPU_HOST
  mkdir -p ~/mtah_farm
  tar xzf ~/mtah_farm_perceive.tar.gz -C ~/mtah_farm
  cd ~/mtah_farm
  ~/ball_data/.venv/bin/python farm_perceive.py --list

If `--list` reports a missing video, homography or ATD, copy the file with the
correct stem into the dataset directory.

3. Smoke test
-------------
Run this on the GPU server before starting the full job:

  ~/ball_data/.venv/bin/python farm_perceive.py \
      --start 11340 --stop 11420 --step 1 --imgsz 1024 \
      --output-dir ~/perception_smoke

4a. Full run through SLURM
--------------------------
Check the GPU partition first:

  sinfo -o "%P %G"

If your partition is not called `gpu`, edit the `#SBATCH --partition=` line in
`farm_perceive.sbatch`. Then submit:

  sbatch farm_perceive.sbatch
  tail -f farm_perceive-*.log

To process every match in the dataset:

  MATCH=all sbatch farm_perceive.sbatch

To process every second frame:

  STEP=2 sbatch farm_perceive.sbatch

4b. Full run directly on a GPU server
-------------------------------------
If the server is not managed through SLURM, run the script directly inside
`tmux` or `nohup` so it survives the SSH session ending:

  mkdir -p ~/perception_runs
  cd ~/mtah_farm

  nohup ~/ball_data/.venv/bin/python -u farm_perceive.py \
      --dataset-dir /home/alumnes/joan.ignasi.cid/dataset/tactical \
      --ball-weights ~/ball_data/weights/ball_yolo26x_1024_clu.pt \
      --players-weights ~/ball_data/player_data/weights/players_yolo26x_1024_clu.pt \
      --output-dir ~/perception_runs \
      --imgsz 1024 --step 1 \
      > ~/perception_runs/run.log 2>&1 &

Monitor it with:

  tail -f ~/perception_runs/run.log
  find ~/perception_runs -maxdepth 1 -type f -name 'BAR-ATH*'

5. Download the finished cache
------------------------------
The final cache is named after the video stem. For the example above:

  scp GPU_USER@GPU_HOST:~/perception_runs/BAR-ATH_tactical-001_perception.npz \
      possession_lab/outputs/perception_new.npz
  scp GPU_USER@GPU_HOST:~/perception_runs/BAR-ATH_tactical-001_perception.npz.json \
      possession_lab/outputs/perception_new.npz.json

The intermediate `_ball.npz` and `_players.npz` files are useful for resuming a
farm run, but the final `_perception.npz` is the file consumed by fusion.

6. Build fusion and open the dashboard
--------------------------------------
From the repository root on the laptop:

  .venv/bin/python -m possession_lab fusion \
      --perception possession_lab/outputs/perception_new.npz \
      --fusion possession_lab/outputs/fusion_new.npz

  .venv/bin/python -m possession_lab viewer \
      --perception possession_lab/outputs/perception_new.npz \
      --fusion possession_lab/outputs/fusion_new.npz \
      --port 8765 --no-open

Open `http://127.0.0.1:8765/` and jump to frame 11344.

Notes
-----
* The default step is 1, so the cache covers every tactical frame (199,319).
* The ball and player models are never loaded at the same time, so VRAM stays low.
* `--resume` is enabled by default. It reuses `<match>_ball.npz`, `<match>_players.npz`
  and the final cache when they cover the requested frame range.
* The player model has player, goalkeeper and referee classes. All three are treated
  as people, matching the existing viewer and fusion contract.
