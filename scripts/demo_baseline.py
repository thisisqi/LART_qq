import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from phalp.configs.base import CACHE_DIR, FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
import gc
import torch

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

# create the tracker with hmr2 backend
class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)

class RenderOnly(HMR2_4dhuman):
    def setup_predictor(self):
        # 已经有 pkl，不重新计算 LART/SlowFast
        self.pose_predictor = None

    def setup_detectron2(self):
        # 已经有 pkl，不重新检测人体
        self.detector = None
        self.detector_x = None

# create the tracker with action predictor
class LART(HMR2_4dhuman):
    def __init__(self, cfg):

        download_files = {
            "lart_mvit.config" : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.config", os.path.join(CACHE_DIR, "phalp/ava")],
            "lart_mvit.ckpt"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.ckpt", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.yaml"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.yaml", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.pyth"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.pyth", os.path.join(CACHE_DIR, "phalp/ava")],
        }
        self.cached_download_from_drive(download_files)
        super().__init__(cfg)

    def setup_predictor(self):
        # setup predictor model witch predicts actions from poses
        log.info("Loading Predictor model...")
        from lart.utils.wrapper_phalp import Pose_transformer
        self.pose_predictor = Pose_transformer(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)





@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    # True：使用已有 pkl
    # False：从视频重新开始计算
    USE_EXISTING_PKL = False

    # True：只绘制人体模型，不重新计算 LART/SlowFast
    # False：会运行 LART，可能重新计算 SlowFast 特征
    RENDER_ONLY = False
    TRACK_ONLY = False

    cfg.phalp.low_th_c = 0.5
    cfg.phalp.small_w = 30
    cfg.phalp.small_h = 60

    cfg.phalp.start_frame = -1
    cfg.phalp.end_frame = -1

    cfg.overwrite = True
    cfg.render.fps = 15

    if USE_EXISTING_PKL:
        # 使用已经计算好的 pkl
        pkl_path = (
            "/home/lsy/Code/LART/outputs/"
            "1019_test_qq_smallobj_v2/results/"
            "demo_1019_test_demo_15fps.pkl"
        )
    else:
        # 从视频重新进行人体跟踪
        cfg.render.enable = False

        phalp_tracker = HMR2_4dhuman(cfg)
        _, pkl_path = phalp_tracker.track()
        del phalp_tracker

        if TRACK_ONLY:
            print(f"Tracking finished. pkl saved at: {pkl_path}")
            return

    cfg.render.enable = True
    cfg.render.colors = "slahmr"
    cfg.render.type = "HUMAN_MESH"

    if RENDER_ONLY:
        # 只读取已有 pkl 并绘制人体模型
        renderer = RenderOnly(cfg)
        renderer.setup_postprocessor()
        renderer.postprocessor.run_renderer(pkl_path)
    else:
        # 运行 LART，可能重新计算 SlowFast 特征
        cfg.pose_predictor.config_path = (
            f"{CACHE_DIR}/phalp/ava/lart_mvit.config"
        )
        cfg.pose_predictor.weights_path = (
            f"{CACHE_DIR}/phalp/ava/lart_mvit.ckpt"
        )
        cfg.post_process.save_fast_tracks = True

        lart_model = LART(cfg)
        lart_model.setup_postprocessor()
        lart_model.postprocessor.run_lart(pkl_path)

if __name__ == "__main__":
    main()
