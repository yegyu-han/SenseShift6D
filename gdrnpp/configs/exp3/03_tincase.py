"""
This code is a modified version of the original implementation from:
https://github.com/shanice-l/gdrnpp_bop2022
"""

# 실험1) PBR + lv3 AE (general setting)
# 실험2) PBR + lv3 AE + aug (aug_prob = 0.8)
# 실험3) PBR + ALL (foundation setting)


_base_ = ["../../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/SS6D/ICLR_inference/tincase_new"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    CHANGE_BG_PROB=0.0,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        # "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
        # cosy+aae
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=32,
    TOTAL_EPOCHS=120,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,

)
LEVELS = [ "lv1", "lv2", "lv3", "lv4", "lv5" ]

TE_RGBs = [
    "AE",
    "E9G16", "E9G48", "E9G80", "E9G112",
    "E39G16", "E39G48", "E39G80", "E39G112",
    "E156G16", "E156G48", "E156G80", "E156G112",
    "E625G16", "E625G48", "E625G80", "E625G112",
    "E2500G16", "E2500G48", "E2500G80", "E2500G112"
    ]

TR_RGBs_AE = [ "AE", ]

TR_RGBs_SC = [
            "E2G0", "E2G32", "E2G64", "E2G96", "E2G128",
            "E4G0", "E4G32", "E4G64", "E4G96", "E4G128",
            "E19G0", "E19G32", "E19G64", "E19G96", "E19G128",
            "E78G0", "E78G32", "E78G64", "E78G96", "E78G128",
            "E312G0", "E312G32", "E312G64", "E312G96", "E312G128",
            "E1250G0", "E1250G32", "E1250G64", "E1250G96", "E1250G128",
            "E5000G0", "E5000G32", "E5000G64", "E5000G96", "E5000G128",
            "E10000G0", "E10000G32", "E10000G64", "E10000G96", "E10000G128"
            ]

TR_RGBs = TR_RGBs_AE + TR_RGBs_SC

DEPTHs = ["0"]
TEST = tuple(f"ss6d_tincase_{lv.lower()}{rgb.lower()}_d{depth}_te" for lv in LEVELS for rgb in TE_RGBs for depth in DEPTHs)

DATASETS = dict(
    TRAIN=("ss6d_tincase_train_pbr",),
    TRAIN2 = tuple(f"ss6d_tincase_{lv}{rgb.lower()}_tr" for lv in LEVELS for rgb in TR_RGBs),
    TRAIN2_RATIO=0.71930,
    
    TEST = TEST,
    # AP        AP50    AP75    AR      inf.time
    DET_FILES_TEST=tuple(
        "datasets/BOP_DATASETS/SenseShift6D/test/test_bboxes/scene_gt_info_bboxes.json"
        for _ in TEST),
)

DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.3,
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",  # VISIB or AMODAL
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        XYZ_ONLINE=True,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(3,),
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownDoubleMaskXyzRegionHead",
                in_dim=1024,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            FULL_MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=True,  # NOTE: sym loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
    ),
)
"""
VAL = dict(
    DATASET_NAME="ss6d",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets.json",
    ERROR_TYPES="mspd,mssd,vsd,ad,reS,teS",
    # ERROR_TYPES="ad, rete,",
    RENDERER_TYPE="cpp",  # cpp, python, egl
    SPLIT="test",
    SPLIT_TYPE="",
    N_TOP=1,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
    EVAL_CACHED=False,  # if the predicted poses have been saved
    SCORE_ONLY=False,  # if the errors have been calculated
    EVAL_PRINT_ONLY=False,  # if the scores/recalls have been saved
    EVAL_PRECISION=False,  # use precision or recall
    USE_BOP=True,  # whether to use bop toolkit
)
"""
TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
