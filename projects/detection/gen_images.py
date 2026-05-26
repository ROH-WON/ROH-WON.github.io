#!/usr/bin/env python3
"""README용 설명 이미지 생성 스크립트"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import shutil
import cv2
from pathlib import Path

OUT = Path("/home/addinedu/detection/docs/images")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fm.fontManager.addfont(FONT_PATH)
fm.fontManager.addfont(FONT_BOLD_PATH)
KR = fm.FontProperties(fname=FONT_PATH)
KR_B = fm.FontProperties(fname=FONT_BOLD_PATH)

DARK   = "#0d1117"
BOX_BG = "#161b22"
C1     = "#00b4d8"   # blue
C2     = "#e94560"   # red
C3     = "#2ea043"   # green
C4     = "#d29922"   # yellow
C5     = "#7c3aed"   # purple
WHITE  = "#f0f6fc"
GRAY   = "#8b949e"


def ktext(ax, x, y, s, fontsize=10, color=WHITE, bold=False, **kw):
    prop = KR_B if bold else KR
    ax.text(x, y, s, fontproperties=prop, fontsize=fontsize, color=color, **kw)


# ── Fig 1: 시스템 전체 파이프라인 ─────────────────────────────────────────────
def fig1_overview():
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    stages = [
        (0.3,  1.5, 3.0, 5.0, C1,  "데이터 수집",
         ["SAM3 자동 라벨링", '"red figure"', "텍스트 프롬프트", "YOLO 형식 저장", "761장 학습 데이터"]),
        (3.8,  1.5, 3.0, 5.0, C4,  "모델 학습",
         ["YOLOv8n 기반", "hands_up /", "hands_down 분류", "200 에폭 설정", "(조기종료 21에폭)"]),
        (7.3,  1.5, 3.0, 5.0, C3,  "데모 검증",
         ["demo.py", "카메라 실시간", "포즈 감지", "비ROS 환경", "동작 확인"]),
        (10.8, 1.5, 3.2, 5.0, C2,  "기하학 검증",
         ["Solidity 분석", "ReachRatio 분석", "AND 이중 임계값", "오탐 억제", "신뢰도 향상"]),
        (14.5, 1.5, 2.4, 5.0, C5,  "원근 변환",
         ["카메라 캘리브", "단응행렬 H", "픽셀→맵 좌표", "(미터 단위)"]),
        (17.3, 1.5, 2.4, 5.0, C2,  "ROS2 발행",
         ["PoseStamped", "/hand_raise_goal", "Nav2 연동", "로봇 안내"]),
    ]

    for x, y, w, h, color, title, lines in stages:
        rect = plt.Rectangle((x, y), w, h, linewidth=2.5,
                              edgecolor=color, facecolor=BOX_BG, zorder=2)
        ax.add_patch(rect)
        bar = plt.Rectangle((x, y + h - 0.65), w, 0.65,
                             linewidth=0, facecolor=color + "55", zorder=3)
        ax.add_patch(bar)
        ktext(ax, x + w/2, y + h - 0.33, title, fontsize=11,
              bold=True, ha="center", va="center", zorder=4)
        for i, line in enumerate(lines):
            ktext(ax, x + w/2, y + h - 1.1 - i*0.72, line, fontsize=8.5,
                  color=GRAY, ha="center", va="center", zorder=4)

    arrow_xs = [(3.3, 3.8), (6.8, 7.3), (10.3, 10.8), (14.0, 14.5), (17.0, 17.3)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 4.0), xytext=(x1, 4.0),
                    arrowprops=dict(arrowstyle="->", color=C2, lw=2.5), zorder=5)

    ktext(ax, 10.0, 7.4, "demo_ros.py 구현을 위한 전체 파이프라인",
          fontsize=14, bold=True, ha="center", va="center", zorder=5)
    ktext(ax, 10.0, 0.6,
          "SAM3 자동 라벨링 → YOLOv8 학습 → 기하학 검증 → 원근변환 → ROS2 목표 발행 → 로봇 자율 안내",
          fontsize=9, color=GRAY, ha="center", va="center", zorder=5)

    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "fig1_overview.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig1_overview.png saved")


# ── Fig 2: 모델 아키텍처 ──────────────────────────────────────────────────────
def fig2_model_arch():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    def box(x, y, w, h, title, subs, color):
        r = plt.Rectangle((x, y), w, h, lw=2, edgecolor=color,
                           facecolor=BOX_BG, zorder=2)
        ax.add_patch(r)
        ktext(ax, x + w/2, y + h - 0.45, title, fontsize=9.5,
              bold=True, color=color, ha="center", va="center", zorder=4)
        for i, s in enumerate(subs):
            ktext(ax, x + w/2, y + h - 1.0 - i*0.55, s, fontsize=8,
                  color=GRAY, ha="center", va="center", zorder=4)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=C2, lw=2), zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ktext(ax, mx, my+0.25, label, fontsize=7.5, color=C2,
                  ha="center", va="center", zorder=6)

    # Inference pipeline (top)
    box(0.3, 5.0, 2.8, 3.0, "입력 프레임",
        ["상위뷰 카메라", "1280×720", "왜곡 보정 후"], C1)
    box(3.5, 5.0, 3.2, 3.0, "YOLOv8n Backbone",
        ["C2f 블록", "Feature Pyramid", "다중 스케일 특징"], C4)
    box(7.2, 5.0, 3.2, 3.0, "Detection Head",
        ["BBox 회귀", "클래스 분류", "Confidence Score"], C3)
    box(10.9, 5.0, 3.5, 3.0, "기하학 검증기",
        ["Solidity < 0.85?", "ReachRatio > 1.80?", "AND 조건 판별"], C2)
    box(14.9, 5.0, 2.8, 3.0, "포즈 결과",
        ["hands_up", "hands_down", "(픽셀 좌표 포함)"], C5)

    arrow(3.1, 6.5, 3.5, 6.5, "프레임")
    arrow(6.7, 6.5, 7.2, 6.5, "특징맵")
    arrow(10.4, 6.5, 10.9, 6.5, "BBox+cls")
    arrow(14.4, 6.5, 14.9, 6.5, "최종 판정")

    # Training info (bottom)
    box(0.3, 0.5, 4.2, 3.8, "학습 데이터셋",
        ["761장 train / 127장 val", "hands_up / hands_down", "SAM3 자동 라벨링",
         "상위뷰 피규어 촬영", "데이터 증강 포함"], C1)
    box(5.0, 0.5, 3.5, 3.8, "학습 설정",
        ["Epochs: 200 (실제: 21)", "Batch: 16 / ImgSize: 640",
         "Optimizer: Adam(Auto)", "Patience: 50 (조기종료)", "Device: GPU 0"], C4)
    box(9.0, 0.5, 3.5, 3.8, "최종 성능",
        ["mAP50:    99.4%", "mAP50-95: 92.6%",
         "Precision: 97.4%", "Recall:    99.7%", "손실 수렴 안정"], C3)
    box(13.0, 0.5, 4.7, 3.8, "추론 설정",
        ["Conf threshold: 0.40", "모델: standing_v3/best.pt",
         "기하학 검증: 병렬 수행", "30 FPS 실시간 처리", "카메라 왜곡 보정 적용"], C5)

    ktext(ax, 9.0, 8.6, "YOLOv8n 기반 포즈 분류 모델 구조 및 학습 설정",
          fontsize=13, bold=True, ha="center", va="center")

    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "fig2_model_arch.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig2_model_arch.png saved")


# ── Fig 3: 기하학 검증 개념도 ─────────────────────────────────────────────────
def fig3_geometric():
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        print("scipy not available, skipping fig3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(DARK)

    for idx, ax in enumerate(axes):
        ax.set_facecolor(BOX_BG)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 0:
            title = "hands_down (정상 자세)"
            theta = np.linspace(0, 2*np.pi, 300)
            body_r = 0.58 + 0.07*np.cos(3*theta) + 0.04*np.sin(2*theta)
            bx = body_r * np.cos(theta)
            by = body_r * np.sin(theta)
            ax.fill(bx, by, color="#2d6a4f", alpha=0.75, zorder=2)
            hull_x = bx * 1.06
            hull_y = by * 1.06
            ax.plot(hull_x, hull_y, "--", color=C1, lw=2, label="볼록 껍질", zorder=3)
            ax.fill(hull_x, hull_y, color=C1, alpha=0.07, zorder=1)
            sol, reach = 0.92, 1.43
            col = C3
            verdict = "hands_down"
        else:
            title = "hands_up (손 들기)"
            np.random.seed(0)
            theta_body = np.linspace(0, 2*np.pi, 200)
            body_r = 0.42 + 0.04*np.cos(3*theta_body)
            bx = body_r * np.cos(theta_body)
            by = body_r * np.sin(theta_body)

            arm_angles = [np.pi*0.28, np.pi*0.72]
            for aa in arm_angles:
                arm_t = np.linspace(aa - 0.25, aa + 0.25, 40)
                arm_r = np.linspace(0.42, 1.25, 40)
                bx = np.concatenate([bx, arm_r * np.cos(arm_t)])
                by = np.concatenate([by, arm_r * np.sin(arm_t)])

            pts = np.stack([bx, by], axis=1)
            hull_obj = ConvexHull(pts)
            hull_pts = pts[hull_obj.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            ax.fill(bx, by, color="#1e3a5f", alpha=0.75, zorder=2)
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=C1, alpha=0.1, zorder=1)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], "--", color=C1, lw=2,
                    label="볼록 껍질", zorder=3)

            cx, cy = np.mean(bx), np.mean(by)
            ax.plot(cx, cy, "o", color=C2, ms=8, zorder=5)
            for aa in arm_angles:
                ax.annotate("", xy=(1.1*np.cos(aa), 1.1*np.sin(aa)),
                            xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=C4, lw=1.8))
            sol, reach = 0.71, 2.15
            col = C2
            verdict = "hands_up"

        ktext(ax, 0, 1.55, title, fontsize=11, bold=True,
              ha="center", va="center", color=WHITE)

        info = f"Solidity  = {sol:.2f}\nReachRatio = {reach:.2f}\n판정: {verdict}"
        ktext(ax, 0, -1.42, info, fontsize=9.5, color=WHITE, bold=True,
              ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.5",
                        facecolor=col+"44", edgecolor=col, lw=1.5))

        if idx == 0:
            ktext(ax, 0, 0, "몸통\n(컴팩트)", fontsize=9, color=WHITE,
                  ha="center", va="center", zorder=6)
        ax.legend(loc="upper right", fontsize=8, facecolor=BOX_BG,
                  edgecolor=GRAY, prop=KR)

    ktext(axes[0].figure, 0, 0, "", fontsize=1)  # dummy
    fig.text(0.5, 1.01,
             "기하학적 포즈 분류: Solidity + ReachRatio 이중 검증",
             ha="center", fontsize=13, fontweight="bold", color=WHITE,
             fontproperties=KR_B)
    fig.text(0.5, -0.02,
             "Solidity = 마스크 면적 / 볼록껍질 면적 (팔을 뻗으면 낮아짐)   "
             "ReachRatio = 최대 거리 / 평균 거리 (팔이 멀리 뻗으면 커짐)",
             ha="center", fontsize=9, color=GRAY, fontproperties=KR)

    plt.tight_layout(pad=0.8)
    plt.savefig(OUT / "fig3_geometric.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig3_geometric.png saved")


# ── Fig 4: ROS2 네트워크 토폴로지 ─────────────────────────────────────────────
def fig4_ros2_network():
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    def node(x, y, w, h, title, subs, color):
        r = plt.Rectangle((x, y), w, h, lw=2.5, edgecolor=color,
                           facecolor=BOX_BG, zorder=2)
        ax.add_patch(r)
        ktext(ax, x+w/2, y+h-0.55, title, fontsize=10, bold=True,
              color=color, ha="center", va="center", zorder=4)
        for i, s in enumerate(subs):
            ktext(ax, x+w/2, y+h-1.15-i*0.6, s, fontsize=8,
                  color=GRAY, ha="center", va="center", zorder=4)

    def link(x1, y1, x2, y2, label, color=C2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5), zorder=5)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ktext(ax, mx, my+0.3, label, fontsize=8, color=color,
              ha="center", va="center", zorder=6,
              bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK, edgecolor="none"))

    node(0.3, 1.5, 3.8, 5.2, "탑뷰 카메라 PC",
         ["IP: 192.168.1.9", "final_udp_top.py",
          "USB 카메라 캡처", "JPEG 인코딩",
          "UDP 청크 전송", "하트비트 발송",
          "적응형 화질 조절"], C1)

    node(4.6, 1.5, 4.0, 5.2, "AI 추론 서버",
         ["IP: 192.168.1.121", "ai_hand_raise.py",
          "UDP 프레임 수신", "YOLOv8 포즈 감지",
          "Solidity 검증", "ReachRatio 검증",
          "JSON 결과 반환"], C4)

    node(9.2, 1.5, 4.3, 5.2, "메인 서버 (ROS2)",
         ["IP: 192.168.1.120", "demo_ros.py",
          "결과 JSON 수신", "픽셀→맵 좌표 변환",
          "ROS_DOMAIN_ID=13",
          "/hand_raise_goal 발행",
          "Cooldown 5초 제어"], C3)

    node(14.1, 1.5, 3.6, 5.2, "로봇 (Nav2)",
         ["ROS_DOMAIN_ID=13", "nav2 스택",
          "PoseStamped 수신", "자율 경로 계획",
          "장애물 회피 이동",
          "매장 손님 안내"], C5)

    link(4.1, 4.1, 4.6, 4.1, "UDP 프레임\n(포트 6006)", C1)
    link(8.6, 4.1, 9.2, 4.1, "JSON 결과\n(포트 7007)", C4)
    link(13.5, 4.1, 14.1, 4.1, "ROS2 Topic\n(PoseStamped)", C3)

    ktext(ax, 9.0, 7.5, "분산 처리 기반 ROS2 네트워크 구조",
          fontsize=13, bold=True, ha="center", va="center")
    ktext(ax, 9.0, 0.6,
          "카메라 PC가 UDP로 AI 서버에 프레임 전송 → AI 서버가 포즈 감지 후 결과 반환 → 메인 서버에서 ROS2 목표 발행 → 로봇 이동",
          fontsize=9, color=GRAY, ha="center", va="center")

    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "fig4_ros2_network.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig4_ros2_network.png saved")


# ── Fig 5: 원근 변환 설명도 ───────────────────────────────────────────────────
def fig5_perspective():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor(DARK)

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(BOX_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)
        ax.tick_params(colors=GRAY, labelsize=8)

    ax1.set_xlim(0, 1280)
    ax1.set_ylim(720, 0)
    ax1.set_title("카메라 픽셀 좌표계", color=WHITE, fontsize=11,
                  fontweight="bold", fontproperties=KR_B, pad=10)
    ax1.set_xlabel("픽셀 x", color=GRAY, fontproperties=KR)
    ax1.set_ylabel("픽셀 y", color=GRAY, fontproperties=KR)

    table_px = np.array([[200,100],[1080,100],[1100,620],[180,620],[200,100]])
    ax1.plot(table_px[:,0], table_px[:,1], "-", color=C1, lw=2)
    ax1.fill(table_px[:,0], table_px[:,1], alpha=0.1, color=C1)

    figs_px = [(350, 290), (650, 240), (950, 390), (500, 490)]
    labels  = ["피규어A", "피규어B", "피규어C (손 들기)", "피규어D"]
    colors  = [C3, C3, C2, C3]
    for (px, py), lbl, col in zip(figs_px, labels, colors):
        ax1.scatter(px, py, s=180, c=col, zorder=5, marker="^")
        ax1.text(px+25, py-35, lbl, fontsize=7.5, color=col,
                 fontproperties=KR)
        rect = plt.Rectangle((px-55, py-75), 110, 110, lw=1.5,
                              edgecolor=col, facecolor="none", linestyle="--")
        ax1.add_patch(rect)

    ax1.text(640, 680, "상위뷰 카메라 프레임 (1280×720)",
             ha="center", fontsize=9, color=GRAY, fontproperties=KR)

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-4, 4)
    ax2.set_title("로봇 맵 좌표계 (미터)", color=WHITE, fontsize=11,
                  fontweight="bold", fontproperties=KR_B, pad=10)
    ax2.set_xlabel("맵 x (m)", color=GRAY, fontproperties=KR)
    ax2.set_ylabel("맵 y (m)", color=GRAY, fontproperties=KR)
    ax2.grid(True, color=GRAY, alpha=0.2)
    ax2.set_aspect("equal")

    map_coords = np.array([[-3,-2.5],[3,-2.5],[3.2,2.5],[-3.2,2.5],[-3,-2.5]])
    ax2.plot(map_coords[:,0], map_coords[:,1], "-", color=C1, lw=2)
    ax2.fill(map_coords[:,0], map_coords[:,1], alpha=0.08, color=C1)

    figs_map  = [(-1.5,1.2),(0.1,1.5),(1.8,0.3),(-0.5,-0.8)]
    mcolors = [C3, C3, C2, C3]
    for i, ((mx, my), col) in enumerate(zip(figs_map, mcolors)):
        ax2.scatter(mx, my, s=180, c=col, zorder=5, marker="^")
        ax2.text(mx+0.15, my+0.2, f"({mx:.1f},{my:.1f})m",
                 fontsize=7.5, color=col, fontproperties=KR)

    ax2.scatter(0, -3.2, s=280, c=C5, zorder=5, marker="s",
                label="로봇 위치")
    ax2.annotate("", xy=(1.8, 0.3), xytext=(0, -3.2),
                arrowprops=dict(arrowstyle="->", color=C2, lw=2,
                                linestyle="dashed"))
    ax2.text(1.0, -1.5, "목표 이동 경로", fontsize=8, color=C2,
             rotation=60, fontproperties=KR)
    ax2.legend(loc="lower right", fontsize=8, facecolor=BOX_BG,
               edgecolor=GRAY, prop=KR)

    fig.text(0.5, 0.0,
             "(mx, my) = H · (px, py, 1)ᵀ   —   단응행렬(Homography) H로 픽셀→맵 좌표 변환",
             ha="center", fontsize=10, color=GRAY, fontproperties=KR)

    plt.tight_layout(pad=0.8)
    plt.savefig(OUT / "fig5_perspective.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig5_perspective.png saved")


# ── Copy existing training result images ──────────────────────────────────────
def copy_existing():
    copies = [
        ("/home/addinedu/detection/sm3_model/runs/standing_v3/results.png",
         OUT / "fig6_training_curves.png"),
        ("/home/addinedu/detection/sm3_model/runs/standing_v3/confusion_matrix_normalized.png",
         OUT / "fig7_confusion_matrix.png"),
        ("/home/addinedu/detection/sm3_model/runs/standing_v3/val_batch0_pred.jpg",
         OUT / "fig8_val_predictions.jpg"),
        ("/home/addinedu/detection/sm3_model/runs/standing_v3/BoxF1_curve.png",
         OUT / "fig9_f1_curve.png"),
        ("/home/addinedu/detection/sm3_model/runs/standing_v3/BoxPR_curve.png",
         OUT / "fig10_pr_curve.png"),
    ]
    for src, dst in copies:
        if Path(src).exists():
            shutil.copy(src, dst)
            print(f"Copied {Path(src).name} -> {dst.name}")
        else:
            print(f"MISSING: {src}")


# ── Dataset mosaic ─────────────────────────────────────────────────────────────
def fig_dataset_mosaic():
    train_dir = Path("/home/addinedu/detection/dataset_standing/images/train")
    imgs_all  = sorted(train_dir.glob("*.jpg"))
    if len(imgs_all) < 12:
        print("Not enough training images for mosaic")
        return

    idxs     = np.linspace(0, len(imgs_all)-1, 12, dtype=int)
    selected = [imgs_all[i] for i in idxs]

    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    fig.patch.set_facecolor(DARK)
    fig.text(0.5, 0.98,
             "학습 데이터셋 샘플 — hands_up / hands_down 피규어 (상위뷰 촬영)",
             ha="center", fontsize=12, fontweight="bold", color=WHITE,
             fontproperties=KR_B, va="top")

    for ax, img_path in zip(axes.flat, selected):
        img = cv2.imread(str(img_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_facecolor(BOX_BG)

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.96])
    plt.savefig(OUT / "fig_dataset_mosaic.png", dpi=120, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("fig_dataset_mosaic.png saved")


if __name__ == "__main__":
    fig1_overview()
    fig2_model_arch()
    fig3_geometric()
    fig4_ros2_network()
    fig5_perspective()
    copy_existing()
    fig_dataset_mosaic()
    print("\nAll images generated in:", OUT)
