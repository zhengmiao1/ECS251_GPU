import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_card(ax, x, y, w, h, title, subtitle="", fc="#F8FAFC", ec="#4B5563", lw=1.4):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="semibold",
        color="#111827",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=11,
            color="#6B7280",
        )


def add_arrow(ax, p1, p2, label=None, color="#374151", rad=0.0, label_shift=(0, 0.012)):
    ar = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.3,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(ar)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_shift[0]
        my = (p1[1] + p2[1]) / 2 + label_shift[1]
        ax.text(mx, my, label, fontsize=10, color="#4B5563", ha="center", va="bottom")


def draw(path="gpu_scheduler_architecture_paper.png"):
    fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Architecture Overview of the GPU Job Scheduler",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )

    db_fc = "#EEF2FF"
    ui_fc = "#ECFDF5"
    ctl_fc = "#FFFBEB"
    mon_fc = "#F5F3FF"
    ext_fc = "#F9FAFB"

    # Layout tuned for vertical arrow alignment:
    # left branch (CLI -> DB-left) and right branch (DAE -> MON -> SMI -> DB-right)
    db = (0.20, 0.73, 0.60, 0.14)
    cli = (0.17, 0.43, 0.30, 0.15)  # center x = 0.32
    dae = (0.53, 0.43, 0.30, 0.15)  # center x = 0.68
    mon = (0.53, 0.20, 0.30, 0.13)  # center x = 0.68
    smi = (0.53, 0.05, 0.30, 0.10)  # center x = 0.68

    add_card(
        ax,
        *db,
        "SQLite Job Store (scheduler.db)",
        "Job states: pending / running / done / failed / cancelled",
        fc=db_fc,
    )
    add_card(ax, *cli, "User Submission", "submit.py / cancel.py", fc=ui_fc, lw=1.5)
    add_card(ax, *dae, "Scheduling Engine", "daemon.py", fc=ctl_fc, lw=1.5)
    add_card(ax, *mon, "GPU Monitoring Unit", "gpu_monitor.py", fc=mon_fc, lw=1.5)
    add_card(ax, *smi, "System Telemetry Source", "nvidia-smi", fc=ext_fc, lw=1.5)

    # Vertical arrows to top store
    add_arrow(
        ax,
        (cli[0] + cli[2] * 0.50, cli[1] + cli[3]),
        (db[0] + db[2] * 0.20, db[1]),
        "write",
        label_shift=(0.0, 0.008),
    )
    add_arrow(
        ax,
        (dae[0] + dae[2] * 0.50, dae[1] + dae[3]),
        (db[0] + db[2] * 0.80, db[1]),
        "write / update",
        label_shift=(0.0, 0.008),
    )
    add_arrow(
        ax,
        (dae[0] + dae[2] * 0.50, dae[1]),
        (mon[0] + mon[2] * 0.50, mon[1] + mon[3]),
        "spawn / read",
        label_shift=(0.07, 0.0),
    )
    add_arrow(
        ax,
        (mon[0] + mon[2] * 0.50, mon[1]),
        (smi[0] + smi[2] * 0.50, smi[1] + smi[3]),
        "query",
        label_shift=(0.05, 0.0),
    )

    ax.text(0.50, 0.90, "Persistence Layer", ha="center", va="center", fontsize=11, color="#6B7280")
    ax.text(0.32, 0.61, "Interface Layer", ha="center", va="center", fontsize=11, color="#6B7280")
    ax.text(0.68, 0.61, "Control Layer", ha="center", va="center", fontsize=11, color="#6B7280")
    ax.text(0.68, 0.36, "Telemetry Layer", ha="center", va="center", fontsize=11, color="#6B7280")

    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
