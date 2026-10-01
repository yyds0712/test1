"""
积分运算演示程序
================
演示定积分的数值计算方法：
  1. 矩形法（中点公式）
  2. 梯形法
  3. 辛普森法（Simpson）
并与解析解对比，可视化积分区域与误差收敛。

核心计算仅依赖 Python 标准库（math）；
若安装了 matplotlib，则额外生成可视化图像，否则自动跳过。
"""

import os
import math

# 尝试导入 matplotlib（可选）
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，避免无 GUI 时阻塞
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    # 中文显示配置（Windows 常见中文字体）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except ImportError:
    HAS_MATPLOTLIB = False

# ========== 全局配置 ==========
# 图像输出目录：脚本所在目录下的 outputs 文件夹
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def linspace(a, b, n):
    """等价于 numpy.linspace，生成 n 个等间距点（纯 Python）。"""
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


# ========== 被积函数 ==========
def f(x):
    """被积函数: f(x) = x * exp(x)"""
    return x * math.exp(x)


def antiderivative(x):
    """f(x) = x*exp(x) 的原函数: (x - 1) * exp(x)"""
    return (x - 1) * math.exp(x)


def exact_integral(a, b):
    """解析解: ∫_a^b x*exp(x) dx"""
    return antiderivative(b) - antiderivative(a)


# ========== 数值积分方法 ==========
def rectangle_method(f, a, b, n):
    """矩形法（中点公式），O(h^2)"""
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        s += f(x_mid)
    return h * s


def trapezoidal_method(f, a, b, n):
    """梯形法，O(h^2)"""
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s


def simpson_method(f, a, b, n):
    """辛普森法，要求 n 为偶数，O(h^4)"""
    if n % 2 != 0:
        n += 1  # 强制为偶数
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n, 2):
        s += 2 * f(a + i * h)
    return h / 3 * s


# ========== 可视化：积分区域几何意义 ==========
def plot_integration_geometry(a, b, n=8):
    """绘制三种方法的几何意义对比图（需要 matplotlib）"""
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_fine = linspace(a, b, 1000)
    y_fine = [f(x) for x in x_fine]

    methods = [
        ("矩形法（中点）", rectangle_method, "skyblue"),
        ("梯形法", trapezoidal_method, "lightgreen"),
        ("辛普森法", simpson_method, "wheat"),
    ]

    for ax, (name, method, color) in zip(axes, methods):
        ax.plot(x_fine, y_fine, "b-", linewidth=2, label=r"$f(x)=x e^x$")
        ax.fill_between(x_fine, y_fine, alpha=0.15, color="blue", label="真实面积")

        h = (b - a) / n
        if method == rectangle_method:
            for i in range(n):
                xi = a + i * h
                xm = xi + h / 2
                rect = plt.Rectangle(
                    (xi, 0), h, f(xm),
                    edgecolor="black", facecolor=color, alpha=0.6
                )
                ax.add_patch(rect)
        elif method == trapezoidal_method:
            for i in range(n):
                x0, x1 = a + i * h, a + (i + 1) * h
                y0, y1 = f(x0), f(x1)
                ax.fill(
                    [x0, x1, x1, x0], [0, 0, y1, y0],
                    edgecolor="black", facecolor=color, alpha=0.6
                )
        else:
            for i in range(0, n, 2):
                x0, x1, x2 = a + i * h, a + (i + 1) * h, a + (i + 2) * h
                xs = linspace(x0, x2, 50)
                y0, y1, y2 = f(x0), f(x1), f(x2)
                d01, d02, d12 = x0 - x1, x0 - x2, x1 - x2
                ys = []
                for xv in xs:
                    L0 = (xv - x1) * (xv - x2) / (d01 * d02)
                    L1 = (xv - x0) * (xv - x2) / ((x1 - x0) * d12)
                    L2 = (xv - x0) * (xv - x1) / ((x2 - x0) * (x2 - x1))
                    ys.append(y0 * L0 + y1 * L1 + y2 * L2)
                ax.fill_between(xs, ys, edgecolor="black",
                                facecolor=color, alpha=0.6)

        val = method(f, a, b, n)
        ax.set_title(f"{name}\n估计值 = {val:.6f}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"数值积分几何意义 (n={n})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "integration_geometry.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] 积分几何意义图 -> {path}")


# ========== 可视化：误差收敛分析 ==========
def plot_error_convergence(a, b):
    """绘制不同 n 下各方法的误差收敛曲线（需要 matplotlib）"""
    if not HAS_MATPLOTLIB:
        return
    ns = [4, 8, 16, 32, 64, 128, 256, 512]
    exact = exact_integral(a, b)

    errors = {
        "矩形法": [],
        "梯形法": [],
        "辛普森法": [],
    }
    for n in ns:
        errors["矩形法"].append(abs(rectangle_method(f, a, b, n) - exact))
        errors["梯形法"].append(abs(trapezoidal_method(f, a, b, n) - exact))
        errors["辛普森法"].append(abs(simpson_method(f, a, b, n) - exact))

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"矩形法": "o", "梯形法": "s", "辛普森法": "^"}
    colors = {"矩形法": "blue", "梯形法": "green", "辛普森法": "red"}
    for name, errs in errors.items():
        ax.loglog(ns, errs, marker=markers[name], color=colors[name],
                  linewidth=2, label=name)

    # 参考斜率线
    n_ref = [ns[0], ns[-1]]
    ax.loglog(n_ref, [errors["梯形法"][0] * (ns[0] / n) ** 2 for n in n_ref],
              "k--", alpha=0.5, label=r"$O(n^{-2})$ 参考")
    ax.loglog(n_ref, [errors["辛普森法"][0] * (ns[0] / n) ** 4 for n in n_ref],
              "k:", alpha=0.5, label=r"$O(n^{-4})$ 参考")

    ax.set_xlabel("区间数 n", fontsize=12)
    ax.set_ylabel("绝对误差 |数值解 - 解析解|", fontsize=12)
    ax.set_title("数值积分误差收敛性 (双对数坐标)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "error_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] 误差收敛图 -> {path}")


# ========== 主程序 ==========
def main():
    # 积分区间
    a, b = 0.0, 2.0

    print("=" * 60)
    print("        定积分数值计算演示")
    print("=" * 60)
    print(f"被积函数:  f(x) = x * e^x")
    print(f"积分区间:  [{a}, {b}]")
    print(f"原函数:    F(x) = (x - 1) * e^x")
    print()

    exact = exact_integral(a, b)
    print(f"解析解 (精确值):  {exact:.10f}")
    print("-" * 60)

    n_list = [4, 8, 16, 64, 256, 1024]
    print(f"{'n':>8} {'矩形法':>14} {'梯形法':>14} {'辛普森法':>14}")
    print("-" * 60)
    for n in n_list:
        r = rectangle_method(f, a, b, n)
        t = trapezoidal_method(f, a, b, n)
        s = simpson_method(f, a, b, n)
        print(f"{n:>8} {r:>14.8f} {t:>14.8f} {s:>14.8f}")
    print("-" * 60)
    print(f"{'精确值':>8} {exact:>14.8f}")
    print()

    # 误差对比
    print("误差分析 (n=64):")
    n = 64
    err_r = abs(rectangle_method(f, a, b, n) - exact)
    err_t = abs(trapezoidal_method(f, a, b, n) - exact)
    err_s = abs(simpson_method(f, a, b, n) - exact)
    print(f"  矩形法误差:   {err_r:.2e}")
    print(f"  梯形法误差:   {err_t:.2e}")
    print(f"  辛普森法误差: {err_s:.2e}")
    print()

    # 生成可视化（仅当 matplotlib 可用时）
    if HAS_MATPLOTLIB:
        print("正在生成可视化图像...")
        plot_integration_geometry(a, b, n=8)
        plot_error_convergence(a, b)
        print()
        print("演示完成！图像已保存到 outputs 目录。")
    else:
        print("[提示] 未安装 matplotlib，已跳过图像生成。")
        print("  如需生成可视化图像，请安装：python -m pip install matplotlib")
        print()
        print("演示完成（数值计算部分已执行）。")


if __name__ == "__main__":
    main()
