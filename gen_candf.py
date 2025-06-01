#!/usr/bin/env python3
import pandas as pd
import joblib
import numpy as np
import subprocess
from pathlib import Path

# 尝试导入 tflite-runtime，否则退回到 tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# -------- 配置 --------
CSV_FILE        = "rescheduling_dataset_large_10000.csv"
VECT_FILE       = "vectorizer_augmented.joblib"
TFLITE_FILE     = "rescheduler_int8.tflite"

OUT_CAND_H      = Path("candidates.h")
OUT_FEAT_H      = Path("cand_feat.h")
OUT_INIT_H      = Path("initial_instr.h")

TMP_DIR         = Path("cand_temp")
N_CAND          = 10
N_INST          = 100

RISCV_AS        = "riscv64-unknown-elf-as"
RISCV_OBJCOPY   = "riscv64-unknown-elf-objcopy"

# -------- RISC-V 汇编模板（稍加扩展以支持 beq/bne/jal） --------
ASM_TPL = {
    'add':   "  add   x{d},x{s0},x{s1}",
    'sub':   "  sub   x{d},x{s0},x{s1}",
    'addi':  "  addi  x{d},x0,{imm}",
    'lw':    "  lw    x{d},0(x2)",
    'sw':    "  sw    x{s0},0(x2)",
    # beq/bne/jal/jalr 将在 make_asm 里单独处理
}

def make_asm(seq):
    """
    把一个 instr name 列表生成 RISC-V asm 文本。
    seq 中的每个元素是 Python 端 dataset 里用的那种 "add_r1_r2_r3"、"beq_r12_r0"、"jal_r0_loop" 等。
    这里我们要：
      - 对 add/sub/addi/lw/sw 保持原样；
      - 对 beq/bne 生成真正的 RISC-V beq/bne 指令，跳转到标签 loop；
      - 对 jal_r0_label 和 jal_r0_loop 生成真实的 jal x0,label/loop；
      - 最后在末尾写出两个标签 "label:" 和 "loop:"，让跳转有目标。
    """
    lines = [
        "\t.section .text",
        "\t.global _start",
        "_start:",
    ]

    for inst in seq:
        parts = inst.split("_")
        op = parts[0]

        # 1) R-type add/sub
        if op in ("add", "sub"):
            # inst 形如 "add_r1_r2_r3"、"sub_r4_r5_r6"
            _, rd, rs0, rs1 = parts
            lines.append(ASM_TPL[op].format(d=rd[1:], s0=rs0[1:], s1=rs1[1:]))

        # 2) I-type addi
        elif op == "addi":
            # inst 形如 "addi_r14_r0_5"
            _, rd, _, imm = parts
            lines.append(ASM_TPL[op].format(d=rd[1:], imm=imm))

        # 3) lw
        elif op == "lw":
            # inst 形如 "lw_r11_mem"
            _, rd, _ = parts
            lines.append(ASM_TPL[op].format(d=rd[1:]))

        # 4) sw
        elif op == "sw":
            # inst 形如 "sw_r14_mem"
            _, rs0, _ = parts
            lines.append(ASM_TPL[op].format(s0=rs0[1:]))

        # 5) beq
        elif op == "beq":
            # inst 形如 "beq_r12_r0"
            _, rs1, rs2 = parts
            # 统一跳到 loop 标签
            lines.append(f"  beq   x{rs1[1:]},x{rs2[1:]},loop")

        # 6) bne
        elif op == "bne":
            # inst 形如 "bne_r20_r0"
            _, rs1, rs2 = parts
            lines.append(f"  bne   x{rs1[1:]},x{rs2[1:]},loop")

        # 7) jal
        elif op == "jal":
            # Python dataset 里可能会出现 "jal_r0_label" 或 "jal_r0_loop"
            # 我们把所有 jal 统一写成跳到 loop；如果你特别想区分 label/loop，可以额外加判断
            # 但为了跟 Python 端训练时常用的 "jal_r0_loop" token 对齐，这里就一律跳到 loop。
            lines.append("  jal   x0,loop")

        # 8) jalr
        elif op == "jalr":
            # 如果 dataset 里存在类似 "jalr_rD_rS_imm"，可以写成：
            # inst 形如 "jalr_r7_r3_100" （如果你 dataset 里用得上）
            _, rd, rs1, imm = parts
            lines.append(f"  jalr  x{rd[1:]},x{rs1[1:]},{imm}")

        # 9) 其他（不在我们支持范围内）→ 仍旧写 nop
        else:
            lines.append("  nop")

    # 最后加上两个标签，以供 beq
