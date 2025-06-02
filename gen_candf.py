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
CSV_FILE        = "rescheduling_dataset_rich.csv"
VECT_FILE       = "vectorizer.joblib"
TFLITE_FILE     = "rescheduler_int8.tflite"

OUT_CAND_H      = Path("candidates.h")
OUT_FEAT_H      = Path("cand_feat.h")
OUT_INIT_H      = Path("initial_instr.h")

TMP_DIR         = Path("cand_temp")
N_CAND          = 10
N_INST          = 100

RISCV_AS        = "riscv64-unknown-elf-as"
RISCV_OBJCOPY   = "riscv64-unknown-elf-objcopy"

# ----------------------------------------
# make_asm(seq):
#   把 Python 里类似 "add_r1_r2_r3","beq_r12_r0","jal_r0_loop" 这样的指令名字列表
#   翻译成“真实的” RISC-V 汇编代码：
#     - add/sub/addi/lw/sw 按模板生成
#     - beq/bne 生成 “beq xRs0,xRs1,loop”
#     - jal  生成 “jal x0,loop”
#     - jalr 生成 “jalr xRd,xRs1,imm”（如果有 jalr）
#   汇编文件末尾加 “loop: j loop”。
# ----------------------------------------
ASM_TPL = {
    'add':   "  add   x{d},x{s0},x{s1}",
    'sub':   "  sub   x{d},x{s0},x{s1}",
    'addi':  "  addi  x{d},x0,{imm}",
    'lw':    "  lw    x{d},0(x2)",
    'sw':    "  sw    x{s0},0(x2)",
}

def make_asm(seq):
    """
    seq: Python list，比如 ["add_r1_r2_r3", "beq_r12_r0", "sub_r4_r5_r6", ...]
    返回一个字符串，包含：
      .section .text
      .global _start
    _start:
      <每条指令对应的 RISC-V 汇编行>
    loop: j loop
    """
    lines = [
        "\t.section .text",
        "\t.global _start",
        "_start:",
    ]
    for inst in seq:
        parts = inst.split("_")
        op = parts[0]

        if op in ("add", "sub"):
            # "add_r1_r2_r3" → rd='1', rs0='2', rs1='3'
            rd = parts[1][1:]
            rs0 = parts[2][1:]
            rs1 = parts[3][1:]
            lines.append(ASM_TPL[op].format(d=rd, s0=rs0, s1=rs1))

        elif op == "addi":
            # "addi_r14_r0_5" → rd='14', imm='5'
            rd = parts[1][1:]
            imm = parts[3]
            lines.append(ASM_TPL[op].format(d=rd, imm=imm))

        elif op == "lw":
            # "lw_r11_mem" → rd='11'
            rd = parts[1][1:]
            lines.append(ASM_TPL[op].format(d=rd))

        elif op == "sw":
            # "sw_r14_mem" → rs0='14'
            rs0 = parts[1][1:]
            lines.append(ASM_TPL[op].format(s0=rs0))

        elif op == "beq":
            # "beq_r12_r0" → rs0='12', rs1='0'
            rs0 = parts[1][1:]
            rs1 = parts[2][1:]
            lines.append(f"  beq   x{rs0},x{rs1},loop")

        elif op == "bne":
            # "bne_r20_r0" → rs0='20', rs1='0'
            rs0 = parts[1][1:]
            rs1 = parts[2][1:]
            lines.append(f"  bne   x{rs0},x{rs1},loop")

        elif op == "jal":
            # "jal_r0_label" 或 "jal_r0_loop" → 直接跳到 loop
            lines.append("  jal   x0,loop")

        elif op == "jalr":
            # "jalr_rD_rS_imm" → rd, rs1, imm
            rd = parts[1][1:]
            rs1 = parts[2][1:]
            imm = parts[3]
            lines.append(f"  jalr  x{rd},x{rs1},{imm}")

        else:
            # 其他未知操作，当作 nop
            lines.append("  nop")

    # 最后加一个无限循环标签
    lines.append("loop: j loop")
    return "\n".join(lines)


# ----------------------------------------
# assemble_and_pad(s_path, bin_path):
#   1) riscv64-unknown-elf-as 汇编 .s → .o
#   2) riscv64-unknown-elf-objcopy 提取 .text 段的裸二进制
#   3) 截断或 pad 到 N_INST*4 字节（每指令 4 bytes）
# ----------------------------------------
def assemble_and_pad(s_path: Path, bin_path: Path):
    o_path = s_path.with_suffix(".o")
    # 1) 汇编
    subprocess.run([RISCV_AS, "-march=rv32im", "-mabi=ilp32",
                    "-o", str(o_path), str(s_path)], check=True)
    # 2) 提取裸二进制 .text 段
    subprocess.run([RISCV_OBJCOPY, "-O", "binary",
                    "--only-section", ".text",
                    str(o_path), str(bin_path)], check=True)
    data = bin_path.read_bytes()[: N_INST*4]
    # 3) pad：nop 的编码是 0x00000013（addi x0, x0, 0）
    nop = (0x00000013).to_bytes(4, "little")
    data = data + nop * ((N_INST*4 - len(data) + 3)//4)
    return data[: N_INST*4]


# ----------------------------------------
# main(): 生成 initial_instr.h、candidates.h、cand_feat.h
# ----------------------------------------
def main():
    TMP_DIR.mkdir(exist_ok=True)

    # 1) 读 CSV，取所有 sequence 并计算 cycles
    df = pd.read_csv(CSV_FILE)
    df['seq_list'] = df['sequence'].apply(eval)

    # 找出 dataset 里真正的最小仿真周期
    best_cycle = df['cycles'].min()
    # 找第一条 cycles > best_cycle 的记录，让它做 initial
    cand_df = df[df['cycles'] > best_cycle]
    if not cand_df.empty:
        row = cand_df.iloc[0]
    else:
        # 如果全部记录都一样（极端情况），退回到第一行
        row = df.iloc[0]

    init_seq = row['seq_list']
    init_cycle = row['cycles']
    print(f"Selected initial sequence with cycle = {init_cycle} (dataset best = {best_cycle})")

    # 2) 生成 initial.s / initial.bin，并写 initial_instr.h
    init_s   = TMP_DIR/"initial.s"
    init_b   = TMP_DIR/"initial.bin"
    init_s.write_text(make_asm(init_seq))
    init_bytes = assemble_and_pad(init_s, init_b)

    with OUT_INIT_H.open("w") as f:
        f.write("// Auto-generated by gen_candf.py\n")
        f.write("#pragma once\n\n")
        f.write(f"#define INIT_COUNT {N_INST}\n\n")
        f.write("static const uint32_t initial_instr[INIT_COUNT] = {\n")
        for i in range(N_INST):
            w = int.from_bytes(init_bytes[i*4:(i+1)*4], "little")
            f.write(f"  0x{w:08X},\n")
        f.write("};\n")
    print("Wrote initial_instr.h")

    # 3) 随机抽样 N_CAND 条做 candidates（用于 candidates.h）
    samples = df['seq_list'].sample(N_CAND, random_state=123).tolist()
    bins = []
    for i, seq in enumerate(samples):
        s_file = TMP_DIR/f"cand{i}.s"
        b_file = TMP_DIR/f"cand{i}.bin"
        s_file.write_text(make_asm(seq))
        bins.append(assemble_and_pad(s_file, b_file))

    with OUT_CAND_H.open("w") as f:
        f.write("// Auto-generated by gen_candf.py\n")
        f.write("#pragma once\n\n")
        f.write(f"#define N_CAND {N_CAND}\n")
        f.write(f"#define N_INST {N_INST}\n\n")
        f.write("static const uint32_t cand_seq[N_CAND][N_INST] = {\n")
        for data in bins:
            words = [f"0x{int.from_bytes(data[j*4:(j+1)*4],'little'):08X}"
                     for j in range(N_INST)]
            f.write("  { " + ", ".join(words) + " },\n")
        f.write("};\n")
    print("Wrote candidates.h")

    # 4) 生成 cand_feat.h
    vectorizer = joblib.load(VECT_FILE)
    FEATURE_DIM = len(vectorizer.get_feature_names_out())

    # 4.1) 读取 TFLite 模型输入量化参数
    interp = Interpreter(model_path=TFLITE_FILE)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    scale, zp = inp_det['quantization']

    # 4.2) 文本 → n-gram → 浮点 → 量化 int8
    lines = [" ".join(seq) for seq in samples]
    X = vectorizer.transform(lines).toarray().astype(np.float32)
    Xq = np.clip(np.round(X/scale + zp), -128, 127).astype(np.int8)

    with OUT_FEAT_H.open("w") as f:
        f.write("// Auto-generated by gen_candf.py\n")
        f.write("#pragma once\n\n")
        f.write(f"#define N_CAND      {N_CAND}\n")
        f.write(f"#define FEATURE_DIM {FEATURE_DIM}\n\n")
        f.write("static const int8_t cand_feat[N_CAND][FEATURE_DIM] = {\n")
        for row in Xq:
            f.write("  { " + ", ".join(str(int(v)) for v in row) + " },\n")
        f.write("};\n")
    print("Wrote cand_feat.h")


if __name__ == "__main__":
    main()
