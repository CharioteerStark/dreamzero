#!/usr/bin/env python3
"""Standalone gripper actuation test — isolates the gripper control path from the model.

Cycles set_gripper_position across the range with wait=True (blocking) and reads back
get_gripper_position, so you can see:
  (a) whether the gripper physically MOVES at all (control problem?),
  (b) the valid range (0-85 vs 0-850) -> resolves the ×10 scaling question,
  (c) which end is open vs closed.

  python test_gripper.py --right-arm-ip 192.168.10.201 --arm right
  python test_gripper.py --arm both --positions 850,600,400,200,0
"""
import argparse, time
from xarm.wrapper import XArmAPI


def setup(ip, label):
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.clean_warn()
    arm.clean_error()
    time.sleep(0.2)
    code_en = arm.set_gripper_enable(True)
    try:
        code_mode = arm.set_gripper_mode(0)  # 0 = location/position mode
    except Exception as e:
        code_mode = f"n/a ({e})"
    time.sleep(0.3)
    print(f"[{label}] set_gripper_enable -> {code_en} | set_gripper_mode(0) -> {code_mode}")
    return arm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left-arm-ip", default="192.168.10.22")
    p.add_argument("--right-arm-ip", default="192.168.10.201")
    p.add_argument("--arm", choices=["left", "right", "both"], default="right")
    p.add_argument("--positions", default="850,600,400,200,0",
                   help="comma-separated targets to command (watch which opens/closes)")
    p.add_argument("--speed", type=int, default=2000)
    args = p.parse_args()

    arms = {}
    if args.arm in ("left", "both"):
        arms["left"] = setup(args.left_arm_ip, "left")
    if args.arm in ("right", "both"):
        arms["right"] = setup(args.right_arm_ip, "right")

    for lbl, arm in arms.items():
        code0, pos0 = arm.get_gripper_position()
        print(f"\n[{lbl}] start: get_gripper_position -> code={code0} pos={pos0}")
        for tgt in [int(x) for x in args.positions.split(",")]:
            set_code = arm.set_gripper_position(tgt, wait=True, speed=args.speed)
            time.sleep(0.5)
            rd_code, pos = arm.get_gripper_position()
            print(f"[{lbl}] set_gripper_position({tgt:>4}) -> set_code={set_code} | "
                  f"readback pos={pos} (read_code={rd_code})  <-- did it physically move?")
    print("\nWatch the hardware: which target opened vs closed, did it move at all, and what's the readback range?")


if __name__ == "__main__":
    main()
