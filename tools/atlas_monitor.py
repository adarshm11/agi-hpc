#!/usr/bin/env python3
"""Atlas System Monitor — High-density terminal UI.

Shows GPUs, CPUs, memory, disk I/O, network, and processes
in a single terminal with 2-second refresh.

Usage: ssh claude@192.168.0.7 'source env/bin/activate && python atlas_monitor.py'
Requires: psutil (pip install psutil)
"""

import curses
import subprocess
import time
import os
import socket

try:
    import psutil
except ImportError:
    os.system("pip install psutil")
    import psutil


def get_gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
             "memory.used,memory.total,power.draw,power.limit,fan.speed",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        gpus = []
        for line in r.stdout.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 9:
                gpus.append({
                    "idx": int(p[0]), "name": p[1],
                    "temp": float(p[2]), "util": float(p[3]),
                    "mem_used": float(p[4]), "mem_total": float(p[5]),
                    "power": float(p[6]), "power_cap": float(p[7]),
                    "fan": p[8],
                })
        r2 = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        procs = []
        for line in r2.stdout.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4 and p[1].isdigit():
                procs.append({"pid": p[1], "mem": p[2], "name": p[3][:40]})
        return gpus, procs
    except Exception:
        return [], []


prev_disk = None
prev_net = None
prev_time = None


def bar(width, pct, ch="█", empty="░"):
    filled = int(width * pct / 100)
    return ch * filled + empty * (width - filled)


def color_for_pct(pct):
    if pct > 80:
        return 3  # red
    if pct > 50:
        return 4  # yellow
    return 2  # green


def draw(stdscr):
    global prev_disk, prev_net, prev_time

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(2000)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_CYAN, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_BLUE, -1)
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLUE)

    while True:
        try:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            now = time.time()

            # Header
            hostname = socket.gethostname()
            uptime = now - psutil.boot_time()
            up_d = int(uptime // 86400)
            up_h = int((uptime % 86400) // 3600)
            up_m = int((uptime % 3600) // 60)
            load = os.getloadavg()
            header = f" ⬡ {hostname.upper()} │ up {up_d}d {up_h}h {up_m}m │ load {load[0]:.1f} {load[1]:.1f} {load[2]:.1f} │ {time.strftime('%H:%M:%S')} "
            stdscr.addstr(0, 0, header.ljust(w), curses.color_pair(8) | curses.A_BOLD)

            row = 2

            # ═══ GPUs ═══
            gpus, gpu_procs = get_gpu_info()
            stdscr.addstr(row, 0, "─── GPUs ", curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(row, 9, "─" * (w - 10), curses.color_pair(7))
            row += 1

            for g in gpus:
                mem_pct = g["mem_used"] / g["mem_total"] * 100
                util_bar = bar(20, g["util"])
                mem_bar = bar(20, mem_pct)

                stdscr.addstr(row, 1, f"GPU {g['idx']}", curses.color_pair(5) | curses.A_BOLD)
                stdscr.addstr(row, 7, f" {g['name'][:22]}", curses.color_pair(1))

                # Util bar
                stdscr.addstr(row, 32, "Util ", curses.color_pair(1))
                stdscr.addstr(row, 37, util_bar, curses.color_pair(color_for_pct(g["util"])))
                stdscr.addstr(row, 58, f"{g['util']:5.1f}%", curses.color_pair(color_for_pct(g["util"])) | curses.A_BOLD)

                # VRAM bar
                stdscr.addstr(row, 66, "VRAM ", curses.color_pair(1))
                stdscr.addstr(row, 71, mem_bar, curses.color_pair(color_for_pct(mem_pct)))
                stdscr.addstr(row, 92, f"{g['mem_used']/1024:.1f}/{g['mem_total']/1024:.0f}G", curses.color_pair(1))

                # Temp, power, fan
                tc = 3 if g["temp"] > 80 else (4 if g["temp"] > 65 else 2)
                stdscr.addstr(row, 105, f"{g['temp']:.0f}°C", curses.color_pair(tc))
                stdscr.addstr(row, 111, f"{g['power']:.0f}/{g['power_cap']:.0f}W", curses.color_pair(1))
                stdscr.addstr(row, 123, f"fan {g['fan']}%", curses.color_pair(1))
                row += 1

            if gpu_procs:
                for p in gpu_procs[:4]:
                    stdscr.addstr(row, 4, f"PID {p['pid']:>6}  {p['mem']:>6} MiB  {p['name']}", curses.color_pair(7))
                    row += 1

            row += 1

            # ═══ CPU ═══
            cpu_pct = psutil.cpu_percent(percpu=True)
            cpu_avg = sum(cpu_pct) / len(cpu_pct)
            freq = psutil.cpu_freq()
            stdscr.addstr(row, 0, "─── CPU ", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(row, 8, f" {len(cpu_pct)} threads │ avg {cpu_avg:.0f}% │ {freq.current:.0f} MHz " if freq else f" {len(cpu_pct)} threads │ avg {cpu_avg:.0f}%", curses.color_pair(1))
            stdscr.addstr(row, 60, "─" * (w - 61), curses.color_pair(7))
            row += 1

            # CPU bars — 2 rows of cores
            half = len(cpu_pct) // 2
            for start in [0, half]:
                chunk = cpu_pct[start:start + half]
                label = f" {start:>2}-{start+len(chunk)-1:>2} "
                stdscr.addstr(row, 0, label, curses.color_pair(1))
                col = 7
                for pct in chunk:
                    ch = "█" if pct > 50 else ("▄" if pct > 10 else "░")
                    stdscr.addstr(row, col, ch, curses.color_pair(color_for_pct(pct)))
                    col += 1
                avg_chunk = sum(chunk) / len(chunk)
                stdscr.addstr(row, col + 1, f"{avg_chunk:5.1f}%", curses.color_pair(color_for_pct(avg_chunk)))
                row += 1

            row += 1

            # ═══ Memory ═══
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage("/")

            stdscr.addstr(row, 0, "─── Memory & Disk ", curses.color_pair(6) | curses.A_BOLD)
            stdscr.addstr(row, 18, "─" * (w - 19), curses.color_pair(7))
            row += 1

            # RAM
            stdscr.addstr(row, 1, "RAM  ", curses.color_pair(1))
            stdscr.addstr(row, 6, bar(30, mem.percent), curses.color_pair(color_for_pct(mem.percent)))
            stdscr.addstr(row, 37, f" {mem.used/1e9:.1f}/{mem.total/1e9:.0f}G ({mem.percent:.0f}%)", curses.color_pair(1))

            # Swap
            swap_pct = swap.percent if swap.total > 0 else 0
            stdscr.addstr(row, 68, "Swap ", curses.color_pair(1))
            stdscr.addstr(row, 73, bar(15, swap_pct), curses.color_pair(color_for_pct(swap_pct)))
            stdscr.addstr(row, 89, f" {swap.used/1e9:.1f}/{swap.total/1e9:.0f}G", curses.color_pair(1))
            row += 1

            # Disk
            stdscr.addstr(row, 1, "Disk ", curses.color_pair(1))
            stdscr.addstr(row, 6, bar(30, disk.percent), curses.color_pair(color_for_pct(disk.percent)))
            stdscr.addstr(row, 37, f" {disk.used/1e9:.0f}/{disk.total/1e9:.0f}G ({disk.percent:.0f}%)", curses.color_pair(1))
            row += 1

            row += 1

            # ═══ I/O ═══
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            stdscr.addstr(row, 0, "─── I/O ", curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(row, 8, "─" * (w - 9), curses.color_pair(7))
            row += 1

            if prev_disk and prev_time:
                dt = max(now - prev_time, 0.1)
                dr = (disk_io.read_bytes - prev_disk.read_bytes) / dt / 1e6
                dw = (disk_io.write_bytes - prev_disk.write_bytes) / dt / 1e6
                ri = (disk_io.read_count - prev_disk.read_count) / dt
                wi = (disk_io.write_count - prev_disk.write_count) / dt
                nr = (net_io.bytes_recv - prev_net.bytes_recv) / dt / 1e6
                ns = (net_io.bytes_sent - prev_net.bytes_sent) / dt / 1e6

                stdscr.addstr(row, 1, "Disk R:", curses.color_pair(1))
                stdscr.addstr(row, 9, f"{dr:7.1f} MB/s", curses.color_pair(2))
                stdscr.addstr(row, 24, f"({ri:.0f} iops)", curses.color_pair(7))

                stdscr.addstr(row, 40, "Disk W:", curses.color_pair(1))
                stdscr.addstr(row, 48, f"{dw:7.1f} MB/s", curses.color_pair(4))
                stdscr.addstr(row, 63, f"({wi:.0f} iops)", curses.color_pair(7))

                stdscr.addstr(row, 80, "Net ↓:", curses.color_pair(1))
                stdscr.addstr(row, 87, f"{nr:6.1f} MB/s", curses.color_pair(2))
                stdscr.addstr(row, 101, "↑:", curses.color_pair(1))
                stdscr.addstr(row, 104, f"{ns:6.1f} MB/s", curses.color_pair(4))
            else:
                stdscr.addstr(row, 1, "collecting baseline...", curses.color_pair(7))

            prev_disk = disk_io
            prev_net = net_io
            prev_time = now
            row += 1

            row += 1

            # ═══ Top Processes ═══
            stdscr.addstr(row, 0, "─── Top Processes ", curses.color_pair(3) | curses.A_BOLD)
            stdscr.addstr(row, 18, "─" * (w - 19), curses.color_pair(7))
            row += 1

            stdscr.addstr(row, 1, f"{'PID':>7}  {'Name':<25} {'CPU%':>6}  {'MEM%':>6}  {'RSS':>8}", curses.color_pair(7))
            row += 1

            procs = sorted(
                psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "memory_info"]),
                key=lambda p: p.info.get("cpu_percent", 0) or 0, reverse=True
            )
            remaining = h - row - 1
            for p in procs[:min(remaining, 15)]:
                info = p.info
                cpu = info.get("cpu_percent", 0) or 0
                mem = info.get("memory_percent", 0) or 0
                rss = (info.get("memory_info") or psutil._common.pmem(0, 0, 0, 0)).rss / 1e6
                name = (info.get("name") or "?")[:25]
                cc = 3 if cpu > 50 else (4 if cpu > 10 else 1)
                try:
                    stdscr.addstr(row, 1, f"{info['pid']:>7}  {name:<25} {cpu:>6.1f}  {mem:>6.1f}  {rss:>7.0f}M", curses.color_pair(cc))
                except curses.error:
                    pass
                row += 1

            stdscr.refresh()

        except curses.error:
            pass

        key = stdscr.getch()
        if key == ord('q') or key == 27:
            break


if __name__ == "__main__":
    curses.wrapper(draw)
