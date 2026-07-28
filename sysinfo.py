"""
Capture the system configuration at run time, so timing results are self-describing.

Uses psutil if available for clock speed / memory; otherwise falls back to the
standard library. Never raises -- missing info is reported as 'n/a'.
"""

import os
import platform


def system_info():
    """Return a multi-line string describing the machine and its current load."""
    lines = []
    lines.append(f"platform    : {platform.platform()}")
    lines.append(f"processor   : {platform.processor() or 'n/a'}")
    lines.append(f"python      : {platform.python_version()}")

    logical = os.cpu_count()
    lines.append(f"cpu cores   : {logical} logical")

    try:
        import psutil

        # clock speed (MHz): current / max where available
        freq = psutil.cpu_freq()
        if freq:
            lines.append(f"clock speed : {freq.current:.0f} MHz current"
                         + (f", {freq.max:.0f} MHz max" if freq.max else ""))
        else:
            lines.append("clock speed : n/a")

        # current CPU load (percent, sampled briefly)
        load = psutil.cpu_percent(interval=0.3)
        lines.append(f"current load: {load:.0f}% CPU")

        # memory: total and currently used
        vm = psutil.virtual_memory()
        gb = 1024 ** 3
        lines.append(f"memory total: {vm.total / gb:.1f} GB")
        lines.append(f"memory used : {vm.used / gb:.1f} GB "
                     f"({vm.percent:.0f}% in use) at run start")
    except ImportError:
        # stdlib fallback -- less detail, but no dependency required
        lines.append("clock speed : n/a (install psutil for clock/memory detail)")
        if hasattr(os, "getloadavg"):
            la1, la5, la15 = os.getloadavg()
            lines.append(f"current load: {la1:.2f} (1-min load average)")
        else:
            lines.append("current load: n/a")
        lines.append("memory total: n/a (install psutil)")
        lines.append("memory used : n/a (install psutil)")

    return "\n".join(lines)


if __name__ == "__main__":
    print(system_info())
