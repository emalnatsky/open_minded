import concurrent.futures
import ipaddress
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlparse


NAOQI_PORT = 9559


@dataclass(frozen=True)
class LocalIPv4Network:
    address: str
    network: ipaddress.IPv4Network
    source: str = ""


@dataclass(frozen=True)
class NAODiscoveryResult:
    selected_ip: str
    configured_ip: str = ""
    candidates: Tuple[str, ...] = ()
    scanned_networks: Tuple[str, ...] = ()
    used_configured: bool = False
    auto_discovery_enabled: bool = True


class NAODiscoveryError(RuntimeError):
    def __init__(self, message: str, candidates=(), scanned_networks=()):
        super().__init__(message)
        self.candidates = tuple(candidates or ())
        self.scanned_networks = tuple(scanned_networks or ())


def normalize_network_host(value: str) -> str:
    clean = str(value or "").strip()
    if "://" in clean:
        parsed = urlparse(clean)
        clean = parsed.hostname or clean
    return clean.strip().strip("/")


def parse_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    clean = str(value).strip().lower()
    if not clean:
        return default
    if clean in ("1", "true", "yes", "y", "on", "auto"):
        return True
    if clean in ("0", "false", "no", "n", "off", "disabled"):
        return False
    return default


def parse_float(value: Optional[str], default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def is_usable_ipv4(ip: ipaddress.IPv4Address) -> bool:
    return not (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or str(ip).startswith("169.254.")
    )


def _add_network(items, address: str, network_value: str, source: str):
    try:
        ip = ipaddress.ip_address(str(address).strip())
        network = ipaddress.ip_network(str(network_value).strip(), strict=False)
    except ValueError:
        return
    if ip.version != 4 or network.version != 4 or not is_usable_ipv4(ip):
        return
    items.append(LocalIPv4Network(address=str(ip), network=network, source=source))


def _networks_from_psutil():
    try:
        import psutil
    except Exception:
        return []
    items = []
    for iface_name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if getattr(addr, "family", None) != socket.AF_INET:
                continue
            ip = getattr(addr, "address", "")
            netmask = getattr(addr, "netmask", "")
            if ip and netmask:
                _add_network(items, ip, f"{ip}/{netmask}", f"psutil:{iface_name}")
    return items


def _networks_from_netsh():
    if platform.system().lower() != "windows":
        return []
    try:
        output = subprocess.check_output(
            ["netsh", "interface", "ipv4", "show", "addresses"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3,
        )
    except Exception:
        return []
    items = []
    current_ip = ""
    for line in output.splitlines():
        ip_match = re.search(r"\bIP Address:\s*([0-9.]+)", line, re.IGNORECASE)
        if ip_match:
            current_ip = ip_match.group(1)
            continue
        prefix_match = re.search(r"\bSubnet Prefix:\s*([0-9.]+/\d+)", line, re.IGNORECASE)
        if prefix_match and current_ip:
            _add_network(items, current_ip, prefix_match.group(1), "netsh")
            current_ip = ""
    return items


def _networks_from_ip_addr():
    try:
        output = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3,
        )
    except Exception:
        return []
    items = []
    for line in output.splitlines():
        match = re.search(r"\binet\s+([0-9.]+)/(\d+)", line)
        if match:
            ip = match.group(1)
            prefix = match.group(2)
            iface = line.split(":", 2)[1].strip() if ":" in line else "ip"
            _add_network(items, ip, f"{ip}/{prefix}", f"ip:{iface}")
    return items


def _mask_to_prefix(mask: str) -> Optional[int]:
    clean = str(mask or "").strip()
    if clean.startswith("0x"):
        try:
            return bin(int(clean, 16)).count("1")
        except ValueError:
            return None
    try:
        return ipaddress.IPv4Network(f"0.0.0.0/{clean}").prefixlen
    except ValueError:
        return None


def _networks_from_ifconfig():
    try:
        output = subprocess.check_output(
            ["ifconfig"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3,
        )
    except Exception:
        return []
    items = []
    iface = "ifconfig"
    for line in output.splitlines():
        if line and not line.startswith((" ", "\t")):
            iface = line.split(":", 1)[0]
        match = re.search(r"\binet\s+([0-9.]+).*?\bnetmask\s+([0-9a-fA-Fx.]+)", line)
        if not match:
            continue
        ip = match.group(1)
        prefix = _mask_to_prefix(match.group(2))
        if prefix is not None:
            _add_network(items, ip, f"{ip}/{prefix}", f"ifconfig:{iface}")
    return items


def _networks_from_hostname():
    items = []
    ips = set()
    try:
        _host, _aliases, host_ips = socket.gethostbyname_ex(socket.gethostname())
        ips.update(host_ips)
    except Exception:
        pass
    for target in ("8.8.8.8", "1.1.1.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((target, 80))
            ips.add(sock.getsockname()[0])
        except OSError:
            pass
        finally:
            sock.close()
    for ip in sorted(ips):
        _add_network(items, ip, f"{ip}/24", "hostname-fallback")
    return items


def list_local_ipv4_networks() -> Tuple[LocalIPv4Network, ...]:
    candidates = []
    candidates.extend(_networks_from_psutil())
    candidates.extend(_networks_from_netsh())
    candidates.extend(_networks_from_ip_addr())
    candidates.extend(_networks_from_ifconfig())
    candidates.extend(_networks_from_hostname())

    deduped = []
    seen = set()
    for item in candidates:
        key = (item.address, str(item.network))
        if key in seen:
            continue
        if any(
            item.address == existing.address and item.network.subnet_of(existing.network)
            for existing in deduped
        ):
            continue
        deduped = [
            existing for existing in deduped
            if not (
                item.address == existing.address
                and existing.network.subnet_of(item.network)
            )
        ]
        seen.add(key)
        deduped.append(item)
    return tuple(deduped)


def list_arp_ipv4_candidates() -> Tuple[str, ...]:
    try:
        output = subprocess.check_output(
            ["arp", "-a"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3,
        )
    except Exception:
        return ()

    candidates = []
    seen = set()
    for line in output.splitlines():
        if line.strip().lower().startswith("interface:"):
            continue
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line):
            raw = match.group(0)
            try:
                ip = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if ip.version != 4 or not is_usable_ipv4(ip):
                continue
            clean = str(ip)
            if clean not in seen:
                seen.add(clean)
                candidates.append(clean)
    return tuple(candidates)


def tcp_port_open(ip: str, port: int = NAOQI_PORT, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((str(ip), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _scan_network_for_hosts(entry: LocalIPv4Network, max_hosts_per_network: int):
    network = entry.network
    if network.num_addresses > max_hosts_per_network + 2:
        network = ipaddress.ip_network(f"{entry.address}/24", strict=False)
    return [str(host) for host in network.hosts()]


def scan_for_nao_candidates(
    networks: Sequence[LocalIPv4Network],
    timeout_seconds: float = 3.0,
    port: int = NAOQI_PORT,
    port_check_fn: Callable[[str, int, float], bool] = tcp_port_open,
    priority_hosts: Sequence[str] = (),
    max_workers: int = 64,
    max_hosts_per_network: int = 1024,
    max_total_hosts: int = 4096,
) -> Tuple[str, ...]:
    local_ips = {entry.address for entry in networks}
    hosts = []
    seen = set()
    network_values = [entry.network for entry in networks]

    def add_host(raw_host: str):
        try:
            ip = ipaddress.ip_address(str(raw_host).strip())
        except ValueError:
            return False
        clean = str(ip)
        if ip.version != 4 or not is_usable_ipv4(ip):
            return False
        if clean in seen or clean in local_ips:
            return False
        if network_values and not any(ip in network for network in network_values):
            return False
        seen.add(clean)
        hosts.append(clean)
        return True

    for host in priority_hosts or ():
        add_host(host)

    for entry in networks:
        for host in _scan_network_for_hosts(entry, max_hosts_per_network):
            add_host(host)
            if len(hosts) >= max_total_hosts:
                break
        if len(hosts) >= max_total_hosts:
            break

    if not hosts:
        return ()

    per_host_timeout = min(0.35, max(0.05, timeout_seconds / 4.0))
    candidates = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(port_check_fn, host, port, per_host_timeout): host
        for host in hosts
    }
    try:
        for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
            host = futures[future]
            try:
                if future.result():
                    candidates.append(host)
            except Exception:
                pass
    except concurrent.futures.TimeoutError:
        pass
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    return tuple(sorted(candidates, key=lambda ip: ipaddress.ip_address(ip)))


def discover_nao_ip(
    configured_ip: str = "",
    auto_discover: bool = True,
    timeout_seconds: float = 3.0,
    port: int = NAOQI_PORT,
    networks_fn: Callable[[], Sequence[LocalIPv4Network]] = list_local_ipv4_networks,
    port_check_fn: Callable[[str, int, float], bool] = tcp_port_open,
    priority_hosts_fn: Callable[[], Sequence[str]] = list_arp_ipv4_candidates,
) -> NAODiscoveryResult:
    configured_ip = normalize_network_host(configured_ip)
    timeout_seconds = max(0.2, float(timeout_seconds or 3.0))
    check_timeout = min(0.75, max(0.1, timeout_seconds / 2.0))

    if configured_ip and port_check_fn(configured_ip, port, check_timeout):
        return NAODiscoveryResult(
            selected_ip=configured_ip,
            configured_ip=configured_ip,
            candidates=(configured_ip,),
            used_configured=True,
            auto_discovery_enabled=auto_discover,
        )

    if not auto_discover:
        if not configured_ip:
            raise NAODiscoveryError(
                "NAO auto-discovery is disabled and no NAO IP is configured."
            )
        return NAODiscoveryResult(
            selected_ip=configured_ip,
            configured_ip=configured_ip,
            candidates=(),
            used_configured=True,
            auto_discovery_enabled=False,
        )

    networks = tuple(networks_fn() or ())
    scanned_networks = tuple(str(entry.network) for entry in networks)
    priority_hosts = tuple(priority_hosts_fn() or ())
    candidates = scan_for_nao_candidates(
        networks,
        timeout_seconds=timeout_seconds,
        port=port,
        port_check_fn=port_check_fn,
        priority_hosts=priority_hosts,
    )

    if len(candidates) == 1:
        return NAODiscoveryResult(
            selected_ip=candidates[0],
            configured_ip=configured_ip,
            candidates=candidates,
            scanned_networks=scanned_networks,
            used_configured=False,
            auto_discovery_enabled=True,
        )

    if len(candidates) > 1:
        raise NAODiscoveryError(
            "Multiple possible NAO robots were found. Set CRI_NAO_IP or update test_config.pl.",
            candidates=candidates,
            scanned_networks=scanned_networks,
        )

    if configured_ip:
        message = (
            f"Configured NAO IP {configured_ip} is not reachable, and no NAO was "
            "discovered on the local networks."
        )
    else:
        message = "No NAO IP is configured, and no NAO was discovered on the local networks."
    raise NAODiscoveryError(message, candidates=(), scanned_networks=scanned_networks)


def discover_nao_ip_from_env(
    configured_ip: str = "",
    env: Optional[dict] = None,
    **kwargs,
) -> NAODiscoveryResult:
    env = env or os.environ
    override_ip = normalize_network_host(env.get("CRI_NAO_IP", ""))
    auto_discover = parse_bool(env.get("CRI_NAO_AUTO_DISCOVER"), default=True)
    timeout_seconds = parse_float(env.get("CRI_NAO_DISCOVERY_TIMEOUT_SECONDS"), 3.0)
    return discover_nao_ip(
        configured_ip=override_ip or configured_ip,
        auto_discover=auto_discover,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
