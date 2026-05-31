# Network Setup

The NAO robot, the laptop running the dialogue, and the tablet showing the memory book must all be on the same local network. This document covers first-time setup of a new router (in case TAILSCALE is not ready), as well as the connectivity checks needed before every session.

---

## 1. Adding NAO to a New Router

The first time a NAO robot is used with a new router, it must be taught the new network. This is a one-time setup per robot, per router.

1. **Connect NAO to a working network.** Use the lab Wi-Fi or a phone hotspot, importantly it is anything NAO can already reach.
2. **Connect your laptop to the same network.**
3. **Find NAO's IP address.** Press NAO's chest button once. NAO will announce its IP aloud.
4. **Open NAO's web interface** by entering the IP address in your laptop's browser.
5. **Add the new router network** in NAO's network settings (e.g. `Netgear40`). Enter the password.
6. **Connect NAO to the new network.** NAO will disconnect from the temporary one and join the router.
7. **Switch your laptop to the same router network.**
8. **Find NAO's new IP** by pressing the chest button again. The IP will have changed.
9. **Verify in the browser** that NAO's web interface is reachable on the new IP.

NAO will now remember this network and reconnect automatically when powered on within range.

---

## 2. Connecting the Tablet

1. Connect the tablet to the same router network as the laptop and NAO
2. Open Chrome on the iPad
3. Navigate to `http://<laptop-ip>:8080`. The laptop's IP is printed by the tablet server at startup
4. The tablet book cover should appear

The tablet will lock the screen if left idle. For long sessions, disable auto-lock in iPad settings during the study, then re-enable it afterwards.

---

## 3. Connectivity Checklist

Before every session, verify all three pairwise connections:

| Connection | Test |
|------------|------|
| Laptop ↔ Robot | `ping <robot_ip>` from the laptop terminal |
| Laptop ↔ Tablet | Open `http://<laptop-ip>:8080` on the iPad — book cover appears |
| Robot ↔ Tablet | After the dialogue starts, the tablet shows the correct child name |

If any of the three fails, do not proceed with the session until it is resolved. See [troubleshooting](troubleshooting.md) for diagnosis steps.

---

## 4. Updating the Robot IP in Config

NAO's IP is stored in `CRI-BRANCH-BASIC4_0.py`:

```python
def __init__(self, openai_env_path=None, nao_ip="10.0.0.241"):
```

If NAO's IP changes (for example after switching routers), update this line to match. The IP can also be passed at startup but the default in the code should reflect the current lab setup.

---

## 5. Next Steps

- [Microphone setup](microphone.md) — DJI wireless mic
- [Running a session](running.md) — full startup sequence
