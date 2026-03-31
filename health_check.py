#!/usr/bin/env python3
"""Quick system health and metrics status check"""

from datetime import datetime

import requests


def check_service(name, url, timeout=5):
    """Check if a service is responding"""
    try:
        resp = requests.get(url, timeout=timeout)
        status = "✅" if resp.status_code == 200 else f"⚠️ ({resp.status_code})"
        return status
    except Exception as e:
        return f"❌ {str(e)[:30]}"


def get_metrics():
    """Get current metrics from API"""
    try:
        resp = requests.get("http://localhost:8000/stats", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_prometheus_targets():
    """Get Prometheus target status"""
    try:
        resp = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            targets = data["data"]["activeTargets"]
            return targets
    except Exception:
        pass
    return []


def main():
    print("\n" + "=" * 70)
    print(" 🚀 MONITORING SYSTEM STATUS CHECK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Check services
    print("📡 SERVICE HEALTH")
    print("-" * 70)
    services = [
        ("FastAPI", "http://localhost:8000/health"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("Grafana", "http://localhost:3000/api/health"),
        ("MongoDB", "http://localhost:27017/"),  # Won't respond but we check connection
    ]

    for name, url in services:
        status = check_service(name, url)
        print(f"  {name:20} {status}")

    # Get metrics from API
    print("\n📊 API STATISTICS")
    print("-" * 70)
    stats = get_metrics()
    if stats:
        print(f"  Total Requests:      {stats.get('total_requests', 0)}")
        print(f"  Successful:          {stats.get('successful', 0)}")
        print(f"  Failed:              {stats.get('failed', 0)}")
        print(f"  Success Rate:        {stats.get('success_rate', 0)*100:.1f}%")
        print(f"  Avg Latency:         {stats.get('avg_latency_ms', 0):.2f}ms")
    else:
        print("  ❌ Could not fetch stats")

    # Get Prometheus targets
    print("\n🎯 PROMETHEUS TARGETS")
    print("-" * 70)
    targets = get_prometheus_targets()
    if targets:
        for target in targets:
            job = target["labels"].get("job", "unknown")
            instance = target["labels"].get("instance", "unknown")
            health = target["health"]
            status_icon = "✅" if health == "up" else "❌"
            print(f"  {status_icon} {job:20} ({instance})")
    else:
        print("  ❌ Could not fetch targets")

    print("\n" + "=" * 70)
    print(" 📈 NEXT: Visit http://localhost:3000 (Grafana) to visualize data")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
