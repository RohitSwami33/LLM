import os
import socket
import urllib.request

print("==== KAGGLE PROBE ====")


def check_torch():
    try:
        import torch
        print("torch:", getattr(torch, "__version__", "?"))
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch FAIL:", type(e).__name__, e)


def check_env():
    print("KAGGLE_API_TOKEN:", bool(os.environ.get("KAGGLE_API_TOKEN")))
    print("KAGGLE_USERNAME:", os.environ.get("KAGGLE_USERNAME"))
    print("KAGGLE_KEY:", bool(os.environ.get("KAGGLE_KEY")))


def check_dns(host):
    try:
        ip = socket.gethostbyname(host)
        print(f"DNS {host}: OK -> {ip}")
    except Exception as e:
        print(f"DNS {host}: FAIL {type(e).__name__}: {e}")


def check_get(name, url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "probe"})
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"GET {name}: {r.status} {r.headers.get('content-type') or ''}")
    except Exception as e:
        print(f"GET {name}: FAIL {type(e).__name__}: {e}")


def check_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("kaggle AUTH: OK as", api.get_config_value(api.CONFIG_NAME_USER))
    except Exception as e:
        print("kaggle AUTH: FAIL", type(e).__name__, str(e)[:300])


check_torch()
check_env()
for host in ["www.kaggle.com", "api.kaggle.com", "www.google.com"]:
    check_dns(host)
check_get("www.kaggle.com", "https://www.kaggle.com/")
check_get("api.kaggle.com", "https://api.kaggle.com/")
check_api()
print("=" * 25 + " PROBE DONE " + "=" * 25)