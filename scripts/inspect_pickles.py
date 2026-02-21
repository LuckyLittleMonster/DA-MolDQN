"""Inspect the structure of experiment pickle files."""
import pickle
import sys

base = "Experiments"
files = {
    "dock_history": f"{base}/reasyn_dock_proxy_1_history.pickle",
    "dock_paths": f"{base}/reasyn_dock_proxy_1_paths.pickle",
    "dock_recent": f"{base}/reasyn_dock_proxy_1_recent_episodes.pickle",
    "moo2_history": f"{base}/reasyn_multi_proxy_2_history.pickle",
    "moo2_paths": f"{base}/reasyn_multi_proxy_2_paths.pickle",
    "moo2_recent": f"{base}/reasyn_multi_proxy_2_recent_episodes.pickle",
    "moo1_history": f"{base}/reasyn_multi_proxy_1_history.pickle",
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"  {name}: {path}")
    print(f"{'='*60}")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  Type: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"    {k}: list[{len(v)}]", end="")
                    if len(v) > 0:
                        s = repr(v[0])[:200]
                        if isinstance(v[0], dict):
                            print(f" first_keys={list(v[0].keys())}")
                        else:
                            print(f" first={s}")
                    else:
                        print()
                elif isinstance(v, dict):
                    ks = list(v.keys())[:10]
                    print(f"    {k}: dict[{len(v)}] keys={ks}...")
                else:
                    s = repr(v)[:200]
                    print(f"    {k}: {type(v).__name__} = {s}")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
            if len(data) > 0:
                first = data[0]
                print(f"  First element type: {type(first).__name__}")
                if isinstance(first, dict):
                    print(f"  First element keys: {list(first.keys())}")
                    for k2, v2 in first.items():
                        s = repr(v2)[:200]
                        print(f"    {k2}: {type(v2).__name__} = {s}")
                else:
                    print(f"  First element: {repr(first)[:300]}")
                if len(data) > 1:
                    last = data[-1]
                    print(f"  Last element type: {type(last).__name__}")
                    if isinstance(last, dict):
                        print(f"  Last element keys: {list(last.keys())}")
                        for k2, v2 in last.items():
                            s = repr(v2)[:200]
                            print(f"    {k2}: {type(v2).__name__} = {s}")
        else:
            print(f"  Value: {repr(data)[:500]}")
    except Exception as e:
        print(f"  ERROR: {e}")
