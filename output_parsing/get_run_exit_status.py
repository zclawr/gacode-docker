import os
from collections import defaultdict

def collect_statuses_per_batch(base_dir):
    batch_summary = {}

    for batch in sorted(os.listdir(base_dir)):
        cgyro_dir = os.path.join(base_dir, batch, "cgyro")
        if not os.path.isdir(cgyro_dir):
            continue

        exit_counts = defaultdict(int)
        error_messages = []
        error_keys = []  # Track input_dirs with errors

        for input_dir in sorted(os.listdir(cgyro_dir)):
            input_path = os.path.join(cgyro_dir, input_dir)
            info_path = os.path.join(input_path, "out.cgyro.info")

            if not os.path.isfile(info_path):
                continue

            status_line = None
            local_errors = []

            with open(info_path, 'r') as f:
                for line in f:
                    if line.startswith("EXIT: (CGYRO)"):
                        status_line = line.strip()
                    elif line.startswith("ERROR: (CGYRO)"):
                        local_errors.append(line.strip())

            if status_line:
                exit_counts[status_line] += 1
            elif local_errors:
                exit_counts["ERROR"] += 1
                error_messages.extend(local_errors)
                error_keys.append(input_dir)  # Save this ky/input-dir
            else:
                exit_counts["NO STATUS OR ERROR"] += 1
                error_keys.append(input_dir)

        batch_summary[batch] = {
            "exit_counts": dict(exit_counts),
            "errors": error_messages,
            "error_keys": error_keys
        }

    return batch_summary


def print_batch_summary(summary):
    for batch, data in summary.items():
        print(f"\n📦 Batch: {batch}")
        print("   Exit Status Counts:")
        for status, count in data["exit_counts"].items():
            print(f"     - {status}: {count}")

        if data["error_keys"]:
            print("   ❌ Error ky/input directories:")
            for err, key in zip(data["errors"], data["error_keys"]):
                print(f"    {err} - {key}")
        else:
            print("   ✅ No CGYRO errors reported.")


# === Example usage ===
if __name__ == "__main__":
    base_path = "/Users/wesleyliu/Documents/Github/gacode-docker/test_outputs"
    batch_results = collect_statuses_per_batch(base_path)
    print_batch_summary(batch_results)