#!/usr/bin/env python3
"""Comprehensive flake8 fixes for all src files"""

import re
from pathlib import Path


def fix_file(filepath):
    """Fix all linting issues in a file"""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    newlines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix W293: blank line contains whitespace
        if line.strip() == "" and len(line) > 1:
            newlines.append("\n")
        else:
            newlines.append(line)

        i += 1

    content = "".join(newlines)

    # Fix specific issues by file
    if "model.py" in filepath:
        # Remove F401: math import
        content = content.replace("import math\n", "")
        # Fix long line in __init__
        content = re.sub(
            r"    def __init__\(self, action_dim, obs_dim, hidden_dim=64, num_layers=2, dropout=0\.3\)",
            "    def __init__(self, action_dim, obs_dim, hidden_dim=64,\n                 num_layers=2, dropout=0.3)",
            content,
        )

    if "train.py" in filepath:
        # Fix long lines
        content = re.sub(
            r"'final_train_loss': float\(self\.train_losses\[-1\]\) if self\.train_losses else None,",
            "'final_train_loss': float(self.train_losses[-1])\n            if self.train_losses else None,",
            content,
        )
        content = re.sub(
            r"'final_val_loss': float\(self\.val_losses\[-1\]\) if self\.val_losses else None,",
            "'final_val_loss': float(self.val_losses[-1])\n            if self.val_losses else None,",
            content,
        )
        content = re.sub(
            r'print\(f"⚠️  NaN/Inf loss at batch \{batch_idx\}: \{loss\.item\(\)}"\)',
            'print(f"⚠️  NaN/Inf loss at batch {batch_idx}: "\n                          f"{loss.item()}")',
            content,
        )

    if "monitoring.py" in filepath:
        # Fix E302: expected 2 blank lines
        content = re.sub(r"\nclass MetricsCollector:", r"\n\nclass MetricsCollector:", content)

    if "mlops.py" in filepath:
        # Fix E305: expected 2 blank lines after function/class
        content = re.sub(r"(?<=\n    \})\n\ndef ", r"\n\n\ndef ", content)

    # Ensure proper spacing between top-level functions
    content = re.sub(r"\n\ndef ", r"\n\n\ndef ", content)
    content = re.sub(r"\n\nclass ", r"\n\n\nclass ", content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Fixed {Path(filepath).name}")


def main():
    """Fix all Python files in src"""
    src_dir = Path("src")
    for py_file in sorted(src_dir.glob("*.py")):
        if py_file.name != "__pycache__":
            fix_file(str(py_file))

    print("\n✅ All linting fixes applied!")


if __name__ == "__main__":
    main()
