#!/usr/bin/env python3
"""Fix linting errors in src files"""

import re
from pathlib import Path

def fix_whitespace_on_blank_lines(content):
    """Remove trailing whitespace from blank lines"""
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if line.strip() == '':
            fixed_lines.append('')
        else:
            fixed_lines.append(line)
    return '\n'.join(fixed_lines)

def fix_spacing_around_classes(content):
    """Ensure 2 blank lines before class/function definitions"""
    # Fix class definitions
    content = re.sub(r'\n\nclass ', r'\n\n\nclass ', content)
    content = re.sub(r'(\n\nclass )', r'\n\nclass ', content)
    
    # Fix function definitions at module level (not indented)
    lines = content.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        # Check for unindented function definitions after non-function, non-class lines
        if i > 0 and line.startswith('def ') and not lines[i-1].startswith(' '):
            if i >= 2 and not lines[i-2].startswith('class ') and not lines[i-2].startswith('def '):
                if not (i >= 2 and lines[i-2].strip() == ''):
                    # Need 2 blank lines
                    pass
    
    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file"""
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Fix 1: Remove trailing whitespace from blank lines
    content = fix_whitespace_on_blank_lines(content)
    
    # Fix 2: Remove unused imports
    if filepath.endswith('model.py'):
        if 'import math' in content and 'math.' not in content:
            content = content.replace('import math\n\n', '')
    
    if filepath.endswith('train.py'):
        if 'import torch.nn as nn' in content and 'nn.' not in content:
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if 'import torch.nn as nn' not in line:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
    
    # Fix 3: Fix known long lines
    fixes = {
        'model.py': [
            (r'    def __init__\(self, action_dim, obs_dim, hidden_dim=64, num_layers=2, dropout=0\.3\):',
             '    def __init__(self, action_dim, obs_dim, hidden_dim=64,\n                 num_layers=2, dropout=0.3):'),
        ],
        'train.py': [
            (r'                    print\(f"⚠️  NaN/Inf loss at batch \{batch_idx\}: \{loss\.item\(\)}"\)',
             '                    print(f"⚠️  NaN/Inf loss at batch {batch_idx}: '
             f'{loss.item()}")'),
        ]
    }
    
    filename = Path(filepath).name
    if filename in fixes:
        for pattern, replacement in fixes[filename]:
            content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed")
    else:
        print(f"  - No changes needed")

def main():
    """Fix all src files"""
    src_dir = Path('src')
    
    for py_file in sorted(src_dir.glob('*.py')):
        process_file(str(py_file))
    
    print("\n✅ Linting fixes applied!")

if __name__ == '__main__':
    main()
