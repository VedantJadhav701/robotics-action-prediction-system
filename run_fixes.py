#!/usr/bin/env python3
"""Fix all linting errors in src/*.py"""

import re
from pathlib import Path

def fix_all_files():
    """Fix all Python files"""
    
    # Fix train.py
    train_path = Path('src/train.py')
    with open(train_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove blank lines with whitespace
    lines = content.split('\n')
    lines = ['' if line.strip() == '' else line for line in lines]
    content = '\n'.join(lines)
    
    # Fix long lines
    content = re.sub(
        r"'final_train_loss': float\(self\.train_losses\[-1\]\) if self\.train_losses else None,",
        "'final_train_loss': float(self.train_losses[-1])\n            if self.train_losses else None,",
        content
    )
    content = re.sub(
        r"'final_val_loss': float\(self\.val_losses\[-1\]\) if self\.val_losses else None,",
        "'final_val_loss': float(self.val_losses[-1])\n            if self.val_losses else None,",
        content
    )
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ train.py fixed')
    
    # Fix monitoring.py
    monitoring_path = Path('src/monitoring.py')
    with open(monitoring_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove blank lines with whitespace
    lines = content.split('\n')
    lines = ['' if line.strip() == '' else line for line in lines]
    content = '\n'.join(lines)
    
    # Add blank line before class
    content = re.sub(r'^class ', '\n\nclass ', content, flags=re.MULTILINE)
    content = re.sub(r'\n\n\nclass ', r'\n\nclass ', content)  # Remove triple blank lines
    
    with open(monitoring_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ monitoring.py fixed')
    
    # Fix mlops.py
    mlops_path = Path('src/mlops.py')
    with open(mlops_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove blank lines with whitespace
    lines = content.split('\n')
    lines = ['' if line.strip() == '' else line for line in lines]
    content = '\n'.join(lines)
    
    # Fix spacing before module-level function
    content = re.sub(r'(\n        \})\n\nif __name__', r'\1\n\n\nif __name__', content)
    
    with open(mlops_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ mlops.py fixed')
    
    # Fix model.py
    model_path = Path('src/model.py')
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove blank lines with whitespace
    lines = content.split('\n')
    lines = ['' if line.strip() == '' else line for line in lines]
    content = '\n'.join(lines)
    
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ model.py fixed')
    
    # Fix data_loader.py
    loader_path = Path('src/data_loader.py')
    if loader_path.exists():
        with open(loader_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        lines = ['' if line.strip() == '' else line for line in lines]
        content = '\n'.join(lines)
        
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✓ data_loader.py fixed')
    
    # Fix __init__.py if it exists
    init_path = Path('src/__init__.py')
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        lines = ['' if line.strip() == '' else line for line in lines]
        content = '\n'.join(lines)
        
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✓ __init__.py fixed')
    
    print('\n✅ All linting fixes applied!')

if __name__ == '__main__':
    fix_all_files()
