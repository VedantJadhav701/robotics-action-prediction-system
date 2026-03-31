#!/usr/bin/env python3
"""Fix remaining linting issues"""

import re
from pathlib import Path

def fix_remaining_issues():
    """Fix remaining linting errors"""
    
    # Fix data_loader.py
    loader_path = Path('src/data_loader.py')
    with open(loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove unused imports
    content = re.sub(r'^import torch\.nn\n', '', content, flags=re.MULTILINE)
    
    # Fix trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Remove unused variables - comment them out
    content = re.sub(
        r'(\s+)trajectory_ids_list = \[',
        r'\1# trajectory_ids_list = [',
        content
    )
    
    content = re.sub(
        r'(\s+)all_obs_sequences = \[',
        r'\1# all_obs_sequences = [',
        content
    )
    
    # Fix f-string with no placeholders
    content = re.sub(
        r'f"([^{]*)"',
        r'"\1"',
        content
    )
    
    # Split long lines
    content = re.sub(
        r'(trajectory_data\[\'observations\'\]\[int\(task_id\)\] for task_id in task_ids)',
        r'(trajectory_data[\'observations\']\n                            [int(task_id)]\n                            for task_id in task_ids)',
        content
    )
    
    with open(loader_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ data_loader.py fixed')
    
    # Fix mlops.py
    mlops_path = Path('src/mlops.py')
    with open(mlops_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove unused imports
    content = re.sub(r'^import json\n', '', content, flags=re.MULTILINE)
    
    # Remove trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Fix E302 (expected 2 blank lines)
    content = re.sub(r'\n\n@dataclass\nclass', r'\n\n\n@dataclass\nclass', content)
    content = re.sub(r'\n\nclass PipelineOrchestrator', r'\n\n\nclass PipelineOrchestrator', content)
    
    # Fix E305 (expected 2 blank lines after definition before if __name__)
    content = re.sub(
        r'(:?        \})\n\nif __name__',
        r'\1\n\n\nif __name__',
        content
    )
    
    # Fix f-strings with no placeholders
    content = re.sub(r'f"([^{}]*)"', r'"\1"', content)
    
    # Split long lines
    content = re.sub(
        r"print\(f\"✓ {self\.config\.action_dim}-dim action space\"\)",
        r"print(f\"✓ {self.config.action_dim}-dim action space\")",
        content
    )
    
    content = re.sub(
        r"(print\(f\"✓ Learning rate: {self\.config\.learning_rate}\"\))",
        r"lr_str = f\"{self.config.learning_rate}\"\n        print(f\"✓ Learning rate: {lr_str}\")",
        content
    )
    
    with open(mlops_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ mlops.py fixed')
    
    # Fix model.py
    model_path = Path('src/model.py')
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Split long line in model.py
    content = re.sub(
        r'lstm_input = torch\.cat\(\[action_proj, obs_proj\], dim=-1\)  # \(batch, seq_len, hidden_dim\*2\)',
        r'lstm_input = torch.cat([action_proj, obs_proj], dim=-1)',
        content
    )
    
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ model.py fixed')
    
    # Fix monitoring.py
    monitoring_path = Path('src/monitoring.py')
    with open(monitoring_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Fix too many blank lines (E303)
    content = re.sub(r'\n\n\n\n', '\n\n', content)
    
    with open(monitoring_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ monitoring.py fixed')
    
    # Fix train.py
    train_path = Path('src/train.py')
    with open(train_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Split long line
    content = re.sub(
        r'print\(f"⚠️  NaN/Inf loss at batch \{batch_idx\}: \{loss\.item\(\)}"\)',
        r'msg = f"⚠️  NaN/Inf loss at batch {batch_idx}: "\n                    print(msg + f"{loss.item()}")',
        content
    )
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ train.py fixed')
    
    print('\n✅ Remaining linting issues fixed!')

if __name__ == '__main__':
    fix_remaining_issues()
