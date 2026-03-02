import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def get_python_files():

    current_file = os.path.basename(__file__)
    py_files = []
    
    for file in os.listdir('.'):
        if file.endswith('.py') and file != current_file:
            py_files.append(file)
    
    py_files.sort()
    return py_files

def find_project_root(start_path=None):
    if start_path is None:
        start_path = Path.cwd()
    
    markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', 'README.md']
    
    current = Path(start_path).resolve()
    
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    return current.parent.parent

def log_message(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')
    
    print(log_line)

def run_python_files():
    py_files = get_python_files()
    
    if not py_files:
        return
    
    project_root = find_project_root()
    log_file = "execution_log.txt"
    
    log_message(log_file, f"{len(py_files)}")
    log_message(log_file, f"{', '.join(py_files)}")
    log_message(log_file, f"{project_root}")
    
    current_script_dir = Path(__file__).parent.resolve()
    
    success_count = 0
    fail_count = 0
    
    for i, py_file in enumerate(py_files, 1):
        try:
            py_file_path = current_script_dir / py_file
            log_message(log_file, f"{i}: {py_file}")
            log_message(log_file, f"{py_file_path}")
            
            result = subprocess.run(
                [sys.executable, str(py_file_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=3600,
                env={**os.environ, 'PYTHONPATH': str(project_root)}
            )
            
            if result.returncode == 0:
                success_count += 1
                log_message(log_file, f"{py_file} finished")
                
                if result.stdout.strip():
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        if line.strip():
                            log_message(log_file, f"{py_file} : {line.strip()}")
                    
            else:
                fail_count += 1
                log_message(log_file, f"{py_file} failed : {result.returncode}")
                if result.stderr.strip():
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[:5]:
                        log_message(log_file, f"{py_file} error: {line.strip()}")
                
                log_message(log_file, "failed")
                break
            
        except FileNotFoundError as e:
            fail_count += 1
            log_message(log_file, f"{py_file}: FileNotFoundError {str(e)}")
            log_message(log_file, "failed")
            break
            
        except Exception as e:
            fail_count += 1
            log_message(log_file, f"{py_file} Exception: {str(e)}")
            log_message(log_file, "failed")
            break
    
    log_message(log_file, f"finished: {success_count}, failed: {fail_count}, total: {len(py_files)}")
    log_message(log_file, "=" * 50 + "\n")

if __name__ == "__main__":
    run_python_files()