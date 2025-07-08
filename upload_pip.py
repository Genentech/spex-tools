import os
import shutil
import subprocess
import sys

env_python = sys.executable
env_pip = os.path.join(os.path.dirname(env_python), 'pip')

def run(cmd, check=True):
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=check)

# Очистка старых артефактов
for d in ['dist', 'build', 'spex.egg-info']:
    if os.path.exists(d):
        shutil.rmtree(d)

# Проверка путей
print("Python:", env_python)
print("Pip:", env_pip)

# Установка зависимостей в текущее окружение
run([env_pip, 'install', '--force-reinstall', '--no-cache-dir', 'build', 'twine'])

# Сборка пакета
run([env_python, '-m', 'build'])

# Загрузка на PyPI
run(['twine', 'upload', '--repository', 'testpypi'] + sorted([f'dist/{f}' for f in os.listdir('dist')]))