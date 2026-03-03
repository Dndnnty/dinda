import os
from jinja2 import Environment, FileSystemLoader

# Setup Jinja2 env
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
env = Environment(loader=FileSystemLoader(template_dir))

for root, dirs, files in os.walk(template_dir):
    for filename in files:
        if filename.endswith('.html'):
            rel_path = os.path.relpath(os.path.join(root, filename), template_dir)
            # Use forward slashes for Jinja
            rel_path = rel_path.replace('\\', '/')
            try:
                env.get_template(rel_path)
                print(f"[OK] {rel_path}")
            except Exception as e:
                print(f"[ERROR] {rel_path}: {e}")
