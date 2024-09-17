from PyInstaller.utils.hooks import collect_all, copy_metadata
import streamlit, site, sys

block_cipher = None

# Add the exclusion of problematic modules
excluded_modules = ['torch.distributions', 'torch.testing']

# Collect streamlit data, excluding some unnecessary items
streamlit_data = collect_all('streamlit')
streamlit_static_dir = os.path.join(os.path.dirname(streamlit.__file__), 'static')

# Get the site-packages directory for the current environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # We're in a virtual environment
    site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
else:
    # We're not in a virtual environment
    site_packages = site.getsitepackages()[0]

torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

# Prepare the datas list

datas = []
datas.extend([
        (os.path.join(site_packages, "streamlit"), "streamlit"),
        (os.path.join(site_packages, "torch"), "torch"),
        (os.path.join(site_packages, "transformers"), "transformers"),
    ])
datas.extend(copy_metadata('streamlit'))
datas.extend(copy_metadata('importlib_metadata'))
#datas.extend(streamlit_data[0])
datas.extend(streamlit_data[1])
#datas.extend(streamlit_data[2])
datas.extend(torch_datas)

# Include streamlit data
import streamlit
streamlit_root = os.path.dirname(streamlit.__file__)
datas.extend([(streamlit_root, 'streamlit')])

# Assuming the paths are relative to the root of the project or the current working directory
project_dir = os.path.abspath('.')  # Use current working directory in spec file context


additional_datas = [
    # Use relative paths or project_dir for portability
    (os.path.join(project_dir, 'api_cost'), 'api_cost'),
    (os.path.join(project_dir, 'bin'), 'bin'),
    (os.path.join(project_dir, 'custom_prompts'), 'custom_prompts'),
    (os.path.join(project_dir, 'demo'), 'demo'),
    (os.path.join(project_dir, 'img'), 'img'),
    (os.path.join(project_dir, 'pages'), 'pages'),
    (os.path.join(project_dir, 'settings'), 'settings'),
    (os.path.join(project_dir, 'uploads'), 'uploads'),
    (os.path.join(project_dir, 'uploads_small'), 'uploads_small'),
    (os.path.join(project_dir, '.streamlit'), '.streamlit'),
    (os.path.join(project_dir, 'vouchervision'), 'vouchervision'),
    (os.path.join(project_dir, 'app.py'), '.'),  # Root-level file
    (os.path.join(project_dir, 'create_desktop_shortcut.py'), '.'),  # Root-level file
    (os.path.join(project_dir, 'create_desktop_shortcut_mac.py'), '.'),  # Root-level file
    (os.path.join(project_dir, 'LICENSE'), '.'),  # Root-level file
    (os.path.join(project_dir, 'api_status.yaml'), '.'),  # Root-level file
    (os.path.join(project_dir, 'environment.yaml'), '.'),  # Root-level file
    (os.path.join(project_dir, 'install_dependencies.sh'), '.'),  # Root-level file
    (os.path.join(project_dir, 'requirements.txt'), '.'),  # Root-level file
    (os.path.join(project_dir, 'VoucherVision_Reference.yaml'), '.'),  # Root-level file
    (streamlit_static_dir, 'streamlit/static'),  # Streamlit static directory
]
datas.extend(additional_datas)

a = Analysis(
    ['run_VoucherVision.py'],
    pathex=['C:\\Users\\willwe\\Documents\\VoucherVision'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'streamlit',
        'torch',
        'transformers',
        'pkg_resources.py2_warn',
        'streamlit', 'importlib_metadata', 
        'streamlit.web', 'streamlit.runtime', 
        'streamlit.components',
        'torch.distributed.pipeline',
    ] + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['streamlit_hook.py'],
    excludes=excluded_modules,  # Exclude torch.distributions
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoucherVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoucherVision',
)
