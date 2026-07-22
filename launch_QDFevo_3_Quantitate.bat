@echo off
setlocal

rem Quint Deep Flow evo launcher.
rem QDFevo_3_Quantitate launcher v2026.07.22.2
rem If QDF_RUNTIME_ROOT is not set, the launcher tries a local runtime first,
rem then %USERPROFILE%\Documents\QDF_portable, then the active Python on PATH.
set "REPO_ROOT=%~dp0"
if not defined QDF_RUNTIME_ROOT set "QDF_RUNTIME_ROOT=%USERPROFILE%\Documents\QDF_portable"
set "QUINT_PORTABLE_BUNDLE_ROOT=%REPO_ROOT%"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%REPO_ROOT%app;%PYTHONPATH%"
set "QDF_QDF2D_NUMBA_WARP=1"
if not defined NUMBA_NUM_THREADS set "NUMBA_NUM_THREADS=8"
set "QUINTDEEPFLOW_IMPORT_QDF1_OMIT=1"
set "QUINTDEEPFLOW_EXPORT_PATCH_IDS=1"

if exist "%REPO_ROOT%rt\py\python.exe" (
  set "QDF_APP_PYTHON=%REPO_ROOT%rt\py\python.exe"
) else if exist "%QDF_RUNTIME_ROOT%\rt\py\python.exe" (
  set "QDF_APP_PYTHON=%QDF_RUNTIME_ROOT%\rt\py\python.exe"
) else (
  set "QDF_APP_PYTHON=python"
)

if exist "%REPO_ROOT%rt\ds\python.exe" (
  set "QUINTDEEPFLOW_DEEPSLICE_PYTHON=%REPO_ROOT%rt\ds\python.exe"
) else if exist "%QDF_RUNTIME_ROOT%\rt\ds\python.exe" (
  set "QUINTDEEPFLOW_DEEPSLICE_PYTHON=%QDF_RUNTIME_ROOT%\rt\ds\python.exe"
)
set "QUINTDEEPFLOW2_DEEPSLICE_PYTHON=%QUINTDEEPFLOW_DEEPSLICE_PYTHON%"

if exist "%QDF_RUNTIME_ROOT%\rt\py" (
  set "PORTABLE_APP_ROOT=%QDF_RUNTIME_ROOT%\rt\py"
  set "PATH=%PORTABLE_APP_ROOT%;%PORTABLE_APP_ROOT%\Scripts;%PORTABLE_APP_ROOT%\DLLs;%PATH%"
)

if exist "%REPO_ROOT%app\tools\portable_launch.py" (
  "%QDF_APP_PYTHON%" "%REPO_ROOT%app\tools\portable_launch.py" --app QDFevo_3_Quantitate --script "%REPO_ROOT%app\QDFevo_3_Quantitate.py"
) else (
  "%QDF_APP_PYTHON%" "%REPO_ROOT%app\QDFevo_3_Quantitate.py"
)

endlocal
