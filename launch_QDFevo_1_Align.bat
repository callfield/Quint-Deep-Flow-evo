@echo off
setlocal

rem Quint Deep Flow evo launcher.
rem If QDF_RUNTIME_ROOT is not set, the launcher tries a local runtime first,
rem then %USERPROFILE%\Documents\QDF_portable, then the active Python on PATH.
set "REPO_ROOT=%~dp0"
if not defined QDF_RUNTIME_ROOT set "QDF_RUNTIME_ROOT=%USERPROFILE%\Documents\QDF_portable"
set "QUINT_PORTABLE_BUNDLE_ROOT=%REPO_ROOT%"

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
  "%QDF_APP_PYTHON%" "%REPO_ROOT%app\tools\portable_launch.py" --app QDFevo_1_Align --script "%REPO_ROOT%app\QDFevo_1_Align.py"
) else (
  "%QDF_APP_PYTHON%" "%REPO_ROOT%app\QDFevo_1_Align.py"
)

endlocal
