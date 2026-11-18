@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py -3.13"

if /I "%~1"=="unit" goto unit
if /I "%~1"=="fake-server" goto fake_server
if /I "%~1"=="fake-client" goto fake_client

echo Usage:
echo   scripts\run_dsh_smoke.cmd unit
echo   scripts\run_dsh_smoke.cmd fake-server [normal^|missing-usage^|error^|transient^|hang]
echo   scripts\run_dsh_smoke.cmd fake-client
exit /b 2

:unit
%PYTHON% -m unittest discover -s tests -p "test_dsh_smoke.py" -v
exit /b %errorlevel%

:fake_server
set "MODE=%~2"
if "%MODE%"=="" set "MODE=normal"
%PYTHON% -m dsh_smoke.fake_deepseek_server --mode "%MODE%" --capture ".runtime\fake-model-requests.jsonl"
exit /b %errorlevel%

:fake_client
set "PYTHONPATH=%CD%\.deps\deepseek-harness\python\sdk\src;%CD%\.deps\deepseek-harness\python\sdk-runtime\src"
set "DSH_RUNTIME_MODE=node"
%PYTHON% -m dsh_smoke.role_smoke --base-url "http://127.0.0.1:8765/v1" --workspace ".runtime\workspace" --dsh-home ".runtime\dsh-home"
exit /b %errorlevel%
