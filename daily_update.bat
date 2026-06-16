@echo off
setlocal enabledelayedexpansion

REM Daily NHL data update script.
REM Runs at ~5 AM local time via Windows Task Scheduler.
REM Updates PBP shot store and current-season Elo ratings so the
REM web app serves fresh stats every day.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "LOG_FILE=%SCRIPT_DIR%daily_update.log"
set "PYTHON=python"

echo =========================================== >> "%LOG_FILE%"
echo Daily update started: %date% %time% >> "%LOG_FILE%"
echo =========================================== >> "%LOG_FILE%"

%PYTHON% update_pbp_stats.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] update_pbp_stats.py failed at %date% %time% >> "%LOG_FILE%"
) else (
    echo [OK] update_pbp_stats.py completed at %date% %time% >> "%LOG_FILE%"
)

%PYTHON% update_elo_ratings.py --current-season >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] update_elo_ratings.py failed at %date% %time% >> "%LOG_FILE%"
) else (
    echo [OK] update_elo_ratings.py completed at %date% %time% >> "%LOG_FILE%"
)

echo Daily update finished: %date% %time% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

endlocal
