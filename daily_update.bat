@echo off
setlocal enabledelayedexpansion

REM Daily NHL data update script.
REM Runs at ~5 AM local time via Windows Task Scheduler.
REM Updates PBP shot store, current-season Elo ratings, and cached
REM odds from The Odds API so the web app serves fresh stats every day.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "LOG_FILE=%SCRIPT_DIR%daily_update.log"
set "PYTHON=python"

echo =========================================== >> "%LOG_FILE%"
echo Daily update started: %date% %time% >> "%LOG_FILE%"
echo =========================================== >> "%LOG_FILE%"

REM Determine current NHL season start year (October cutoff).
for /f %%i in ('%PYTHON% -c "import datetime; y=datetime.datetime.utcnow().year; m=datetime.datetime.utcnow().month; print(y if m >= 10 else y-1)"') do set "CURR_SEASON=%%i"

%PYTHON% update_pbp_stats.py --season %CURR_SEASON% --stype 2 >> "%LOG_FILE%" 2>&1
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

%PYTHON% update_odds.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] update_odds.py failed at %date% %time% >> "%LOG_FILE%"
) else (
    echo [OK] update_odds.py completed at %date% %time% >> "%LOG_FILE%"
)

echo Daily update finished: %date% %time% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

endlocal
