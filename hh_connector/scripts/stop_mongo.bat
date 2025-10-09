@echo off
REM Trying to terminate processes mongod.exe by name (Windows)
echo Completing the processes mongod.exe (if any)...
tasklist /fi "imagename eq mongod.exe" | findstr /i mongod.exe >nul
if %ERRORLEVEL% EQU 0 (
	taskkill /f /im mongod.exe
	echo Completed.
) else (
	echo Processes mongod.exe not found.
)
pause
