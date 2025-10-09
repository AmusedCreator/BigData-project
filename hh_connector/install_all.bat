@echo off
REM Installs MongoDB locally (in the db/ folder) and Python dependencies
echo === Installing MongoDB in the project folder ===
py db/setup_mongo.py

@if %ERRORLEVEL% NEQ 0 (
	echo Error when installing MongoDB. Check the output above.
	pause
	exit /b 1
)

echo === Installing Python dependencies ===
call scripts\install_requirements.bat

echo === Done ===
pause
