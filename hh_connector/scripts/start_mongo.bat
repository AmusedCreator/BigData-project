@echo off
REM Launches mongod.exe, unpacked in db/mongodb/ (recursively searching for mongod.exe)
REM This batch file should be in the scripts/ and run from the project folder (or by double-clicking)
setlocal
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set MONGO_ROOT=%PROJECT_DIR%\db\mongodb
set DATA_DIR=%PROJECT_DIR%\mongo_data

if not exist "%DATA_DIR%" (
	mkdir "%DATA_DIR%"
)

REM Looking for mongod.exe recursively
set MONGOD_PATH=
for /r "%MONGO_ROOT%" %%f in (mongod.exe) do (
	set MONGOD_PATH=%%f
	goto :FOUND_MONGOD
)
echo Error: mongod.exe not found in %MONGO_ROOT%\* .
echo Launch python db/setup_mongo.py and unpack MongoDB in db/mongodb
pause
exit /b 1

:FOUND_MONGOD
echo Found mongod: %MONGOD_PATH%

echo Running mongod with dbpath=%DATA_DIR%
start "MongoDB" "%MONGOD_PATH%" --dbpath "%DATA_DIR%"

echo MongoDB started (in a new window). Database files in %DATA_DIR%
pause