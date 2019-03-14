@echo off
setlocal enabledelayedexpansion
REM call :main >ball.log.txt 2>&1
call :main 
pause
exit /b

:main

SET count=100

FOR /L %%i IN (1,1,%count%) DO (
   	echo manta sbfs_flip_water_pipes.py
	manta sbfs_flip_water_pipes.py
)