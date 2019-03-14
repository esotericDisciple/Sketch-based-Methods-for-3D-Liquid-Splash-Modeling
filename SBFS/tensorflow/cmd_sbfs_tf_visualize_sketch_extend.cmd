@echo off
setlocal enabledelayedexpansion
REM call :main >ball.log.txt 2>&1
call :main 
pause
exit /b

:main


echo manta sbfs_tf_visualize_sketch_extend.py
manta sbfs_tf_visualize_sketch_extend.py
