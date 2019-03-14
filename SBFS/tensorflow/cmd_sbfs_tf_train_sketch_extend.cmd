@echo off
setlocal enabledelayedexpansion
REM call :main >ball.log.txt 2>&1
call :main 
pause
exit /b

:main


echo python sbfs_tf_train_sketch_extend.py
python sbfs_tf_train_sketch_extend.py
