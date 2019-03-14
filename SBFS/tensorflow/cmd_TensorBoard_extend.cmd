@echo off
setlocal enabledelayedexpansion
REM call :main >ball.log.txt 2>&1
call :main 
pause
exit /b

:main


echo python -m tensorboard.main --logdir=./log_extend --host=127.0.0.1
python -m tensorboard.main --logdir=./log_extend --host=127.0.0.1