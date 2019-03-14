@echo off
:: Delayed Expansion will cause variables within a batch file to be expanded at execution time rather than at parse time
:: By default expansion will happen just once, before each line is executed.
:: The !delayed! expansion is performed each time the line is executed, or for each loop in a FOR looping command.
SETLOCAL ENABLEDELAYEDEXPANSION
call :main >cmdlogfile 2>&1
pause
exit /b

:main

SET PATH="D:\MATLAB\R2018a";%PATH%

SET scene_name='sbfs_flip_water_pipes_'
SET out_scene_name='sbfs_flip_water_pipes_'
SET sim_start=10500
:: use small batch size to prevent out ot memory error
SET batch_size=2
SET /A sim_end=sim_start+batch_size-1
SET /A num_batch=(10000-sim_start)/batch_size
SET start_frame=20
SET end_frame=99
SET count=1

echo "scene_name=!scene_name! out_scene_name=!out_scene_name! sim_start=!sim_start! batch_size=!batch_size! sim_end=!sim_end! num_batch=!num_batch! start_frame=!start_frame! end_frame=!end_frame! count=!count!"

:EXEC_POINT
echo "start job"
matlab -nosplash -nodesktop -minimize -logfile matlogfile -r "clear all; clc; close all; compute_streamlines(!scene_name!, !out_scene_name!, !sim_start!, !sim_end!, !start_frame!, !end_frame!); f = fopen( 'sync.txt', 'w' ); fclose(f); quit" 
rem :: run script
rem rem matlab -nosplash -nodesktop -minimize -r "try, run ('visStreamline_water_pipes.m'); end; quit" -logfile matlogfile

rem use sync.txt to prevent multiple matlab application running at the same time(MATLAB do not block)
:SYNC_POINT
if exist "sync.txt" (
    SET /A count=count+1
    SET /A sim_start=sim_start+batch_size
    SET /A sim_end=sim_start+batch_size-1
    del "sync.txt"
    echo !count! !sim_start! !sim_end!
    if !count! gtr !num_batch! (
        exit /b
    ) else ( 
        goto EXEC_POINT
    )
) else (
    echo "job not finished, wating..."
    timeout /t 1 /NOBREAK > NUL
    goto SYNC_POINT
)


