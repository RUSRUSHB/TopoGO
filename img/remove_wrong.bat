@echo off
setlocal enabledelayedexpansion

REM 设置当前文件夹路径为 E:\Documents\Code\TopoGO\img\rolfsen_all_no_wrong
set "current_folder=E:\Documents\Code\TopoGO\img\rolfsen_all_no_wrong"
set "wrong_folder=E:\Documents\Code\TopoGO\output\error\wrong"

REM 输出调试信息
echo Current folder: %current_folder%
echo Wrong folder: %wrong_folder%

REM 检查wrong文件夹是否存在
if not exist "%wrong_folder%" (
    echo Wrong folder does not exist.
    pause
    exit /b 1
)

REM 遍历wrong文件夹中的文件
for %%e in ("%wrong_folder%\*.png") do (
    set "filename=%%~nxe"
    REM 输出调试信息
    echo Checking file: !filename!
    REM 如果当前文件夹中存在同名文件，则删除
    if exist "%current_folder%\!filename!" (
        del "%current_folder%\!filename!"
        echo Deleted file: !filename!
    )
)

echo Operation completed.
pause
endlocal
