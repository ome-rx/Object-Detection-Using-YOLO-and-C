@echo off
echo Checking for OpenCV DLLs...
for %%G in (opencv_world4100.dll opencv_world4100d.dll) do (
    if not exist "%%G" (
        echo %%G is missing!
    ) else (
        echo %%G found.
    )
)
pause
