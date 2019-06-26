@echo off
cls
echo .
echo Did you update the version number in setup.py? Also, don't forget to release on GitHub as well!
echo .
pause
echo.
echo Building...
echo.
python setup.py sdist bdist_wheel
echo.
echo Build complete.
echo.
echo Uploading...
echo.
python -m twine upload dist/*
echo.
echo If no errors were produced, it has been uploaded.
echo.
pause