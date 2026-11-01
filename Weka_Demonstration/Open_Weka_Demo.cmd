@echo off
setlocal
if not exist "%~dp0classes\WekaDemo.class" goto missing
if not exist "%~dp0data\Ionosphere_development.arff" goto missing
if not defined WEKA_HOME set "WEKA_HOME=D:\Weka\Weka-3-8-7"
if not exist "%WEKA_HOME%\weka.jar" (
  echo Weka was not found. Set WEKA_HOME to your Weka installation folder.
  pause
  exit /b 1
)
set "WEKA_JAVA=%WEKA_HOME%\jre\jre-25.0.2-full\bin\java.exe"
if not exist "%WEKA_JAVA%" set "WEKA_JAVA=java"
"%WEKA_JAVA%" --add-opens=java.base/java.lang=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -Xmx2g -cp "%WEKA_HOME%\weka.jar;%~dp0classes" WekaDemo "%~dp0." 2> "%~dp0weka_startup.log"
if errorlevel 1 (
  type "%~dp0weka_startup.log"
  echo Launch failed. See weka_startup.log in this folder.
  pause
)
exit /b
:missing
echo Required files are missing.
echo Extract ALL files from the ZIP first, then open this launcher.
echo Keep the classes and data folders beside this CMD file.
pause
exit /b 1
