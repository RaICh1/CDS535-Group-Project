@echo off
setlocal
if not defined WEKA_HOME set "WEKA_HOME=D:\Weka\Weka-3-8-7"
if not exist "%WEKA_HOME%\weka.jar" (
  echo Set WEKA_HOME to your Weka installation directory, then run this file again.
  pause
  exit /b 1
)
set "WEKA_JAVA=%WEKA_HOME%\jre\jre-25.0.2-full\bin\java.exe"
if not exist "%WEKA_JAVA%" set "WEKA_JAVA=java"
"%WEKA_JAVA%" --add-opens=java.base/java.lang=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -Xmx2g -cp "%WEKA_HOME%\weka.jar;%~dp0classes" WekaDemo "%~dp0."
endlocal
