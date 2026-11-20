@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PYTHONIOENCODING=utf-8"
set "GRADIO_ANALYTICS_ENABLED=False"
set "KERAS_BACKEND=jax"
set "JAX_PLATFORMS=cpu"
set "MPLBACKEND=module://matplotlib_inline.backend_inline"
set "KERAS_HOME=%TEMP%\keras-digitos"
set "MPLCONFIGDIR=%TEMP%\mpl-digitos"
if /i "%~1"=="preparar" goto preparar
set "TALLER_PY=.venv\Scripts\python.exe"
if not exist "%TALLER_PY%" if exist ".python-local" set /p TALLER_PY=<".python-local"
if not exist "%TALLER_PY%" (
    echo Falta el entorno de Python. Ejecuta taller.cmd preparar
    exit /b 1
)
if "%~1"=="" goto app
if /i "%~1"=="app" goto app
if /i "%~1"=="entrenar" goto entrenar
if /i "%~1"=="evaluar" goto evaluar
if /i "%~1"=="pruebas" goto pruebas
echo Uso: taller.cmd [app ^| entrenar ^| evaluar ^| pruebas ^| preparar]
exit /b 2

:preparar
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
if not exist ".env" copy /y ".env.example" ".env" >nul
exit /b 0

:app
echo Abre http://127.0.0.1:7860 cuando arranque la aplicacion. Ctrl+C la detiene.
"%TALLER_PY%" -u app.py
exit /b %errorlevel%

:entrenar
"%TALLER_PY%" -u -m src.ejecutar_en_proceso entrenamiento_u4
exit /b %errorlevel%

:evaluar
"%TALLER_PY%" -u -m src.ejecutar_en_proceso evaluacion_agente_u5
set "RESULTADO=%errorlevel%"
if exist "assets\evaluacion_actual.json" "%TALLER_PY%" -m src.actualizar_readme
exit /b %RESULTADO%

:pruebas
set "TALLER_TEST_UI=1"
"%TALLER_PY%" -m unittest discover -s tests -v
exit /b %errorlevel%
