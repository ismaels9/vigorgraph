@echo off

REM Caminho para o diretório do projeto
set PROJECT_DIR=C:\VigorGraphAnalise

REM Caminho para o script do servidor
set SCRIPT=%PROJECT_DIR%\src\servidor.py

REM Caminho para o ambiente virtual
set VENV_DIR=%PROJECT_DIR%\venv

REM Verifica se o diretório do ambiente virtual existe
if not exist %VENV_DIR% (
    echo Criando ambiente virtual...
    python -m venv %VENV_DIR%
)

REM Ativa o ambiente virtual
call %VENV_DIR%\Scripts\activate

REM Instala as dependências se houver um arquivo requirements.txt
if exist %PROJECT_DIR%\requirements.txt (
    echo Instalando dependências...
    pip install -r %PROJECT_DIR%\requirements.txt
)

REM Executa o script do servidor
echo Iniciando o servidor...
python %SCRIPT%

REM Mantém a janela aberta após a execução
pause
