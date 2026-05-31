@echo off
chcp 65001 >nul
title 基金预测系统 - 环境安装

echo ========================================
echo    📈 基金预测系统 v2 - 环境安装
echo ========================================
echo.

:: 检查 Python
echo [1/3] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python 已安装

:: 安装 Python 依赖
echo.
echo [2/3] 安装 Python 依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [警告] 部分依赖安装失败，请手动执行: pip install -r requirements.txt
)
echo [OK] Python 依赖安装完成

:: 检查并安装 Ollama
echo.
echo [3/3] 检查 Ollama 大模型...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 未检测到 Ollama，正在安装...
    echo.
    echo 请手动下载安装 Ollama:
    echo https://ollama.com/download/windows
    echo.
    echo 安装完成后，按任意键继续拉取模型...
    pause >nul
)

:: 拉取 Qwen 模型
echo.
echo 正在拉取 Qwen 2.5 7B 模型（约 4GB，首次需要几分钟）...
ollama pull qwen2.5:7b
if %errorlevel% neq 0 (
    echo [错误] 模型拉取失败
    echo 请手动执行: ollama pull qwen2.5:7b
    pause
    exit /b 1
)

echo.
echo ========================================
echo   ✅ 安装完成！
echo.
echo   启动命令: streamlit run app.py
echo   然后浏览器打开: http://localhost:8501
echo ========================================
pause
