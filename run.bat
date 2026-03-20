@echo off
title Xenutron Model Training
echo ========================================
echo  Xenutron Model Training
echo ========================================
echo.
echo [INFO] This window will stay open even on errors.
echo [INFO] Press CTRL+C to stop training at any time.
echo.

REM ── Virtual environment check ─────────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo         Please run the following commands first:
    echo           python -m venv .venv
    echo           .venv\Scripts\activate
    echo           pip install -r requirements.txt
    goto :error
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto :error
)

REM ── Directory setup ───────────────────────────────────────────────────────
echo [INFO] Checking directories...
if not exist "logs"        mkdir logs
if not exist "checkpoints" mkdir checkpoints
if not exist "tokenizer"   mkdir tokenizer

REM ── Config check ─────────────────────────────────────────────────────────
if not exist "config.json" (
    echo [ERROR] config.json not found. Please ensure it exists next to run.bat.
    goto :error
)

REM ── Dataset check ────────────────────────────────────────────────────────
if not exist "dataset" (
    echo [ERROR] dataset\ folder not found. Please ensure your datasets are in .\dataset\.
    goto :error
)

REM ── Tokenizer training (only if not already done) ────────────────────────
echo.
if not exist "tokenizer\tokenizer.json" (
    echo ========================================
    echo  Training tokenizer on datasets...
    echo ========================================

    python -c "
from datasets import load_from_disk
import os, sys

texts = []
genz_path = 'dataset/genz_combined_processed'
oasst_path = 'dataset/oasst1_processed'

if os.path.exists(genz_path):
    print('Loading genz dataset...')
    genz = load_from_disk(genz_path)
    texts += [str(x.get('prompt','')) + ' ' + str(x.get('response','')) for x in genz]

if os.path.exists(oasst_path):
    print('Loading oasst1 dataset...')
    oasst = load_from_disk(oasst_path)
    texts += [str(x.get('prompt','')) + ' ' + str(x.get('response','')) for x in oasst]

if not texts:
    print('ERROR: No dataset examples found in dataset/')
    sys.exit(1)

print(f'Writing {len(texts)} examples to temp_train.txt ...')
with open('temp_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(texts))
print('Done.')
"
    if errorlevel 1 (
        echo [ERROR] Failed to prepare tokenizer training data.
        goto :error
    )

    python src\tokenizer\train_tokenizer.py --data_path temp_train.txt --vocab_size 50000 --save_path ./tokenizer --tokenizer_type bpe
    if errorlevel 1 (
        echo [ERROR] Tokenizer training failed.
        if exist temp_train.txt del temp_train.txt
        goto :error
    )

    if exist temp_train.txt del temp_train.txt
    echo [INFO] Tokenizer trained and saved.
)

REM ── Start training ────────────────────────────────────────────────────────
echo.
echo ========================================
echo  Starting model training...
echo  Output is shown here AND saved to:
echo    logs\train.log
echo ========================================
echo.

.venv\Scripts\python.exe -u src\training\main.py --config config.json > logs\train.log 2>&1
type logs\train.log

if errorlevel 1 (
    echo.
    echo [ERROR] Training exited with an error. See logs\train.log above for details.
    goto :error
)

echo.
echo ========================================
echo  Training completed successfully!
echo  Checkpoints saved in: checkpoints\
echo  Full log saved in:    logs\train.log
echo ========================================
pause
exit /b 0

:error
echo.
echo ========================================
echo  TRAINING FAILED - see error above
echo ========================================
pause
exit /b 1
