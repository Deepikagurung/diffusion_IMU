@echo off
set mode=%1
set id=%2

if "%mode%"=="dip" (
    echo Finetuning on DIP...
    if exist checkpoints\%id%\finetuned_dip rmdir /s /q checkpoints\%id%\finetuned_dip
    python train.py --module joints --init-from checkpoints\%id%\joints --finetune dip
    python train.py --module poser --init-from checkpoints\%id%\poser --finetune dip
    goto :eof
)

if "%mode%"=="imuposer" (
    echo Finetuning on IMUPoser...
    if exist checkpoints\%id%\finetuned_imuposer rmdir /s /q checkpoints\%id%\finetuned_imuposer
    python train.py --module joints --init-from checkpoints\%id%\finetuned_dip\joints --finetune imuposer
    python train.py --module poser --init-from checkpoints\%id%\finetuned_dip\poser --finetune imuposer
    goto :eof
)

echo Invalid argument. Please specify dip or imuposer
