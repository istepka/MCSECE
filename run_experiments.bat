FOR /L %%i IN (0,1,100) DO (
    call  C:\ProgramData\Anaconda3\Scripts\activate.bat 
    C:\ProgramData\Anaconda3\envs\cf\python.exe D:\Research\ecemosp\src\run_experiment.py adult %%i
)
