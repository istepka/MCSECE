set list=150 154 26 172 48 54 189 68 73 75 92 99 228 103 106 235 236
(FOR %%i IN (%list%) DO (
    call  C:\ProgramData\Anaconda3\Scripts\activate.bat 
    C:\ProgramData\Anaconda3\envs\cf\python.exe D:\Research\ecemosp\src\run_experiment.py adult %%i
))
