#converts video to mp3
import os
import subprocess
files = os.listdir('videos')
for file in  files:
    tut_num = file.split(" ")[0].split("#")[1]
    file_name = file.split(".")[0].split("#")[1]
    print(tut_num,file_name)
    subprocess.run([r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",'-i',f"videos/{file}",f"audios/{tut_num}_{file_name}.mp3"])