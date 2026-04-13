
import json
import whisper
import os
model = whisper.load_model("small")
audios = os.listdir("audios")

for audio in audios:
    print(audio)
    if("_" in audio):
        num = audio.split("_")[0]
        title = audio.split("_")[1].split(".")[0]
        print(num, title)
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate", 
            word_timestamps=False)
        chunks = []
        for segment in result["segments"]:
            
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            chunks.append({
                "num": num,
                "title": title,
                "start": start,
                "end": end,
                "text": text
            })

            chunks_with_metadata = {"chunks":chunks,text:result["text"]}
            with open (f"jsons/{audio}.json", "w") as f:
                json.dump(chunks_with_metadata, f, indent=4)

