import os
import argparse
import math
from openai import OpenAI
from pydub import AudioSegment

def transcribe_audio(input_file, chunk_size_minutes=0, language="en"):
    # Get the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_transcription.txt"
    
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=read_api_key('.openaikey'))
    
    full_transcription = ""
    
    # Determine if we need to chunk the file
    if chunk_size_minutes > 0:
        # Calculate chunk size in milliseconds
        chunk_length_ms = chunk_size_minutes * 60 * 1000
        num_chunks = math.ceil(len(audio) / chunk_length_ms)
        
        print(f"Processing {input_file} in {num_chunks} chunks of {chunk_size_minutes} minutes each")
        
        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            start_time = i * chunk_length_ms
            end_time = min(start_time + chunk_length_ms, len(audio))
            
            # Extract the chunk
            chunk = audio[start_time:end_time]
            
            # Create a temporary file in memory instead of saving to disk
            import io
            chunk_file = io.BytesIO()
            chunk.export(chunk_file, format="mp3")
            chunk_file.seek(0)  # Reset file pointer to beginning
            
            # Transcribe the chunk
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=chunk_file,
                    language=language
                )
                # Append the transcription
                full_transcription += transcription.text + "\n"
                print(f"Chunk {i+1} transcribed successfully.")
            except Exception as e:
                print(f"Error transcribing chunk {i+1}: {str(e)}")
    else:
        # Process the entire file at once
        print(f"Processing {input_file} as a single file")
        
        try:
            with open(input_file, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language
                )
            full_transcription = transcription.text
            print("Transcription completed successfully.")
        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
    
    # Save the transcription
    with open(output_file, "w", encoding="utf-8") as txt_file:
        txt_file.write(full_transcription)
    
    print(f"Transcription saved to {output_file}")
    return full_transcription

def read_api_key(key_file):
    try:
        with open(key_file) as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading API key: {str(e)}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper API")
    parser.add_argument("input_file", help="Path to the audio file to transcribe")
    parser.add_argument("--chunk-size", type=int, default=0, 
                        help="Chunk size in minutes (default: 0, process entire file)")
    parser.add_argument("--language", default="en", 
                        help="Language code for transcription (default: en)")
    
    args = parser.parse_args()
    
    # Run the transcription
    transcribe_audio(args.input_file, args.chunk_size, args.language)