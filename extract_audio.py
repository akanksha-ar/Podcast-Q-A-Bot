from moviepy import VideoFileClip

video = VideoFileClip("podcast.mp4")
video.audio.write_audiofile("audio.wav")

print("Audio extracted successfully!")
